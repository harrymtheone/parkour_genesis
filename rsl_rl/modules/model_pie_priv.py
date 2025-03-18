import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import make_linear_layers, gru_wrapper


class PrivGRU(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()
        hidden_size = policy_cfg.estimator_gru_hidden_size

        self.gru = nn.GRU(input_size=env_cfg.num_critic_obs + 64 + 64, hidden_size=hidden_size, num_layers=2)
        self.hidden_states = None

        self.scan_enc = make_linear_layers(env_cfg.n_scan, 256, 64, activation_func=activation)
        self.edge_mask_enc = make_linear_layers(env_cfg.n_scan, 256, 64, activation_func=activation)

    def inference_forward(self, priv, scan, edge_mask):
        # inference forward
        scan_enc = self.scan_enc(scan.flatten(1))
        edge_enc = self.edge_mask_enc(edge_mask.flatten(1))
        gru_input = torch.cat([priv, scan_enc, edge_enc], dim=1)

        obs_enc, self.hidden_states = self.gru(gru_input.unsqueeze(0), self.hidden_states)
        return obs_enc.squeeze(0)

    def forward(self, priv, scan, edge_mask, hidden_states, **kwargs):
        scan_enc = gru_wrapper(self.scan_enc, scan.flatten(2))
        edge_enc = gru_wrapper(self.edge_mask_enc, edge_mask.flatten(2))
        gru_input = torch.cat([priv, scan_enc, edge_enc], dim=2)

        obs_enc, _ = self.gru(gru_input, hidden_states)
        return obs_enc

    def get_hidden_states(self):
        return None if self.hidden_states is None else self.hidden_states.detach()

    def reset(self, dones):
        self.hidden_states[:, dones] = 0.


class EstimatorGRU(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()

        self.prop_his_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=32, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4),
            activation,
            nn.Flatten()
        )

        self.depth_enc = nn.Sequential(
            nn.Conv2d(in_channels=env_cfg.len_depth_his, out_channels=16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        hidden_size = policy_cfg.estimator_gru_hidden_size
        self.gru = nn.GRU(input_size=256, hidden_size=hidden_size, num_layers=2)
        self.hidden_states = None

    def inference_forward(self, prop_his, depth_his):
        # inference forward
        prop_latent = self.prop_his_enc(prop_his.transpose(1, 2))
        depth_latent = self.depth_enc(depth_his)

        gru_input = torch.cat((prop_latent, depth_latent), dim=1)
        # TODO: transformer here?
        gru_out, self.hidden_states = self.gru(gru_input.unsqueeze(0), self.hidden_states)
        return gru_out.squeeze(0)

    def forward(self, prop_his, depth_his, hidden_states):
        # update forward
        prop_latent = gru_wrapper(self.prop_his_enc.forward, prop_his.transpose(2, 3))

        depth_latent = gru_wrapper(self.depth_enc.forward, depth_his)

        gru_input = torch.cat((prop_latent, depth_latent), dim=2)
        gru_out, _ = self.gru(gru_input, hidden_states)
        return gru_out

    def get_hidden_states(self):
        return None if self.hidden_states is None else self.hidden_states.detach()

    def reset(self, dones):
        self.hidden_states[:, dones] = 0.


class Policy(nn.Module):
    is_recurrent = True

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()
        hidden_size = policy_cfg.estimator_gru_hidden_size

        self.priv_gru = PrivGRU(env_cfg, policy_cfg)
        self.estimator_gru = EstimatorGRU(env_cfg, policy_cfg)

        self.actor_backbone = make_linear_layers(env_cfg.n_proprio + hidden_size,
                                                 *policy_cfg.actor_hidden_dims,
                                                 env_cfg.num_actions,
                                                 activation_func=activation,
                                                 output_activation=False)

        # Action noise
        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def act(self, obs, obs_critic, use_estimated_values, eval_=False, **kwargs):
        # inference forward
        priv_out = self.priv_gru.inference_forward(obs_critic.priv, obs_critic.scan, obs_critic.edge_mask)
        est_out = self.estimator_gru.inference_forward(obs.prop_his, obs.depth)
        gru_out = torch.where(use_estimated_values, est_out, priv_out)

        actor_input = torch.cat([obs.proprio, gru_out], dim=1)
        mean = self.actor_backbone(actor_input)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, obs_critic, hidden_states, use_estimated_values, **kwargs):
        priv_out = self.priv_gru(obs_critic.priv, obs_critic.scan, obs_critic.edge_mask, hidden_states[0])
        est_out = self.estimator_gru(obs.prop_his, obs.depth, hidden_states[1])
        gru_out = torch.where(use_estimated_values, est_out, priv_out)

        actor_input = torch.cat([obs.proprio, gru_out], dim=2)
        mean = gru_wrapper(self.actor_backbone, actor_input)

        self.distribution = Normal(mean, torch.exp(self.log_std))

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    def reset_std(self, std, device):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=device))
        self.log_std.data = new_log_std.data

    def get_hidden_states(self):
        return self.priv_gru.get_hidden_states(), self.estimator_gru.get_hidden_states()

    def reset(self, dones):
        self.priv_gru.reset(dones)
        self.estimator_gru.reset(dones)
