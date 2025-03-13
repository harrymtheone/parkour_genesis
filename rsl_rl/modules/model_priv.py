import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import make_linear_layers, gru_wrapper

gru_hidden_size = 128


class ActorGRU(nn.Module):
    is_recurrent = True

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()

        self.gru = nn.GRU(input_size=env_cfg.num_critic_obs, hidden_size=gru_hidden_size, num_layers=1)
        self.hidden_states = None

        self.scan_enc = make_linear_layers(env_cfg.n_scan, 256, 64,
                                           activation_func=activation)
        self.edge_mask_enc = make_linear_layers(env_cfg.n_scan, 256, 64,
                                                activation_func=activation)

        self.actor_backbone = make_linear_layers(env_cfg.num_critic_obs + gru_hidden_size + 64 + 64,
                                                 *policy_cfg.actor_hidden_dims,
                                                 env_cfg.num_actions,
                                                 activation_func=activation,
                                                 output_activation=False)

        # Action noise
        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def act(self, obs, eval_=False, **kwargs):
        # inference forward
        obs_enc, self.hidden_states = self.gru(obs.priv.unsqueeze(0), self.hidden_states)
        scan_enc = self.scan_enc(obs.scan.flatten(1))
        edge_enc = self.edge_mask_enc(obs.edge_mask.flatten(1))
        actor_input = torch.cat([obs.priv, obs_enc.squeeze(0), scan_enc, edge_enc], dim=1)
        mean = self.actor_backbone(actor_input)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, hidden_states, **kwargs):
        obs_enc, _ = self.gru(obs.priv, hidden_states)
        scan_enc = gru_wrapper(self.scan_enc, obs.scan.flatten(2))
        edge_enc = gru_wrapper(self.edge_mask_enc, obs.edge_mask.flatten(2))
        actor_input = torch.cat([obs.priv, obs_enc, scan_enc, edge_enc], dim=2)
        mean = gru_wrapper(self.actor_backbone.forward, actor_input)

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
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        self.hidden_states[:, dones] = 0.


class Critic(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        activation = nn.ELU()

        self.priv_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.num_critic_obs, out_channels=64, kernel_size=6, stride=4),
            activation,
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=2),
            activation,
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1),
            activation,
            nn.Flatten()
        )
        self.scan_enc = make_linear_layers(env_cfg.n_scan, 256, 64,
                                           activation_func=activation)
        self.edge_mask_enc = make_linear_layers(env_cfg.n_scan, 256, 64,
                                                activation_func=activation)
        self.critic = make_linear_layers(128 + 64 + 64, *train_cfg.critic_hidden_dims, 1,
                                         activation_func=nn.ELU(),
                                         output_activation=False)

    def evaluate(self, obs):
        if obs.priv_his.ndim == 3:
            priv_latent = self.priv_enc(obs.priv_his.transpose(1, 2))
            scan_enc = self.scan_enc(obs.scan.flatten(1))
            edge_enc = self.edge_mask_enc(obs.edge_mask.flatten(1))
            return self.critic(torch.cat([priv_latent, scan_enc, edge_enc], dim=1))
        else:
            priv_latent = gru_wrapper(self.priv_enc, obs.priv_his.transpose(2, 3))
            scan_enc = gru_wrapper(self.scan_enc, obs.scan.flatten(2))
            edge_enc = gru_wrapper(self.edge_mask_enc, obs.edge_mask.flatten(2))
            return gru_wrapper(self.critic, torch.cat([priv_latent, scan_enc, edge_enc], dim=2))
