import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import make_linear_layers, gru_wrapper


class ActorGRU(nn.Module):
    is_recurrent = True

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()
        gru_hidden_size = 128

        self.gru = nn.GRU(input_size=env_cfg.num_critic_obs, hidden_size=gru_hidden_size, num_layers=1)
        self.hidden_states = None

        self.scan_enc = make_linear_layers(env_cfg.n_scan, 256, 64, activation_func=activation)
        self.edge_mask_enc = make_linear_layers(env_cfg.n_scan, 256, 64, activation_func=activation)

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

    def act(self, obs, obs_critic, eval_=False, **kwargs):
        # inference forward
        obs_enc, self.hidden_states = self.gru(obs_critic.priv.unsqueeze(0), self.hidden_states)
        scan_enc = self.scan_enc(obs_critic.scan.flatten(1))
        edge_enc = self.edge_mask_enc(obs_critic.edge_mask.flatten(1))
        actor_input = torch.cat([obs_critic.priv, obs_enc.squeeze(0), scan_enc, edge_enc], dim=1)
        mean = self.actor_backbone(actor_input)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, obs_critic, hidden_states, **kwargs):
        obs_enc, _ = self.gru(obs_critic.priv, hidden_states)
        scan_enc = gru_wrapper(self.scan_enc, obs_critic.scan.flatten(2))
        edge_enc = gru_wrapper(self.edge_mask_enc, obs_critic.edge_mask.flatten(2))
        actor_input = torch.cat([obs_critic.priv, obs_enc, scan_enc, edge_enc], dim=2)
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
