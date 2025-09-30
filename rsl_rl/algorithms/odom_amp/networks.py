from enum import Enum

import torch
import torch.nn as nn
import torch.utils.data
from torch.distributions import Normal

from rsl_rl.modules.utils import make_linear_layers





class Actor(nn.Module):
    def __init__(self, task_cfg):
        super().__init__()
        env_cfg = task_cfg.env
        policy_cfg = task_cfg.policy
        odom_cfg = task_cfg.odometer

        self.scan_encoder = make_linear_layers(2 * 32 * 16, 256, 128,
                                               activation_func=nn.ELU())

        # belief encoder
        self.gru = nn.GRU(env_cfg.n_proprio + 128 + odom_cfg.estimator_output_dim, policy_cfg.actor_gru_hidden_size, num_layers=1)

        self.actor = make_linear_layers(policy_cfg.actor_gru_hidden_size,
                                        *policy_cfg.actor_hidden_dims,
                                        env_cfg.num_actions,
                                        activation_func=nn.ELU(),
                                        output_activation=False)

        # Action noise
        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions), requires_grad=True)
        self.distribution = None

    def safe_nan_to_num(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    def act(self, proprio, scan, priv, hidden_states, eval_=False, **kwargs):
        proprio = self.safe_nan_to_num(proprio)
        scan = self.safe_nan_to_num(scan)
        priv = self.safe_nan_to_num(priv)

        scan_enc = self.scan_encoder(scan.flatten(2))
        x = torch.cat([proprio, scan_enc, priv], dim=2)

        x, hidden_states = self.gru(x, hidden_states)

        mean = self.actor(x)

        if eval_:
            return mean, hidden_states

        self.distribution = Normal(mean, self.log_std.exp())
        return self.distribution.sample(), hidden_states

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1, keepdim=True)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    def reset_std(self, std):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=self.log_std.device))
        self.log_std.data = new_log_std.data
