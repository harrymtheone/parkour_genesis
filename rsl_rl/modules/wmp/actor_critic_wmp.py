import torch
from torch import nn


class ActorWMP(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        activation = nn.ELU()

        self.history_encoder = nn.Sequential(
            nn.Linear(210, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 35),
        )

        self.wm_feature_encoder = nn.Sequential(
            nn.Linear(512, 64),
            activation,
            nn.Linear(64, 64),
            activation,
            nn.Linear(64, 32),
        )

        self.actor = nn.Sequential(
            nn.Linear(70, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, 12),
        )

        self.log_std = nn.Parameter(torch.log(torch.ones(env_cfg.num_actions)))
        self.distribution = None

        # disable args validation for speedup
        torch.distributions.Normal.set_default_validate_args = False

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset_std(self, std, device):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=device))
        self.log_std.data = new_log_std.data

    def get_hidden_state(self):
        return self.obs_gru.get_hidden_state(), self.reconstructor.get_hidden_state()

    def reset(self, dones):
        self.obs_gru.reset(dones)
        self.reconstructor.reset(dones)


class CriticWMP(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        activation = nn.ELU()

        self.wm_feature_encoder = nn.Sequential(
            nn.Linear(512, 64),
            activation,
            nn.Linear(64, 64),
            activation,
            nn.Linear(64, 32),
        )

        self.critic = nn.Sequential(
            nn.Linear(317, 512),
            activation,
            nn.Linear(512, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 1),
        )
