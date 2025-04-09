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

        self.log_std = nn.Parameter(torch.log(train_cfg.policy.init_noise_std * torch.ones(env_cfg.num_actions)))
        self.distribution = None

        # disable args validation for speedup
        torch.distributions.Normal.set_default_validate_args = False


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