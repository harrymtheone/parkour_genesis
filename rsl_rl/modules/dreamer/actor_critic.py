import torch
from torch import nn


class Actor(nn.Module):
    is_recurrent = False

    def __init__(self, env_cfg):
        super().__init__()
        activation = nn.ELU()

        self.actor = nn.Sequential(
            nn.Linear(1536, 256),
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

    def act(self, state, eval_=False, **kwargs):
        mean = self.actor(state)

        if eval_:
            return mean

        self.distribution = torch.distributions.Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, wm_feature):
        return self.act(obs, wm_feature)

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


class Critic(nn.Module):
    def __init__(self, env_cfg):
        super().__init__()
        activation = nn.ELU()

        self.scan_enc = nn.Sequential(
            nn.Linear(env_cfg.n_scan * 2, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 64),
        )

        self.wm_feature_encoder = nn.Sequential(
            nn.Linear(512, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 64),
        )

        self.critic = nn.Sequential(
            nn.Linear(env_cfg.num_critic_obs + 64 + 64, 512),
            activation,
            nn.Linear(512, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 1),
        )

    def evaluate(self, critic_obs, wm_feature):
        scan = torch.cat([critic_obs.scan, critic_obs.base_edge_mask], dim=2).flatten(1)
        scan_enc = self.scan_enc(scan)

        wm_enc = self.wm_feature_encoder(wm_feature)
        return self.critic(torch.cat([critic_obs.priv, scan_enc, wm_enc], dim=1))
