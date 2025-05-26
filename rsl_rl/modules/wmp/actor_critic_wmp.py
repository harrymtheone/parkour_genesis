import torch
from torch import nn


class ActorWMP(nn.Module):
    is_recurrent = False

    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        activation = nn.ELU()

        self.history_encoder = nn.Sequential(
            nn.Linear(210 + 15, 256),  # instead of remove cmd, I set cmd to zero
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 32),
        )

        self.wm_feature_encoder = nn.Sequential(
            nn.Linear(512, 64),
            activation,
            nn.Linear(64, 64),
            activation,
            nn.Linear(64, 32),
        )

        self.actor = nn.Sequential(
            nn.Linear(env_cfg.n_proprio + 32 + 32, 256),
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

    def act(self, obs, wm_feature, eval_=False, **kwargs):
        his_enc = self.history_encoder(obs.prop_his.flatten(1))
        wm_enc = self.wm_feature_encoder(wm_feature)

        mean = self.actor(torch.cat([obs.proprio, his_enc, wm_enc], dim=1))

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

    def get_hidden_state(self):
        return self.obs_gru.get_hidden_state(), self.reconstructor.get_hidden_state()

    def reset(self, dones):
        self.obs_gru.reset(dones)
        self.reconstructor.reset(dones)


class CriticWMP(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        activation = nn.ELU()

        assert env_cfg.len_critic_his == 50
        self.his_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.num_critic_obs, out_channels=64, kernel_size=8, stride=4),
            activation,
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, stride=1),
            activation,
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, stride=1),
            activation,
            nn.Flatten()
        )

        self.wm_feature_encoder = nn.Sequential(
            nn.Linear(512, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 64),
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 512),
            activation,
            nn.Linear(512, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 1),
        )

    def evaluate(self, critic_obs, wm_feature):
        his_enc = self.his_enc(critic_obs.priv_his.transpose(1, 2))
        wm_enc = self.wm_feature_encoder(wm_feature)
        return self.critic(torch.cat([his_enc, wm_enc], dim=1))
