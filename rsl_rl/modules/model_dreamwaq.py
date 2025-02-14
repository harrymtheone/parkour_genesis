import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import make_linear_layers


class Actor(nn.Module):
    is_recurrent = False

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()

        activation = nn.ELU()

        # construct actor network
        channel_size = 16

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=2 * channel_size, kernel_size=8, stride=4),
            activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=6, stride=1),
            activation,
            nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=6, stride=1),
            activation,
            nn.Flatten(),
            nn.Linear(8 * channel_size, 4 * channel_size),
            activation,
            nn.Linear(4 * channel_size, env_cfg.num_actions),
        )

        # Action noise
        self.log_std = nn.Parameter(torch.log(policy_cfg.init_noise_std * torch.ones(env_cfg.num_actions)))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def act(self, obs, eval_=False, **kwargs):
        mean = self.net(obs.prop_his.transpose(1, 2))

        if eval_:
            return mean

        self.distribution = Normal(mean, mean * 0. + torch.exp(self.log_std))
        return self.distribution.sample()

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

    def detach_hidden_state(self):
        pass

    def reset(self, dones):
        pass


class Critic(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        channel_size = 16
        activation = nn.ELU()

        self.priv_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.num_critic_obs, out_channels=2 * channel_size, kernel_size=8, stride=4),
            activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=6, stride=1),
            activation,
            nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=6, stride=1),
            activation,
            nn.Flatten()
        )

        self.critic = make_linear_layers(8 * channel_size + env_cfg.n_scan, *train_cfg.critic_hidden_dims, 1,
                                         activation_func=nn.ELU())
        self.critic.pop(-1)

    def evaluate(self, obs):
        priv_latent = self.priv_enc(obs.priv_his.transpose(1, 2))

        return self.critic(torch.cat([priv_latent, obs.scan.flatten(1)], dim=1))
