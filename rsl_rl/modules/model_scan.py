import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import make_linear_layers


class Actor(nn.Module):
    is_recurrent = False

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()

        activation = nn.ReLU()

        # construct actor network
        channel_size = 16

        self.prop_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.len_prop_his, out_channels=2 * channel_size, kernel_size=7, stride=4),
            activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=5, stride=1),
            activation,
            nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=5, stride=1),
            activation,
            nn.Flatten()
        )

        self.scan_enc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channel_size, kernel_size=7, stride=4, padding=3),
            activation,
            nn.Conv2d(in_channels=channel_size, out_channels=2 * channel_size, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Flatten()
        )

        self.actor_backbone = nn.Sequential(
            nn.Linear(256, 4 * channel_size),
            activation,
            nn.Linear(4 * channel_size, env_cfg.num_actions),
        )

        # Action noise
        self.log_std = nn.Parameter(torch.log(policy_cfg.init_noise_std * torch.ones(env_cfg.num_actions)))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def act(self, obs, eval_=False, **kwargs):
        prop_enc = self.prop_enc(obs.prop_his)
        scan_enc = self.scan_enc(obs.scan.unsqueeze(1))
        mean = self.actor_backbone(torch.cat((prop_enc, scan_enc), dim=1))

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
    from legged_gym.envs.pdd.pdd_scan_environment import CriticObs

    def __init__(self, env_cfg, train_cfg):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.len_critic_his, out_channels=64, kernel_size=9, stride=4),
            nn.ELU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten()
        )

        self.critic = make_linear_layers(128 * 2 + env_cfg.n_scan, *train_cfg.policy.critic_hidden_dims, 1,
                                         activation_func=nn.ELU())
        self.critic.pop(-1)

    def evaluate(self, obs: CriticObs, masks=None):
        his_enc = self.encoder(obs.priv_his)
        return self.critic(torch.cat([his_enc, obs.scan.flatten(1)], dim=1))
