import torch
from torch import nn as nn
from torch.distributions import Normal

from rsl_rl.modules.utils import make_linear_layers, gru_wrapper


class Actor(nn.Module):
    def __init__(self, task_cfg):
        super().__init__()
        env_cfg = task_cfg.env
        policy_cfg = task_cfg.policy
        odom_cfg = task_cfg.odometer

        self.scan_encoder = make_linear_layers(2 * 32 * 16, 256, 128,
                                               activation_func=nn.ELU())

        # belief encoder
        self.gru = nn.GRU(env_cfg.n_proprio + 128 + env_cfg.priv_actor_len, policy_cfg.actor_gru_hidden_size, num_layers=1)
        self.hidden_states = None

        self.actor = make_linear_layers(policy_cfg.actor_gru_hidden_size,
                                        *policy_cfg.actor_hidden_dims,
                                        env_cfg.num_actions,
                                        activation_func=nn.ELU(),
                                        output_activation=False)

        # Action noise
        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions), requires_grad=True)
        self.distribution = None

    def act(self, obs, recon, est, use_estimated_values, eval_=False, **kwargs):
        if torch.any(use_estimated_values):
            scan_enc = torch.where(
                use_estimated_values,
                self.scan_encoder(recon.flatten(1)),
                self.scan_encoder(obs.scan.flatten(1))
            )

            x = torch.where(
                use_estimated_values,
                torch.cat([obs.proprio, scan_enc, est], dim=1),
                torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=1)
            )

        else:
            scan_enc = self.scan_encoder(obs.scan.flatten(1))
            x = torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=1)

        x, self.hidden_states = self.gru(x.unsqueeze(0), self.hidden_states)
        x = x.squeeze(0)

        mean = self.actor(x)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, scan, est, hidden_states, use_estimated_values):
        if torch.any(use_estimated_values):
            scan_enc = torch.where(
                use_estimated_values,
                gru_wrapper(self.scan_encoder, scan.flatten(2)),
                gru_wrapper(self.scan_encoder, obs.scan.flatten(2)),
            )

            x = torch.where(
                use_estimated_values,
                torch.cat([obs.proprio, scan_enc, est], dim=2),
                torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=2),
            )

        else:
            scan_enc = gru_wrapper(self.scan_encoder, obs.scan.flatten(2))
            x = torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=2)

        x, _ = self.gru(x, hidden_states)

        mean = gru_wrapper(self.actor, x)

        self.distribution = Normal(mean, torch.exp(self.log_std))

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

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        if self.hidden_states is not None:
            self.hidden_states[:, dones] = 0.


class DepthEstimator(nn.Module):
    def __init__(self, task_cfg):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # (batch, 16, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (batch, 32, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch, 64, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch, 128, 4, 4)
            nn.ReLU(inplace=True),
            nn.Flatten(),  # (batch, 128*4*4)
            nn.Linear(128 * 4 * 4, 128),  # (batch, 128)
            nn.ReLU(inplace=True)
        )

        self.gru = nn.GRU(input_size=128, hidden_size=128)
        self.hidden_states = None

        self.mlp_est = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)
        )

    def inference_forward(self, depth):
        x = self.cnn(depth)
        x, self.hidden_states = self.gru(x.unsqueeze(0), self.hidden_states)
        x = x.squeeze(0)

        return x, self.mlp_est(x)

    def forward(self, depth, hidden_states):
        x = gru_wrapper(self.cnn, depth)
        x, _ = self.gru(x, hidden_states)

        return x, self.mlp_est(x)

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        if self.hidden_states is not None:
            self.hidden_states[:, dones] = 0.


class ActorROA(Actor):
    def act(self, obs, depth_enc, est, use_estimated_values, eval_=False, **kwargs):
        if torch.any(use_estimated_values):
            scan_enc = self.scan_encoder(obs.scan.flatten(1))

            x = torch.where(
                use_estimated_values,
                torch.cat([obs.proprio, depth_enc, est], dim=1),
                torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=1)
            )

        else:
            scan_enc = self.scan_encoder(obs.scan.flatten(1))
            x = torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=1)

        x, self.hidden_states = self.gru(x.unsqueeze(0), self.hidden_states)
        x = x.squeeze(0)

        mean = self.actor(x)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, depth_enc, est, hidden_states, use_estimated_values):
        if torch.any(use_estimated_values):
            scan_enc = gru_wrapper(self.scan_encoder, obs.scan.flatten(2))

            x = torch.where(
                use_estimated_values,
                torch.cat([obs.proprio, depth_enc, est], dim=2),
                torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=2),
            )

        else:
            scan_enc = gru_wrapper(self.scan_encoder, obs.scan.flatten(2))
            x = torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=2)

        x, _ = self.gru(x, hidden_states)

        mean = gru_wrapper(self.actor, x)

        self.distribution = Normal(mean, torch.exp(self.log_std))
