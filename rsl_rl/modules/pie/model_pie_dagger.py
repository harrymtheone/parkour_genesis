import math

import torch
import torch.nn as nn

from rsl_rl.modules.utils import make_linear_layers, gru_wrapper
from . import EstimatorVAE


class PrivEncoderGRU(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()

        self.prop_his_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=16, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4),
            activation,
            nn.Flatten()
        )

        self.scan_enc = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        hidden_size = policy_cfg.estimator_gru_hidden_size
        self.gru = nn.GRU(input_size=128, hidden_size=hidden_size, num_layers=1)
        self.hidden_states = None

    def inference(self, prop_his, scan_edge, **kwargs):
        # inference forward
        prop_latent = self.prop_his_enc(prop_his.transpose(1, 2))
        scan_latent = self.scan_enc(scan_edge)

        gru_input = torch.cat((prop_latent, scan_latent), dim=1)

        # TODO: transformer here?
        gru_out, self.hidden_states = self.gru(gru_input.unsqueeze(0), self.hidden_states)
        return gru_out.squeeze(0)

    def forward(self, prop_his, scan_edge, hidden_states, **kwargs):
        # update forward
        prop_latent = gru_wrapper(self.prop_his_enc.forward, prop_his.transpose(2, 3))

        scan_latent = gru_wrapper(self.scan_enc.forward, scan_edge)

        gru_input = torch.cat((prop_latent, scan_latent), dim=2)

        gru_out, _ = self.gru(gru_input, hidden_states)
        return gru_out

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach()

    def reset(self, dones):
        if self.hidden_states is not None:
            self.hidden_states[:, dones] = 0.


class Actor(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.actor = make_linear_layers(env_cfg.n_proprio + policy_cfg.len_estimation + policy_cfg.len_hmap_latent,
                                        *policy_cfg.actor_hidden_dims, env_cfg.num_actions,
                                        activation_func=nn.ELU(), output_activation=False)

    def forward(self, proprio, vel, z):
        return self.actor(torch.cat([proprio, vel, z], dim=1))


class PolicyDagger(nn.Module):
    is_recurrent = True
    from legged_gym.envs.T1.t1_pie_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.estimator = PrivEncoderGRU(env_cfg, policy_cfg)
        self.vae = EstimatorVAE(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

    def act(
            self,
            obs: ActorObs,
            eval_=False,
            **kwargs
    ):  # <-- my mood be like
        # encode history proprio
        gru_out = self.estimator.inference(obs.prop_his, obs.scan_edge)

        vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, hmap = self.vae(gru_out, sample=not eval_)
        mean = self.actor(obs.proprio, vel, z)

        if eval_:
            return mean, vel

        # sample action from distribution
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp())
        return self.distribution.sample()

    def train_act(
            self,
            obs: ActorObs,
            hidden_states,
    ):  # <-- my mood be like
        gru_out = self.estimator(obs.prop_his, obs.scan_edge, hidden_states)
        vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, hmap = gru_wrapper(self.vae, gru_out)

        # compute action
        mean = gru_wrapper(self.actor, obs.proprio, vel, z)

        # sample action from distribution
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp())
        return vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, hmap

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

    def clip_std(self, min_std: float, max_std: float) -> None:
        self.log_std.data = torch.clamp(self.log_std.data, math.log(min_std), math.log(max_std))

    def get_hidden_states(self):
        return self.estimator.get_hidden_states()

    def detach_hidden_states(self):
        self.estimator.detach_hidden_states()

    def reset(self, dones):
        self.estimator.reset(dones)
