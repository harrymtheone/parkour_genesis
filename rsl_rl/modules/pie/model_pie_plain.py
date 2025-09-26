import math

import torch
import torch.nn as nn

from rsl_rl.modules.utils import make_linear_layers, gru_wrapper
from . import Mixer, Actor


class EstimatorPlain(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()
        hidden_size = policy_cfg.estimator_gru_hidden_size
        self.len_latent_z = policy_cfg.len_latent_z
        self.len_latent_hmap = policy_cfg.len_latent_hmap

        self.mlp_vel = nn.Linear(hidden_size, 3)
        self.mlp_z = nn.Linear(hidden_size, self.len_latent_z + self.len_latent_hmap)

        self.ot1_predictor = make_linear_layers(3 + self.len_latent_z + self.len_latent_hmap, 128, env_cfg.n_proprio,
                                                activation_func=activation, output_activation=False)

        self.hmap_recon = make_linear_layers(self.len_latent_hmap, 256, env_cfg.n_scan,
                                             activation_func=activation, output_activation=False)

    def forward(self, gru_out):
        vel = self.mlp_vel(gru_out)
        z = self.mlp_z(gru_out)

        ot1 = self.ot1_predictor(torch.cat([vel, z], dim=1))
        hmap = self.hmap_recon(z[:, -self.len_latent_hmap:])

        return vel, z, ot1, hmap


class PolicyPlain(nn.Module):
    is_recurrent = True
    from legged_gym.envs.T1.t1_pie_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.estimator = Mixer(env_cfg, policy_cfg)
        self.vae = EstimatorPlain(env_cfg, policy_cfg)
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
        with torch.no_grad():
            gru_out = self.estimator.inference_forward(obs.prop_his, obs.depth)
            vel, z = self.vae(gru_out)[:2]

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
        with torch.no_grad():
            gru_out = self.estimator(obs.prop_his, obs.depth, hidden_states)
            vel, z = gru_wrapper(self.vae, gru_out)[:2]

        # compute action
        mean = gru_wrapper(self.actor, obs.proprio, vel, z)

        # sample action from distribution
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp())

    def estimate(
            self,
            obs: ActorObs,
            hidden_states
    ):
        gru_out = self.estimator(obs.prop_his, obs.depth, hidden_states)
        vel, z, ot1, hmap = gru_wrapper(self.vae, gru_out)
        return vel, z, ot1, hmap

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
