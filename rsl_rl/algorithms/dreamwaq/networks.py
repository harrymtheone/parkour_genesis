import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.utils import make_linear_layers

gru_hidden_size = 128
latent_size = 64  # v_t, z_t


class VAE(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()

        self.gru_num_layers = 1
        self.gru = nn.GRU(input_size=env_cfg.n_proprio, hidden_size=gru_hidden_size, num_layers=self.gru_num_layers)
        self.hidden_states = None

        self.mlp_vel_mu = make_linear_layers(gru_hidden_size, 128, 3,
                                             activation_func=activation,
                                             output_activation=False)

        self.mlp_vel_logvar = make_linear_layers(gru_hidden_size, 128, 3,
                                                 activation_func=activation,
                                                 output_activation=False)

        self.mlp_z_mu = make_linear_layers(gru_hidden_size, 128, latent_size,
                                           activation_func=activation,
                                           output_activation=False)

        self.mlp_z_logvar = make_linear_layers(gru_hidden_size, 128, latent_size,
                                               activation_func=activation,
                                               output_activation=False)

        self.decoder = make_linear_layers(3 + latent_size, 64, env_cfg.n_proprio,
                                          activation_func=activation,
                                          output_activation=False)

    def forward(self, proprio, hidden_states=None, sample=True):
        if hidden_states is None:
            obs_enc, self.hidden_states = self.gru(proprio, self.hidden_states)
        else:
            obs_enc, hidden_states = self.gru(proprio, hidden_states)

        mu_vel = self.mlp_vel_mu(obs_enc)
        logvar_vel = self.mlp_vel_logvar(obs_enc)
        mu_z = self.mlp_z_mu(obs_enc)
        logvar_z = self.mlp_z_logvar(obs_enc)

        vel = self.reparameterize(mu_vel, logvar_vel) if sample else mu_vel
        z = self.reparameterize(mu_z, logvar_z) if sample else mu_z

        ot1 = self.decoder(torch.cat([vel, z], dim=-1))

        if hidden_states is None:
            return vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1
        else:
            return vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, hidden_states

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        if self.hidden_states is not None:
            self.hidden_states[:, dones] = 0.


class Actor(nn.Module):
    is_recurrent = True

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.activation = nn.ELU()

        self.actor_backbone = make_linear_layers(3 + latent_size + env_cfg.n_proprio,
                                                 *policy_cfg.actor_hidden_dims,
                                                 env_cfg.num_actions,
                                                 activation_func=self.activation,
                                                 output_activation=False)

        # Action noise
        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def forward(self, proprio, vel, z):
        return self.actor_backbone(torch.cat([proprio, vel, z], dim=-1))

    def act(self, proprio, vel, z, eval_=False, **kwargs):
        mean = self.forward(proprio, vel, z)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

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

    def reset_std(self, std, device):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=device))
        self.log_std.data = new_log_std.data

    def clip_std(self, min_std: float, max_std: float) -> None:
        self.log_std.data = torch.clamp(self.log_std.data, math.log(min_std), math.log(max_std))
