import math

import torch
import torch.nn as nn

from rsl_rl.modules.utils import make_linear_layers, recurrent_wrapper


class Mixer(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()

        self.prop_his_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=32, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4),
            activation,
            nn.Flatten()
        )

        self.depth_enc = nn.Sequential(
            nn.Conv2d(in_channels=2 * env_cfg.len_depth_his, out_channels=32, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.gru = nn.GRU(input_size=256, hidden_size=policy_cfg.estimator_gru_hidden_size, num_layers=1)

    def forward(self, prop_his, depth_his, hidden_states, **kwargs):
        # update forward
        prop_latent = recurrent_wrapper(self.prop_his_enc, prop_his.transpose(2, 3))

        depth_latent = recurrent_wrapper(self.depth_enc, depth_his)

        mixer_out, hidden_states_new = self.gru(
            torch.cat((prop_latent, depth_latent), dim=-1),
            hidden_states
        )
        return mixer_out, hidden_states_new


class EstimatorVAE(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()
        hidden_size = policy_cfg.estimator_gru_hidden_size
        self.len_latent_z = policy_cfg.len_latent_z
        self.len_latent_hmap = policy_cfg.len_latent_hmap

        self.encoder = make_linear_layers(hidden_size, hidden_size, activation_func=activation)

        self.mlp_vel = nn.Linear(hidden_size, 3)
        self.mlp_vel_logvar = nn.Linear(hidden_size, 3)
        self.mlp_z = nn.Linear(hidden_size, self.len_latent_z + self.len_latent_hmap)
        self.mlp_z_logvar = nn.Linear(hidden_size, self.len_latent_z + self.len_latent_hmap)

        self.ot1_predictor = make_linear_layers(3 + self.len_latent_z + self.len_latent_hmap, 128, env_cfg.n_proprio,
                                                activation_func=activation, output_activation=False)

        self.hmap_recon = make_linear_layers(self.len_latent_hmap, 256, env_cfg.n_scan,
                                             activation_func=activation, output_activation=False)

        self.mlp_vel_logvar.bias.data.fill_(-4.0)
        self.mlp_z_logvar.bias.data.fill_(-4.0)

    def forward(self, mixer_out, sample=True):
        mu_vel = self.mlp_vel(mixer_out)
        logvar_vel = self.mlp_vel_logvar(mixer_out)
        mu_z = self.mlp_z(mixer_out)
        logvar_z = self.mlp_z_logvar(mixer_out)

        vel = self.reparameterize(mu_vel, logvar_vel) if sample else mu_vel
        z = self.reparameterize(mu_z, logvar_z) if sample else mu_z

        ot1 = self.ot1_predictor(torch.cat([vel, z], dim=-1))
        hmap = self.hmap_recon(z[:, :, -self.len_latent_hmap:])

        return vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, hmap

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class Actor(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.actor = make_linear_layers(env_cfg.n_proprio + 3 + policy_cfg.len_latent_z + policy_cfg.len_latent_hmap,
                                        *policy_cfg.actor_hidden_dims, env_cfg.num_actions,
                                        activation_func=nn.ELU(), output_activation=False)

    def forward(self, proprio, vel, z):
        return self.actor(torch.cat([proprio, vel, z], dim=-1))


class Policy(nn.Module):
    is_recurrent = True
    from legged_gym.envs.T1.t1_pie_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.mixer = Mixer(env_cfg, policy_cfg)
        self.vae = EstimatorVAE(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

    def act(self, proprio, prop_his, depth, mixer_hidden_states, eval_=False, **kwargs):  # <-- my mood be like
        # encode history proprio
        with torch.no_grad():
            mixer_out, mixer_hidden_states = self.mixer(prop_his, depth, mixer_hidden_states)
            vel, z = self.vae(mixer_out, sample=not eval_)[:2]

        mean = self.actor(proprio, vel, z)

        mean, vel = mean.squeeze(0), vel.squeeze(0)

        if eval_:
            return mean, vel, mixer_hidden_states

        # sample action from distribution
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp())
        return self.distribution.sample(), mixer_hidden_states

    def train_act(
            self,
            obs: ActorObs,
            hidden_states,
    ):  # <-- my mood be like
        with torch.no_grad():
            mixer_out, _ = self.mixer(obs.prop_his, obs.depth, hidden_states)
            vel, z = self.vae(mixer_out)[:2]

        # compute action
        mean = self.actor(obs.proprio, vel, z)

        # sample action from distribution
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp())

    def estimate(
            self,
            obs: ActorObs,
            hidden_states
    ):
        mixer_out, _ = self.mixer(obs.prop_his, obs.depth, hidden_states)
        vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, hmap = self.vae(mixer_out)
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
        return torch.sum(self.distribution.entropy(), dim=-1, keepdim=True)

    def reset_std(self, std, device):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=device))
        self.log_std.data = new_log_std.data

    def clip_std(self, min_std: float, max_std: float) -> None:
        self.log_std.data = torch.clamp(self.log_std.data, math.log(min_std), math.log(max_std))
