from typing import Union

import torch
import torch.nn as nn

from .utils import make_linear_layers, gru_wrapper


class ObsGRU(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ReLU(inplace=True)

        if env_cfg.len_prop_his == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=64, kernel_size=4),
                activation,
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4),
                activation,
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4),  # (64, 1)
                activation,
                nn.Flatten()
            )
        else:
            raise NotImplementedError

        self.gru = nn.GRU(input_size=64, hidden_size=policy_cfg.obs_gru_hidden_size, num_layers=1)
        self.hidden_state = None

    def inference_forward(self, obs_his):
        # inference forward
        out = self.conv_layers(obs_his.transpose(1, 2))
        out, self.hidden_state = self.gru(out.unsqueeze(0), self.hidden_state)
        return out.squeeze(0)

    def forward(self, obs_his, hidden_state):
        # update forward
        out = gru_wrapper(self.conv_layers.forward, obs_his.transpose(2, 3))
        return self.gru(out, hidden_state)

    def get_hidden_state(self):
        if self.hidden_state is None:
            return None
        return self.hidden_state.detach()

    def reset(self, dones):
        self.hidden_state[:, dones] = 0.


class UNet(nn.Module):
    def __init__(self, input_channel):
        super().__init__()

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.encoder_conv2 = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample (32x16 -> 16x8)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.encoder_conv3 = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample (16x8 -> 8x4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample (8x4 -> 4x2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsample (4x2 -> 8x4)
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Upsample (8x4 -> 16x8)
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # Upsample (16x8 -> 32x16)
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Final output layer
        self.output_conv = nn.Conv2d(16, input_channel, kernel_size=1)  # Output a single channel (depth image)

    def forward(self, x):
        # Encoder
        e1 = self.encoder_conv1(x)
        e2 = self.encoder_conv2(e1)
        e3 = self.encoder_conv3(e2)

        # Bottleneck
        b = self.bottleneck_conv(e3)

        # Decoder
        u3 = self.upconv3(b)
        d3 = torch.cat([u3, e3], dim=1)  # Skip connection
        d3 = self.decoder_conv3(d3)

        u2 = self.upconv2(d3)
        d2 = torch.cat([u2, e2], dim=1)  # Skip connection
        d2 = self.decoder_conv2(d2)

        u1 = self.upconv1(d2)
        d1 = torch.cat([u1, e1], dim=1)  # Skip connection
        d1 = self.decoder_conv1(d1)

        # Final output
        return self.output_conv(d1)


class ReconGRU(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.LeakyReLU()

        self.cnn_depth = nn.Sequential(
            nn.Conv2d(in_channels=env_cfg.len_depth_his, out_channels=16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.gru = nn.GRU(input_size=128 + policy_cfg.obs_gru_hidden_size,
                          hidden_size=policy_cfg.recon_gru_hidden_size,
                          num_layers=2)
        self.hidden_state = None

        self.recon_rough = nn.Sequential(
            nn.Unflatten(1, (8, 8, 4)),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            activation,
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            activation,
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
        )

        self.recon_refine = UNet(input_channel=2)

    def inference_forward(self, depth_his, prop_latent):
        # inference forward
        enc_depth = self.cnn_depth(depth_his)

        # concatenate the two latent vectors
        gru_input = torch.cat([enc_depth, prop_latent], dim=1)
        enc_gru, self.hidden_state = self.gru(gru_input.unsqueeze(0), self.hidden_state)

        # reconstruct
        hmap_rough = self.recon_rough(enc_gru.squeeze(0))
        hmap_refine = self.recon_refine(hmap_rough)
        return hmap_rough, hmap_refine

    def forward(self, depth_his, prop_latent, hidden_state):
        # update forward
        enc_depth = gru_wrapper(self.cnn_depth.forward, depth_his)

        # concatenate the two latent vectors
        gru_input = torch.cat([enc_depth, prop_latent], dim=2)
        enc_gru, hidden_state = self.gru(gru_input, hidden_state)

        # reconstruct
        hmap_rough = gru_wrapper(self.recon_rough.forward, enc_gru)
        hmap_refine = gru_wrapper(self.recon_refine.forward, hmap_rough)
        return hmap_rough, hmap_refine, hidden_state

    def get_hidden_state(self):
        if self.hidden_state is None:
            return None
        return self.hidden_state.detach()

    def reset(self, dones):
        self.hidden_state[:, dones] = 0.


class Mixer(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        predictor_hidden_dim = [128, 64]
        activation = nn.ReLU(inplace=True)

        obs_gru_hidden_size = policy_cfg.obs_gru_hidden_size
        mixer_embed_dim = policy_cfg.transformer_embed_dim
        self.len_latent = policy_cfg.len_latent
        vae_output_dim = (policy_cfg.len_latent
                          + policy_cfg.len_base_vel
                          + policy_cfg.len_latent_feet
                          + policy_cfg.len_latent_body)

        # patch embedding
        self.cnn_scan = nn.Sequential(  # 32, 16
            nn.Conv2d(2, 16, 3, 2, 1),  # 16, 8
            activation,
            nn.Conv2d(16, 32, 3, 2, 1),  # 8, 4
            activation,
            nn.Conv2d(32, mixer_embed_dim, 3, 2, 1),  # 4, 2
            activation,
            nn.AdaptiveAvgPool2d((1, 1)),  # 1, 1
            nn.Flatten()

        )

        # observation embedding
        self.mlp_obs = nn.Sequential(
            nn.Linear(obs_gru_hidden_size, mixer_embed_dim),
            activation,
        )

        self.mixer = make_linear_layers(mixer_embed_dim * 2, mixer_embed_dim * 2, mixer_embed_dim * 2,
                                        activation_func=activation)

        # output VAE
        self.mlp_mu = nn.Sequential(
            nn.Linear(mixer_embed_dim * 2, mixer_embed_dim),
            activation,
            nn.Linear(mixer_embed_dim, vae_output_dim),
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(mixer_embed_dim * 2, mixer_embed_dim),
            activation,
            nn.Linear(mixer_embed_dim, self.len_latent)
        )

        # Ot+1 predictor
        self.predictor = nn.Sequential(
            nn.Linear(vae_output_dim, predictor_hidden_dim[0]),
            activation,
            nn.Linear(predictor_hidden_dim[0], predictor_hidden_dim[1]),
            activation,
            nn.Linear(predictor_hidden_dim[1], env_cfg.n_proprio)
        )

    def forward(self, scan, latent_obs):
        latent_scan = self.cnn_scan(scan)  # (1, 1, 32, 16) -> (1, embed_dim, 8, 4)
        latent_obs = self.mlp_obs(latent_obs)
        out = self.mixer(torch.cat([latent_obs, latent_scan], dim=1))

        # decode the vector
        est_mu = self.mlp_mu(out)
        est_latent = est_mu[..., :self.len_latent]
        est = est_mu[..., self.len_latent:]

        est_logvar = self.mlp_logvar(out)

        # reparameterize and predict O_t+1
        z = self.reparameterize(est_latent, est_logvar)
        ot1 = self.predictor(torch.cat([z, est.detach()], dim=1))

        return est_latent, est, est_logvar, ot1

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class Actor(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        vae_output_dim = (policy_cfg.len_latent
                          + policy_cfg.len_base_vel
                          + policy_cfg.len_latent_feet
                          + policy_cfg.len_latent_body)

        self.actor = make_linear_layers(env_cfg.n_proprio + vae_output_dim,
                                        *policy_cfg.actor_hidden_dims,
                                        env_cfg.num_actions,
                                        activation_func=nn.ELU(),
                                        output_activation=False)

    def forward(self, obs, priv):
        return self.actor(torch.cat([obs, priv], dim=1))


class EstimatorNoRecon(nn.Module):
    is_recurrent = True
    from legged_gym.envs.T1.t1_zju_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.obs_gru = ObsGRU(env_cfg, policy_cfg)
        self.reconstructor = ReconGRU(env_cfg, policy_cfg)
        self.mixer = Mixer(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

    def act(self,
            obs: ActorObs,
            use_estimated_values: Union[bool, torch.Tensor],
            eval_=False,
            **kwargs):  # <-- my mood be like

        # encode history proprio
        latent_obs = self.obs_gru.inference_forward(obs.prop_his)

        # cross-model mixing using transformer
        est_latent, _, _, _ = self.mixer(obs.scan, latent_obs)
        latent_input = torch.cat([est_latent, obs.priv_actor], dim=1)

        # compute action
        mean = self.actor(obs.proprio, latent_input)

        # output action
        if eval_:
            return mean
        else:
            # sample action from distribution
            self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())
            return self.distribution.sample()

    def train_act(
            self,
            obs: ActorObs,
            hidden_states,
            use_estimated_values: torch.Tensor,
    ):  # <-- my mood be like

        # encode history proprio
        obs_enc_hidden_states, _ = hidden_states
        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden_states)

        # cross-model mixing using transformer
        est_latent, est, _, _ = gru_wrapper(self.mixer.forward, obs.scan, latent_obs)
        latent_input = torch.cat([est_latent, obs.priv_actor], dim=2)

        # compute action
        mean = gru_wrapper(self.actor.forward, obs.proprio, latent_input)

        # output action
        self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())

    def reconstruct(self, obs, obs_enc_hidden, *args):
        # encode history proprio
        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden)
        est_latent, est, est_logvar, ot1 = gru_wrapper(self.mixer.forward, obs.scan, latent_obs)
        return None, None, est_latent, est, est_logvar, ot1

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
        new_log_std = torch.log(std * torch.ones_like(self.std.data, device=device))
        self.log_std.data = new_log_std.data

    def get_hidden_state(self):
        return (self.obs_gru.get_hidden_state(),)

    def reset(self, dones):
        self.obs_gru.reset(dones)


class EstimatorGRU(nn.Module):
    is_recurrent = True
    from legged_gym.envs.T1.t1_zju_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.obs_gru = ObsGRU(env_cfg, policy_cfg)
        self.reconstructor = ReconGRU(env_cfg, policy_cfg)
        self.mixer = Mixer(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

    def act(self,
            obs: ActorObs,
            use_estimated_values: Union[bool, torch.Tensor],
            eval_=False,
            **kwargs):  # <-- my mood be like

        # encode history proprio
        latent_obs = self.obs_gru.inference_forward(obs.prop_his)

        # compute reconstruction
        with torch.no_grad():
            recon_rough, recon_refine = self.reconstructor.inference_forward(obs.depth, latent_obs)

        # cross-model mixing using transformer
        if type(use_estimated_values) is torch.Tensor:
            recon_output = torch.where(
                use_estimated_values.unsqueeze(-1).unsqueeze(-1),
                recon_refine,
                obs.scan
            )
            est_latent, est, _, _ = self.mixer(recon_output, latent_obs)
            latent_input = torch.where(
                use_estimated_values,
                torch.cat([est_latent, est.detach()], dim=1),
                torch.cat([est_latent, obs.priv_actor], dim=1)
            )

        elif use_estimated_values:
            est_latent, est, _, _ = self.mixer(recon_refine, latent_obs)
            latent_input = torch.cat([est_latent, est.detach()], dim=1)
        else:
            est_latent, _, _, _ = self.mixer(obs.scan, latent_obs)
            latent_input = torch.cat([est_latent, obs.priv_actor], dim=1)

        # compute action
        mean = self.actor(obs.proprio, latent_input)

        # output action
        if eval_:
            return mean, recon_rough.squeeze(1), recon_refine.squeeze(1)
        else:
            # sample action from distribution
            self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())
            return self.distribution.sample()

    def train_act(
            self,
            obs: ActorObs,
            hidden_states,
            use_estimated_values: torch.Tensor,
    ):  # <-- my mood be like

        # encode history proprio
        obs_enc_hidden_states, recon_hidden_states = hidden_states

        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden_states)

        # compute reconstruction
        with torch.no_grad():
            recon_rough, recon_refine, _ = self.reconstructor(obs.depth, latent_obs, recon_hidden_states)

        # cross-model mixing using transformer
        recon_input = torch.where(
            use_estimated_values.unsqueeze(-1).unsqueeze(-1),
            recon_refine,
            obs.scan
        )
        est_latent, est, _, _ = gru_wrapper(self.mixer.forward, recon_input, latent_obs)
        latent_input = torch.where(
            use_estimated_values,
            torch.cat([est_latent, est.detach()], dim=2),
            torch.cat([est_latent, obs.priv_actor], dim=2)
        )

        # compute action
        mean = gru_wrapper(self.actor.forward, obs.proprio, latent_input)

        # output action
        self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())

    def reconstruct(self, obs, obs_enc_hidden, recon_hidden, use_estimated_values):
        # encode history proprio
        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden)
        recon_rough, recon_refine, _ = self.reconstructor(obs.depth, latent_obs.detach(), recon_hidden)

        recon_input = torch.where(
            use_estimated_values.unsqueeze(-1).unsqueeze(-1),
            recon_refine.detach(),
            obs.scan
        )

        est_latent, est, est_logvar, ot1 = gru_wrapper(self.mixer.forward, recon_input, latent_obs)

        return recon_rough.squeeze(2), recon_refine.squeeze(2), est_latent, est, est_logvar, ot1

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
        new_log_std = torch.log(std * torch.ones_like(self.std.data, device=device))
        self.log_std.data = new_log_std.data

    def get_hidden_state(self):
        return self.obs_gru.get_hidden_state(), self.reconstructor.get_hidden_state()

    def reset(self, dones):
        self.obs_gru.reset(dones)
        self.reconstructor.reset(dones)
