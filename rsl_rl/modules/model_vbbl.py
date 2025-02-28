from typing import Union

import torch
import torch.nn as nn


def gru_wrapper(func, *args):
    n_steps = args[0].size(0)
    rtn = func(*[arg.flatten(0, 1) for arg in args])

    if type(rtn) is tuple:
        return [r.unflatten(0, (n_steps, -1)) for r in rtn]
    else:
        return rtn.unflatten(0, (n_steps, -1))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
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
        self.output_conv = nn.Conv2d(16, 1, kernel_size=1)  # Output a single channel (depth image)

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

        self.obs_enc = nn.Sequential(
            nn.Linear(env_cfg.n_proprio, 64),
            activation,
            nn.Linear(64, 128),
            activation,
            nn.Linear(128, 128),
            activation
        )

        self.cnn_depth = nn.Sequential(
            nn.Conv2d(in_channels=env_cfg.len_depth_his, out_channels=16, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            activation,
            nn.Flatten()
        )

        self.gru = nn.GRU(input_size=256, hidden_size=policy_cfg.recon_gru_hidden_size, num_layers=1)
        self.hidden_states = None

        self.recon_rough = nn.Sequential(
            nn.Linear(policy_cfg.recon_gru_hidden_size, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, 512),
            nn.Unflatten(1, (1, 32, 16)),
        )

        self.recon_refine = UNet()

    def forward(self, proprio, depth_his, hidden_state=None):
        if hidden_state is None:
            # inference forward
            enc_obs = self.obs_enc(proprio)
            enc_depth = self.cnn_depth(depth_his)

            # concatenate the two latent vectors
            gru_input = torch.cat([enc_obs, enc_depth], dim=1)
            enc_gru, self.hidden_states = self.gru(gru_input.unsqueeze(0), self.hidden_states)

            # reconstruct
            hmap_rough = self.recon_rough(enc_gru.squeeze(0))
            hmap_refine = self.recon_refine(hmap_rough)
            return hmap_rough, hmap_refine
        else:
            # update forward
            enc_obs = gru_wrapper(self.obs_enc.forward, proprio)
            enc_depth = gru_wrapper(self.cnn_depth.forward, depth_his)

            # concatenate the two latent vectors
            gru_input = torch.cat([enc_obs, enc_depth], dim=2)
            enc_gru, hidden_state = self.gru(gru_input, hidden_state)

            # reconstruct
            hmap_rough = gru_wrapper(self.recon_rough.forward, enc_gru)
            hmap_refine = gru_wrapper(self.recon_refine.forward, hmap_rough)
            return hmap_rough, hmap_refine, hidden_state

    def detach_hidden_states(self):
        if self.hidden_states is None:
            return
        self.hidden_states = self.hidden_states.detach()

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        self.hidden_states[:, dones].zero_()


class Modulator(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.LeakyReLU()

        self.obs_enc = nn.Sequential(
            nn.Linear(env_cfg.n_proprio, 64),
            activation,
            nn.Linear(64, 128),
            activation,
            nn.Linear(128, 128),
            activation
        )

        self.hmap_enc = nn.Sequential(
            nn.Linear(env_cfg.n_scan, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 128),
            activation
        )

        self.gru = nn.GRU(input_size=256, hidden_size=policy_cfg.modulator_hidden_size, num_layers=1)
        self.hidden_states = None

    def forward(self, proprio, hmap, hidden_state=None):
        if hidden_state is None:
            # inference forward
            enc_obs = self.obs_enc(proprio)
            enc_hmap = self.hmap_enc(hmap)

            # concatenate the two latent vectors
            gru_input = torch.cat([enc_obs, enc_hmap], dim=1)
            enc_gru, self.hidden_states = self.gru(gru_input.unsqueeze(0), self.hidden_states)

            # reconstruct
            hmap_rough = self.recon_rough(enc_gru.squeeze(0))
            hmap_refine = self.recon_refine(hmap_rough)
            return hmap_rough, hmap_refine
        else:
            # update forward
            enc_obs = gru_wrapper(self.obs_enc.forward, proprio)
            enc_depth = gru_wrapper(self.cnn_depth.forward, depth_his)

            # concatenate the two latent vectors
            gru_input = torch.cat([enc_obs, enc_depth], dim=2)
            enc_gru, hidden_state = self.gru(gru_input, hidden_state)

            # reconstruct
            hmap_rough = gru_wrapper(self.recon_rough.forward, enc_gru)
            hmap_refine = gru_wrapper(self.recon_refine.forward, hmap_rough)
            return hmap_rough, hmap_refine, hidden_state

    def detach_hidden_states(self):
        if self.hidden_states is None:
            return
        self.hidden_states = self.hidden_states.detach()

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        self.hidden_states[:, dones].zero_()


class VbblGRU(nn.Module):
    is_recurrent = True
    from legged_gym.envs.pdd.pdd_scan_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()

        self.reconstructor = ReconGRU(env_cfg, policy_cfg)
        self.modulator = Modulator(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.log(policy_cfg.init_noise_std * torch.ones(env_cfg.num_actions)))
        self.distribution = None

    def act(self,
            obs: ActorObs,
            use_estimated_values: Union[bool, torch.Tensor],
            eval_=False,
            ):  # <-- my mood be like
        obs = obs.clone()

        # encode history proprio
        latent_obs = self.obs_gru(obs.prop_his)

        # compute reconstruction
        with torch.no_grad():
            recon_rough, recon_refine = self.reconstructor(obs.depth, latent_obs)

        # cross-model mixing using transformer
        if type(use_estimated_values) is torch.Tensor:
            recon_input = torch.where(
                use_estimated_values.unsqueeze(-1).unsqueeze(-1),
                recon_refine,
                obs.scan.unsqueeze(1)
            )
            est_mu, _, _ = self.transformer(recon_input, latent_obs)
            latent_input = torch.where(
                use_estimated_values,
                est_mu,
                torch.cat([est_mu[:, :self.len_latent], obs.priv_actor], dim=1)
            )

        elif use_estimated_values:
            latent_input, _, _ = self.transformer(recon_refine, latent_obs)
        else:
            est_mu, _, _ = self.transformer(obs.scan.unsqueeze(1), latent_obs)
            latent_input = torch.cat([est_mu[:, :self.len_latent], obs.priv_actor], dim=1)

        # compute action
        mean = self.actor(obs.proprio, latent_input)

        # output action
        if eval_:
            return mean, recon_rough.squeeze(1), recon_refine.squeeze(1)
        else:
            # sample action from distribution
            self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())
            return self.distribution.sample(), recon_refine.squeeze(1)

    def train_act(
            self,
            obs: ActorObs,
            hidden_states,
            use_estimated_values: Union[bool, torch.Tensor] = True,
    ):  # <-- my mood be like
        obs = obs.clone()  # encode history proprio
        recon_hidden_states, blind_hidden_states, policy_hidden_states = hidden_states

        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden_states)

        # compute reconstruction
        with torch.no_grad():
            recon_rough, recon_refine, _ = self.reconstructor(obs.depth, latent_obs, recon_hidden_states)

        # cross-model mixing using transformer
        recon_input = torch.where(
            use_estimated_values.unsqueeze(-1).unsqueeze(-1),
            recon_refine,
            obs.scan.unsqueeze(2)
        )
        est_mu, _, _ = gru_wrapper(self.transformer.forward, recon_input, latent_obs)
        latent_input = torch.where(
            use_estimated_values,
            est_mu,
            torch.cat([est_mu[..., :self.len_latent], obs.priv_actor], dim=-1)
        )

        # compute action
        mean = gru_wrapper(self.actor.forward, obs.proprio, latent_input)

        # output action
        self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())

    def reconstruct(self, obs, obs_enc_hidden, recon_hidden, use_estimated_values):
        # encode history proprio
        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden)
        recon_rough, recon_refine, _ = self.reconstructor(obs.depth, latent_obs, recon_hidden)

        recon_input = torch.where(
            use_estimated_values.unsqueeze(-1).unsqueeze(-1),
            recon_refine.detach(),  # TODO: You should detach the gradient here, otherwise reconstruction fails???
            obs.scan.unsqueeze(2)
        )

        est_mu, est_logvar, ot1 = gru_wrapper(self.transformer.forward, recon_input, latent_obs)
        est_latent = est_mu[..., :self.len_latent]
        est = est_mu[..., self.len_latent:]

        return recon_rough.squeeze(2), recon_refine.squeeze(2), est, est_latent, est_logvar, ot1

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
        return (self.estimator.get_hidden_states(),
                self.blind_policy.get_hidden_states(),
                self.modulator.get_hidden_states())

    def detach_hidden_state(self):
        self.reconstructor.detach_hidden_state()
        self.blind_policy.detach_hidden_state()
        self.modulator.detach_hidden_state()

    def reset(self, dones):
        self.reconstructor.reset(dones)
        self.blind_policy.reset(dones)
        self.modulator.reset(dones)
