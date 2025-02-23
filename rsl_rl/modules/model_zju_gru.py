from typing import Union

import torch
import torch.nn as nn

from .utils import make_linear_layers

obs_gru_hidden_size = 64
recon_gru_hidden_size = 256

len_latent = 16
len_base_vel = 3
len_latent_feet = 8
len_latent_body = 16
transformer_embed_dim = 64


def gru_wrapper(func, *args):
    n_steps = args[0].size(0)
    rtn = func(*[arg.flatten(0, 1) for arg in args])

    if type(rtn) is tuple:
        return [r.unflatten(0, (n_steps, -1)) for r in rtn]
    else:
        return rtn.unflatten(0, (n_steps, -1))


class ObsGRU(nn.Module):
    def __init__(self, env_cfg):
        super().__init__()
        activation = nn.ReLU(inplace=True)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.len_prop_his, out_channels=16, kernel_size=7, stride=4),
            activation,
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            activation,
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1),  # (8 * channel_size, 1)
            activation,
            nn.Flatten()
        )

        self.gru = nn.GRU(input_size=64, hidden_size=obs_gru_hidden_size, num_layers=1)
        self.hidden_state = None

    def forward(self, obs_his, hidden_state=None):
        if hidden_state is None:
            # inference forward
            out = self.conv_layers(obs_his)
            out, self.hidden_state = self.gru(out.unsqueeze(0), self.hidden_state)
            return out.squeeze(0)
        else:
            # update forward
            out = gru_wrapper(self.conv_layers.forward, obs_his)
            out, _ = self.gru(out, hidden_state)
            return out

    def detach_hidden_state(self):
        if self.hidden_state is None:
            return
        self.hidden_state = self.hidden_state.detach()

    def get_hidden_state(self):
        if self.hidden_state is None:
            return None
        return self.hidden_state.detach()

    def reset(self, dones):
        self.hidden_state[:, dones].zero_()


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        activation = nn.ReLU(inplace=True)

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            activation
        )

        self.encoder_conv2 = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample (32x16 -> 16x8)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation
        )

        self.encoder_conv3 = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample (16x8 -> 8x4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation
        )

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Downsample (8x4 -> 4x2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsample (4x2 -> 8x4)
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            activation
        )

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Upsample (8x4 -> 16x8)
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            activation
        )

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # Upsample (16x8 -> 32x16)
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            activation
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
        self.activation = nn.ReLU(inplace=True)

        self.cnn_depth = nn.Sequential(
            nn.Conv2d(in_channels=env_cfg.len_depth_his, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            self.activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            self.activation,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            self.activation,
            nn.Flatten()
        )

        self.gru = nn.GRU(input_size=128 + obs_gru_hidden_size, hidden_size=recon_gru_hidden_size, num_layers=2)
        self.hidden_state = None

        self.recon_rough = nn.Sequential(
            nn.Linear(recon_gru_hidden_size, 3 * 32 * 16),
            self.activation,
            nn.Unflatten(1, (3, 32, 16)),
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (16, 32, 16)
            self.activation,
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (8, 32, 16)
            self.activation,
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # (8, 32, 16)
            self.activation,
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # (1, 32, 16)
        )

        self.recon_refine = UNet()

    def forward(self, depth_his, prop_latent, hidden_state=None):
        if hidden_state is None:
            # inference forward
            enc_depth = self.cnn_depth(depth_his)

            # concatenate the two latent vectors
            gru_input = torch.cat([enc_depth, prop_latent], dim=1)
            enc_gru, self.hidden_state = self.gru(gru_input.unsqueeze(0), self.hidden_state)

            # reconstruct
            hmap_rough = self.recon_rough(enc_gru.squeeze(0))
            hmap_refine = self.recon_refine(self.activation(hmap_rough))
            return hmap_rough, hmap_refine
        else:
            # update forward
            enc_depth = gru_wrapper(self.cnn_depth.forward, depth_his)

            # concatenate the two latent vectors
            gru_input = torch.cat([enc_depth, prop_latent], dim=2)
            enc_gru, _ = self.gru(gru_input, hidden_state)

            # reconstruct
            hmap_rough = gru_wrapper(self.recon_rough.forward, enc_gru)
            hmap_refine = gru_wrapper(self.recon_refine.forward, self.activation(hmap_rough))
            return hmap_rough, hmap_refine

    def detach_hidden_state(self):
        if self.hidden_state is None:
            return
        self.hidden_state = self.hidden_state.detach()

    def get_hidden_state(self):
        if self.hidden_state is None:
            return None
        return self.hidden_state.detach()

    def reset(self, dones):
        self.hidden_state[:, dones].zero_()


class LocoTransformer(nn.Module):
    def __init__(self, env_cfg):
        super().__init__()
        num_heads = 4
        num_layers = 2  # 2
        predictor_hidden_dim = [128, 64]

        activation = nn.ReLU(inplace=True)

        # patch embedding
        self.cnn_scan = nn.Conv2d(in_channels=1, out_channels=transformer_embed_dim, kernel_size=4, stride=4)
        self.layer_norm = nn.LayerNorm(transformer_embed_dim)

        # observation embedding
        self.mlp_obs = nn.Sequential(
            nn.Linear(obs_gru_hidden_size, transformer_embed_dim),
            activation,
        )

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 33, transformer_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.2)

        # transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.),
            num_layers=num_layers
        )

        # output VAE
        self.mlp_mu = nn.Sequential(
            nn.Linear(transformer_embed_dim * 2, transformer_embed_dim),
            activation,
            nn.Linear(transformer_embed_dim, len_latent + len_base_vel + len_latent_feet + len_latent_body),
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(transformer_embed_dim * 2, transformer_embed_dim),
            activation,
            nn.Linear(transformer_embed_dim, len_latent)
        )

        # Ot+1 predictor
        self.predictor = nn.Sequential(
            nn.Linear(len_latent + len_base_vel + len_latent_feet + len_latent_body, predictor_hidden_dim[0]),
            activation,
            nn.Linear(predictor_hidden_dim[0], predictor_hidden_dim[1]),
            activation,
            nn.Linear(predictor_hidden_dim[1], env_cfg.n_proprio)
        )

    def forward(self, scan, latent_obs):
        # patch embedding
        x = self.cnn_scan(scan)  # (1, 1, 32, 16) -> (1, embed_dim, 8, 4)
        x = x.flatten(2).transpose(1, 2)  # -> (1, 32, embed_dim)
        x = self.layer_norm(x)

        # convert latent_obs to a token
        latent_obs = self.mlp_obs(latent_obs).unsqueeze(1)
        x = torch.cat([latent_obs, x], dim=1)

        # position encoding
        x += self.pos_embed

        # transformer
        out = self.transformer(x)  # -> (1, 33, embed_dim)
        token_prop = out[:, 0]
        token_terrain = torch.mean(out[:, 1:], dim=1)
        out = torch.cat([token_prop, token_terrain], dim=1)

        # decode the vector
        est_mu = self.mlp_mu(out)
        est_logvar = self.mlp_logvar(out)

        # reparameterize and predict O_t+1
        z = self.reparameterize(est_mu[:, :len_latent], est_logvar)
        ot1 = self.predictor(torch.cat((z, est_mu[:, len_latent:]), dim=1))

        # decode the vector
        return est_mu, est_logvar, ot1

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class Actor(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()

        self.actor = make_linear_layers(env_cfg.n_proprio + len_latent + len_base_vel + len_latent_feet + len_latent_body,
                                        *policy_cfg.actor_hidden_dims,
                                        env_cfg.num_actions,
                                        activation_func=nn.ELU())
        self.actor.pop(-1)

    def forward(self, obs, priv):
        if obs.ndim == 2:
            return self.actor(torch.cat([obs, priv], dim=1))

        elif obs.ndim == 3:  # GRU
            return gru_wrapper(self.actor.forward, torch.cat([obs, priv], dim=2))


class EstimatorGRU(nn.Module):
    is_recurrent = True
    from legged_gym.envs.pdd.pdd_scan_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()

        self.obs_gru = ObsGRU(env_cfg)
        self.reconstructor = ReconGRU(env_cfg, policy_cfg)
        self.transformer = LocoTransformer(env_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

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
        est_gt = obs.priv[:, :len_base_vel + len_latent_feet + len_latent_body]
        if type(use_estimated_values) is torch.Tensor:
            recon_input = torch.where(
                use_estimated_values.unsqueeze(-1).unsqueeze(-1),
                recon_refine,
                obs.scan.unsqueeze(1)
            )
            latent_est, _, _ = self.transformer(recon_input, latent_obs)
            latent_input = torch.where(
                use_estimated_values,
                latent_est,
                torch.cat([latent_est[:, :len_latent], est_gt], dim=1)
            )

        elif use_estimated_values:
            latent_input, _, _ = self.transformer(recon_refine, latent_obs)
        else:
            latent_est, _, _ = self.transformer(obs.scan.unsqueeze(1), latent_obs)
            latent_input = torch.cat([latent_est[:, :len_latent], est_gt], dim=1)

        # compute action
        mean = self.actor(obs.proprio, latent_input)

        # output action
        if eval_:
            return mean, recon_refine.squeeze(1)
        else:
            # sample action from distribution
            self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())
            return self.distribution.sample()

    def train_act(
            self,
            obs: ActorObs,
            hidden_states,
            use_estimated_values: Union[bool, torch.Tensor] = True,
    ):  # <-- my mood be like
        obs = obs.clone()  # encode history proprio
        obs_enc_hidden_states, recon_hidden_states = hidden_states

        latent_obs = self.obs_gru(obs.prop_his, obs_enc_hidden_states)

        # compute reconstruction
        with torch.no_grad():
            recon_rough, recon_refine = self.reconstructor(obs.depth, latent_obs, recon_hidden_states)

        # cross-model mixing using transformer
        recon_input = torch.where(
            use_estimated_values.unsqueeze(-1).unsqueeze(-1),
            recon_refine,
            obs.scan.unsqueeze(2)
        )
        latent_est, est_logvar, ot1 = gru_wrapper(self.transformer.forward, recon_input, latent_obs)
        latent_input = torch.where(
            use_estimated_values,
            latent_est,
            torch.cat([latent_est[..., :len_latent],
                       obs.priv[..., :len_base_vel + len_latent_feet + len_latent_body]], dim=-1)
        )

        # compute action
        mean = gru_wrapper(self.actor.forward, obs.proprio, latent_input)

        # output action
        self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())
        return latent_est, est_logvar, ot1

    def reconstruct(self, prop, depth, obs_enc_hidden, recon_hidden):
        # encode history proprio
        latent_obs = self.obs_gru(prop, obs_enc_hidden)
        recon_rough, recon_refine = self.reconstructor(depth, latent_obs, recon_hidden)
        return recon_rough.squeeze(2), recon_refine.squeeze(2)

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

    def detach_hidden_state(self):
        self.obs_gru.detach_hidden_state()
        self.reconstructor.detach_hidden_state()

    def reset(self, dones):
        self.obs_gru.reset(dones)
        self.reconstructor.reset(dones)
