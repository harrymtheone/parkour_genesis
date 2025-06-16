from typing import Union

import torch
import torch.nn as nn

from .utils import make_linear_layers, gru_wrapper


class ObsEnc(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        enc_hidden_size = 64

        activation = nn.ReLU(inplace=True)

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 2 * enc_hidden_size),
            activation,
            nn.Linear(2 * enc_hidden_size, enc_hidden_size),
            activation,
        )

    def forward(self, obs_his):
        return self.mlp(obs_his.flatten(1, 2))


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


class Recon(nn.Module):
    def __init__(self, ):
        super().__init__()
        len_his = 2
        gru_in_size = 128 + 64 + 64  # depth, auto-regression, proprio
        gru_hidden_size = 512

        activation = nn.ReLU(inplace=True)

        self.cnn_depth = nn.Sequential(
            nn.Conv2d(in_channels=len_his, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            activation,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            activation,
            nn.Flatten()
        )

        self.cnn_recon_prev = nn.Sequential(  # [1, 32, 16]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 1), stride=(2, 1)),  # [16, 16, 16]
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # [32, 16, 16]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [32, 8, 8]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # [64, 8, 8]
            nn.MaxPool2d(kernel_size=2, stride=2),  # [128, 4, 4]
            activation,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4),  # [128, 1, 1]
            activation,
            nn.Flatten()
        )

        self.mlp = nn.Sequential(
            nn.Linear(gru_in_size, gru_hidden_size),
            activation,
            nn.Linear(gru_hidden_size, gru_hidden_size),
            activation,

            nn.Linear(gru_hidden_size, 3 * 32 * 16),
            activation,
            nn.Unflatten(1, (3, 32, 16)),
        )

        self.recon_rough = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # (16, 32, 16)
            activation,
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (8, 32, 16)
            activation,
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # (8, 32, 16)
            activation,
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # (1, 32, 16)
        )

        self.relu = nn.ReLU(inplace=True)
        self.recon_refine = UNet()

    def forward(self, depth_his, recon_prev, prop_latent):
        enc_depth = self.cnn_depth(depth_his)
        enc_recon_prev = self.cnn_recon_prev(recon_prev)

        enc_gru = self.mlp(torch.cat([enc_depth, enc_recon_prev, prop_latent], dim=1))

        hmap_rough = self.recon_rough(torch.cat([enc_gru, recon_prev], dim=1))
        hmap_refine = self.recon_refine(self.relu(hmap_rough))
        return hmap_rough, hmap_refine


class LocoTransformer(nn.Module):
    def __init__(self, ):
        super().__init__()
        embed_dim = 64
        num_heads = 4
        num_layers = 2  # 2
        dim_latent_obs = 64

        len_latent = 16
        len_base_vel = 3
        len_latent_feet = 16
        len_latent_body = 16

        activation = nn.ReLU(inplace=True)

        # patch embedding
        self.cnn_scan = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=4, stride=4)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # observation embedding
        self.mlp_obs = nn.Sequential(
            nn.Linear(dim_latent_obs, embed_dim),
            activation,
        )

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 33, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.2)

        # transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.),
            num_layers=num_layers
        )

        # output MLP
        self.mlp_out = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            activation,
            nn.Linear(embed_dim, len_latent + len_base_vel + len_latent_feet + len_latent_body),
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
        return self.mlp_out(out)


class Actor(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        len_latent = 16
        len_base_vel = 3
        len_latent_feet = 16
        len_latent_body = 16

        self.actor = make_linear_layers(env_cfg.n_proprio + len_latent + len_base_vel + len_latent_feet + len_latent_body,
                                        *policy_cfg.actor_hidden_dims,
                                        env_cfg.num_actions,
                                        activation_func=nn.ELU())
        self.actor.pop(-1)

    def forward(self, obs, priv):
        if obs.ndim == 2:
            return self.actor(torch.cat([obs, priv], dim=1))

        elif obs.ndim == 3:  # GRU
            n_step = len(obs)
            obs, priv = obs.flatten(0, 1), priv.flatten(0, 1)
            return self.actor(torch.cat([obs, priv], dim=1)).unflatten(0, (n_step, -1))


class Estimator(nn.Module):
    is_recurrent = False

    from legged_gym.envs.T1.t1_zju_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()

        # self.obs_gru = ObsGRU()
        self.obs_enc = ObsEnc(env_cfg.len_prop_his * env_cfg.n_proprio)
        self.reconstructor = Recon()
        self.transformer = LocoTransformer()
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.ones(env_cfg.num_actions))
        self.distribution = None

    def act(self,
            obs: ActorObs,
            use_estimated_values: Union[bool, torch.Tensor],
            eval_=False,
    ):  # <-- my mood be like
        obs = obs.clone()

        # encode history proprio
        latent_obs = self.obs_enc(obs.prop_his)

        # compute reconstruction
        with torch.no_grad():
            recon_rough, recon_refine = self.reconstructor(obs.depth, obs.recon_prev.unsqueeze(1), latent_obs)

        # cross-model mixing using transformer
        est_gt = obs.priv[:, :3+16+16]
        if type(use_estimated_values) is torch.Tensor:
            recon_input = torch.where(
                use_estimated_values.unsqueeze(-1).unsqueeze(-1),
                recon_refine,
                obs.scan.unsqueeze(1)
            )
            latent_est = self.transformer(recon_input, latent_obs)
            latent_input = torch.where(
                use_estimated_values,
                latent_est,
                torch.cat([latent_est[:, :16], est_gt], dim=1)
            )

        elif use_estimated_values:
            latent_est = self.transformer(recon_refine, latent_obs)
            # noise = 0.075 * torch.randn_like(recon_refine)
            # latent_est = self.transformer(recon_refine + noise, latent_obs)
            latent_input = latent_est
        else:
            latent_est = self.transformer(obs.scan.unsqueeze(1), latent_obs)
            latent_input = torch.cat([latent_est[:, :16], est_gt], dim=1)

        # compute action
        mean = self.actor(obs.proprio, latent_input)

        # output action
        if eval_:
            return mean, recon_rough.squeeze(1), recon_refine.squeeze(1), latent_est
        else:
            # sample action from distribution
            self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())
            return self.distribution.sample(), recon_rough.squeeze(1), recon_refine.squeeze(1), latent_est

    def train_act(
            self,
            obs: ActorObs,
            use_estimated_values: Union[bool, torch.Tensor] = True,
            **kwargs
    ):  # <-- my mood be like
        obs = obs.clone()        # encode history proprio

        latent_obs = self.obs_enc(obs.prop_his)

        # compute reconstruction
        with torch.no_grad():
            recon_rough, recon_refine = self.reconstructor(obs.depth, obs.recon_prev.unsqueeze(1), latent_obs)

        # cross-model mixing using transformer
        recon_input = torch.where(
            use_estimated_values.unsqueeze(-1).unsqueeze(-1),
            recon_refine,
            obs.scan.unsqueeze(1)
        )
        latent_est = self.transformer(recon_input, latent_obs)
        latent_input = torch.where(
            use_estimated_values,
            latent_est,
            torch.cat([latent_est[:, :16], obs.priv[:, :3+16+16]], dim=1)
        )

        # compute action
        mean = self.actor(obs.proprio, latent_input)

        # output action
        self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())
        return latent_est

    def reconstruct(self, prop, depth, recon_prev):
        # encode history proprio
        latent_obs = self.obs_enc(prop)
        recon_rough, recon_refine = self.reconstructor(depth, recon_prev.unsqueeze(1), latent_obs)
        return recon_rough.squeeze(1), recon_refine.squeeze(1)

    def reset(self, dones):
        pass

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
