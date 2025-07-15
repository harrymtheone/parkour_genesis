import torch
from torch import nn

from rsl_rl.modules.utils import make_linear_layers


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, padding=1),
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
        self.output_conv = nn.Conv2d(16, out_channel, kernel_size=1)  # Output a single channel (depth image)

        self.enc_conv = nn.Conv2d(192, 128, kernel_size=1, bias=False)

    def forward(self, x, enc):
        # Encoder
        e1 = self.encoder_conv1(x)
        e2 = self.encoder_conv2(e1)
        e3 = self.encoder_conv3(e2)

        # Bottleneck
        b = self.bottleneck_conv(e3) + self.enc_conv(enc[:, :, None, None])

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


class OdomAutoRegressionTransformer(nn.Module):
    is_recurrent = False

    def __init__(self, n_proprio, embed_dim):
        super().__init__()
        num_heads = 4
        num_layers = 2  # 2
        activation = nn.LeakyReLU(inplace=True)

        # patch embedding
        self.cnn_depth = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=8, stride=8, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # observation embedding
        self.mlp_prop = nn.Sequential(
            nn.Linear(n_proprio, embed_dim),
            activation,
        )

        # previous reconstruction embedding
        self.cnn_prev_recon = nn.Conv2d(in_channels=2, out_channels=embed_dim, kernel_size=4, stride=4, bias=False)

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 8 * 8 + 8 * 4, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.2)

        # transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.),
            num_layers=num_layers
        )

        # UNet decoder
        self.unet = UNet(in_channel=2, out_channel=2)

        # priv estimator
        self.mlp_priv_est = make_linear_layers(192, 64, 3,
                                               activation_func=activation,
                                               output_activation=False)

        self.prev_recon = None

    def inference_forward(self, prop, depth, **kwargs):
        if self.prev_recon is None:
            self.prev_recon = torch.zeros(prop.size(0), 2, 32, 16, device=prop.device)

        enc = self.transformer_forward(prop, depth)
        recon = self.unet(self.prev_recon, enc)
        priv_est = self.mlp_priv_est(enc)

        self.prev_recon = recon.detach()
        return torch.zeros_like(recon), recon, priv_est

    def transformer_forward(self, prop, depth):
        # proprio
        latent_prop = self.mlp_prop(prop).unsqueeze(1)

        # depth
        latent_depth = self.cnn_depth(depth)  # (1, 1, 64, 64) -> (1, embed_dim, 8, 8)
        latent_depth = latent_depth.flatten(2).transpose(1, 2)  # -> (1, 64, embed_dim)
        latent_depth = self.layer_norm(latent_depth)

        # previous reconstruction
        latent_recon = self.cnn_prev_recon(self.prev_recon)
        latent_recon = latent_recon.flatten(2).transpose(1, 2)  # -> (1, 32, embed_dim)

        x = torch.cat([latent_prop, latent_depth, latent_recon], dim=1)

        # position encoding
        x += self.pos_embed

        # transformer
        out = self.transformer(x)  # -> (1, 97, embed_dim)
        token_prop = out[:, 0]
        token_depth = torch.mean(out[:, 1: 1 + 64], dim=1)
        token_recon = torch.mean(out[:, -32:], dim=1)
        return torch.cat([token_prop, token_depth, token_recon], dim=1)

    def reset(self, dones, **kwargs):
        self.prev_recon[dones] = 0.
