import torch
from torch import nn as nn


class UNetWithEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, transformer_embed_dim):
        super().__init__()

        # Encoder for previous reconstruction
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.encoder_pool1 = nn.MaxPool2d(2)  # Downsample (32x16 -> 16x8)
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.encoder_pool2 = nn.MaxPool2d(2)  # Downsample (16x8 -> 8x4)
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Bottleneck
        self.encoder_pool3 = nn.MaxPool2d(2)  # Downsample (8x4 -> 4x2)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Project transformer output to spatial features
        self.transformer_proj = nn.Sequential(
            nn.Linear(transformer_embed_dim, 128 * 4 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 2))
        )

        # Decoder with skip connections
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsample (4x2 -> 8x4)
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64 + 64 from skip connection
            nn.ReLU(),
        )

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Upsample (8x4 -> 16x8)
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32 + 32 from skip connection
            nn.ReLU(),
        )

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # Upsample (16x8 -> 32x16)
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 16 + 16 from skip connection
            nn.ReLU(),
        )

        # Final output layer
        self.output_conv = nn.Conv2d(16, out_channel, kernel_size=1)

    def forward(self, prev_recon, transformer_output):
        # Encoder path - store features for skip connections
        e1 = self.encoder_conv1(prev_recon)  # (batch, 16, 32, 16)

        e1_pooled = self.encoder_pool1(e1)  # (batch, 16, 16, 8)
        e2 = self.encoder_conv2(e1_pooled)  # (batch, 32, 16, 8)

        e2_pooled = self.encoder_pool2(e2)  # (batch, 32, 8, 4)
        e3 = self.encoder_conv3(e2_pooled)  # (batch, 64, 8, 4)

        e3_pooled = self.encoder_pool3(e3)  # (batch, 64, 4, 2)
        bottleneck = self.bottleneck_conv(e3_pooled)  # (batch, 128, 4, 2)

        # Incorporate transformer features into bottleneck
        transformer_spatial = self.transformer_proj(transformer_output)  # (batch, 128, 4, 2)
        combined_bottleneck = bottleneck + transformer_spatial  # Element-wise addition

        # Decoder path with skip connections
        u3 = self.upconv3(combined_bottleneck)  # (batch, 64, 8, 4)
        d3 = torch.cat([u3, e3], dim=1)  # (batch, 128, 8, 4) - Skip connection
        d3 = self.decoder_conv3(d3)  # (batch, 64, 8, 4)

        u2 = self.upconv2(d3)  # (batch, 32, 16, 8)
        d2 = torch.cat([u2, e2], dim=1)  # (batch, 64, 16, 8) - Skip connection
        d2 = self.decoder_conv2(d2)  # (batch, 32, 16, 8)

        u1 = self.upconv1(d2)  # (batch, 16, 32, 16)
        d1 = torch.cat([u1, e1], dim=1)  # (batch, 32, 32, 16) - Skip connection
        d1 = self.decoder_conv1(d1)  # (batch, 16, 32, 16)

        # Final output
        return self.output_conv(d1)  # (batch, out_channel, 32, 16)


class PrivReconstructor(nn.Module):
    is_recurrent = False

    def __init__(self, n_proprio, embed_dim):
        super().__init__()
        num_heads = 4
        num_layers = 2
        activation = nn.LeakyReLU(inplace=True)

        # proprio embedding
        self.mlp_prop = nn.Sequential(nn.Linear(n_proprio, embed_dim), activation)

        # depth embedding
        self.cnn_depth = nn.Conv2d(in_channels=2, out_channels=embed_dim, kernel_size=8, stride=8, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # position embedding for transformer (prop + depth tokens + prev_recon)
        # prop(1) + depth(8*8=64) + prev_recon(1) = 66 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 8 * 8 + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.2)

        # previous reconstruction embedding
        self.prev_recon_conv = nn.Sequential(
            nn.Conv2d(2, embed_dim, kernel_size=4, stride=4),  # (32, 16) -> (8, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # -> (embed_dim, 1, 1)
        )

        # transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.1),
            num_layers=num_layers
        )

        # refine reconstructor - UNet with encoder for previous reconstruction
        self.recon_refine = UNetWithEncoder(in_channel=2, out_channel=2, transformer_embed_dim=embed_dim)

        # MLP for transformer output to 4D estimation
        self.mlp_est = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            activation,
            nn.Linear(embed_dim // 2, 4)
        )

        # Store previous reconstruction
        self.prev_recon = None

    def forward(self, prop, depth):
        # Get transformer embedding using depth, prop, and previous reconstruction
        transformer_output = self.transformer_forward(prop, depth, self.prev_recon)

        # Generate 4D estimation from transformer output
        est = self.mlp_est(transformer_output)

        # Refine reconstruction using UNet with previous reconstruction encoder
        recon_refine = self.recon_refine(self.prev_recon, transformer_output)

        # Update previous reconstruction for next iteration
        self.prev_recon = recon_refine.detach()

        return recon_refine, est

    def inference_forward(self, prop, depth, **kwargs):
        if self.prev_recon is None:
            self.prev_recon = torch.zeros(prop.size(0), 2, 32, 16, device=prop.device)

        # Get transformer embedding using depth, prop, and previous reconstruction
        transformer_output = self.transformer_forward(prop, depth, self.prev_recon)

        # Generate 4D estimation from transformer output
        est = self.mlp_est(transformer_output)

        # Refine reconstruction using UNet with previous reconstruction encoder
        recon_refine = self.recon_refine(self.prev_recon, transformer_output)

        # Update previous reconstruction for next iteration
        self.prev_recon = recon_refine.detach()

        return recon_refine, est

    def transformer_forward(self, prop, depth, prev_recon):
        batch_size = prop.shape[0]

        # convert proprio to a token
        latent_prop = self.mlp_prop(prop).unsqueeze(1)

        # convert depth to tokens
        latent_depth = self.cnn_depth(depth)  # (batch, embed_dim, 8, 8)
        latent_depth = latent_depth.flatten(2).transpose(1, 2)  # -> (batch, 64, embed_dim)
        latent_depth = self.layer_norm(latent_depth)

        # convert previous reconstruction to a token (handle None case)
        if prev_recon is None:
            device = prop.device
            prev_recon = torch.zeros(batch_size, 2, 32, 16, device=device)
        latent_prev_recon = self.prev_recon_conv(prev_recon)  # -> (batch, embed_dim, 1, 1)
        latent_prev_recon = latent_prev_recon.flatten(2).transpose(1, 2)  # -> (batch, 1, embed_dim)

        # concatenate all tokens
        x = torch.cat([latent_prop, latent_depth, latent_prev_recon], dim=1)

        # Expand position embeddings to match batch size
        pos_embed = self.pos_embed.expand(batch_size, -1, -1)
        x += pos_embed

        # transformer
        out = self.transformer(x)  # -> (batch, total_tokens, embed_dim)

        # Aggregate tokens (you can modify this aggregation strategy)
        return torch.mean(out, dim=1)  # -> (batch, embed_dim)

    def reset(self, dones, **kwargs):
        if self.prev_recon is not None:
            self.prev_recon[dones] = 0.
