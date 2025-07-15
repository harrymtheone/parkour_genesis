import torch
from torch import nn as nn

from rsl_rl.modules.utils import gru_wrapper


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


class OdomRecurrentTransformer(nn.Module):
    is_recurrent = True

    def __init__(self, n_proprio, embed_dim, hidden_size, estimator_out_dim):
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

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 8 * 8, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.2)

        # transformer
        self.transformer = nn.TransformerEncoder(
            # nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.),
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.1),
            num_layers=num_layers
        )

        self.gru = nn.GRU(input_size=embed_dim * 2, hidden_size=hidden_size, num_layers=2)
        self.hidden_states = None
        self.hidden_states_train = None

        # privileged information estimator
        self.estimator = nn.Sequential(
            nn.Linear(embed_dim * 2 + hidden_size, embed_dim),
            activation,
            nn.Linear(embed_dim, estimator_out_dim),
        )

        self.recon_rough = nn.Sequential(
            nn.Linear(embed_dim * 2 + hidden_size, 16 * 8 * 4, bias=False),
            nn.LayerNorm(16 * 8 * 4),
            activation,
            nn.Unflatten(1, (16, 8, 4)),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            activation,
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            activation,
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
        )

        self.recon_refine = UNet(in_channel=1, out_channel=2)

    def inference_forward(self, prop, depth, eval_=True):
        enc = self.transformer_forward(prop, depth)

        if eval_:
            out, self.hidden_states = self.gru(enc.unsqueeze(0), self.hidden_states)
        else:
            out, self.hidden_states_train = self.gru(enc.unsqueeze(0), self.hidden_states_train)

        out = torch.cat([enc, out.squeeze(0)], dim=1)

        # reconstructor
        recon_rough = self.recon_rough(out)
        recon_refine = self.recon_refine(recon_rough)

        # estimator
        est = self.estimator(out)

        return recon_rough, recon_refine, est

    def forward(self, prop, depth, hidden_states):
        enc = gru_wrapper(self.transformer_forward, prop, depth)

        out, _ = self.gru(enc, hidden_states)

        out = torch.cat([enc, out], dim=2)

        # reconstructor
        recon_rough = gru_wrapper(self.recon_rough, out)
        recon_refine = gru_wrapper(self.recon_refine, recon_rough.detach())

        # estimator
        est = gru_wrapper(self.estimator, out)

        return recon_rough, recon_refine, est

    def transformer_forward(self, prop, depth):
        x = self.cnn_depth(depth)  # (1, 1, 64, 64) -> (1, embed_dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # -> (1, 64, embed_dim)
        x = self.layer_norm(x)

        # convert proprio to a token
        latent_prop = self.mlp_prop(prop).unsqueeze(1)
        x = torch.cat([latent_prop, x], dim=1)

        # position encoding
        x += self.pos_embed

        # transformer
        out = self.transformer(x)  # -> (1, 65, embed_dim)
        token_prop = out[:, 0]
        token_terrain = torch.mean(out[:, 1:], dim=1)

        return torch.cat([token_prop, token_terrain], dim=1)

    def detach_hidden_states(self):
        if self.hidden_states_train is not None:
            self.hidden_states_train = self.hidden_states_train.detach()

    def reset(self, dones, eval_=True):
        if eval_:
            if self.hidden_states is not None:
                self.hidden_states[:, dones] = 0.
        else:
            if self.hidden_states_train is not None:
                self.hidden_states_train[:, dones] = 0.
