import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import make_linear_layers, gru_wrapper


class OdomTransformer(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        num_heads = 4
        num_layers = 2  # 2
        activation = nn.ReLU(inplace=True)

        transformer_embed_dim = policy_cfg.odom_transformer_embed_dim

        # patch embedding
        self.cnn_depth = nn.Conv2d(in_channels=1, out_channels=transformer_embed_dim, kernel_size=8, stride=8, bias=False)
        self.layer_norm = nn.LayerNorm(transformer_embed_dim)

        # observation embedding
        self.mlp_prop = nn.Sequential(
            nn.Linear(env_cfg.n_proprio, transformer_embed_dim),
            activation,
        )

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 8 * 8, transformer_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.2)

        # transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.),
            num_layers=num_layers
        )

        self.gru = nn.GRU(input_size=transformer_embed_dim * 2, hidden_size=policy_cfg.odom_gru_hidden_size)
        self.hidden_states = None

        # privileged information estimator
        self.estimator = nn.Sequential(
            nn.Linear(transformer_embed_dim * 2, transformer_embed_dim),
            activation,
            nn.Linear(transformer_embed_dim, policy_cfg.estimator_output_dim),
        )

        self.reconstructor = nn.Sequential(
            nn.Linear(transformer_embed_dim * 2, 8 * 8 * 4),
            activation,
            nn.Unflatten(1, (8, 8, 4)),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            activation,
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            activation,
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
        )

    def forward(self, prop, depth):
        # patch embedding
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
        out = torch.cat([token_prop, token_terrain], dim=1)

        out, self.hidden_states = self.gru(out.unsqueeze(1), self.hidden_states)

        # reconstructor
        recon = self.reconstructor(out.squeeze(1))

        # estimator
        priv = self.estimator(out.squeeze(1))

        return recon, priv


class Actor(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.scan_encoder = make_linear_layers(2 * 32 * 16, 256, 128,
                                               activation_func=nn.ELU())

        # belief encoder
        self.gru = nn.GRU(env_cfg.n_proprio + 128 + policy_cfg.estimator_output_dim, policy_cfg.actor_gru_hidden_size, num_layers=1)
        self.hidden_states = None
        self.gate_prop = nn.Linear(policy_cfg.actor_gru_hidden_size, 128)
        self.gate_scan = nn.Sequential(
            nn.Linear(policy_cfg.actor_gru_hidden_size, 1),
            nn.Sigmoid()
        )

        self.actor = make_linear_layers(128,
                                        *policy_cfg.actor_hidden_dims,
                                        env_cfg.num_actions,
                                        activation_func=nn.ELU(),
                                        output_activation=False)

        # Action noise
        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions), requires_grad=True)
        self.distribution = None

    def act(self, obs, scan, priv, use_estimated_values, eval_=False, **kwargs):
        if torch.any(use_estimated_values):
            scan_enc = torch.where(
                use_estimated_values,
                self.scan_encoder(scan.flatten(1)),
                self.scan_encoder(obs.scan.flatten(1))
            )

            x = torch.where(
                use_estimated_values,
                torch.cat([obs.proprio, scan_enc, priv], dim=1),
                torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=1)
            )

        else:
            scan_enc = self.scan_encoder(obs.scan.flatten(1))
            x = torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=1)

        x, self.hidden_states = self.gru(x.unsqueeze(0), self.hidden_states)
        x = x.squeeze(0)

        x = self.gate_prop(x) + scan_enc * self.gate_scan(x)

        mean = self.actor(x)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, scan, priv, hidden_states, use_estimated_values):
        if torch.any(use_estimated_values):
            scan_enc = torch.where(
                use_estimated_values,
                gru_wrapper(self.scan_encoder, scan.flatten(2)),
                gru_wrapper(self.scan_encoder, obs.scan.flatten(2)),
            )

            x = torch.where(
                use_estimated_values,
                torch.cat([obs.proprio, scan_enc, priv], dim=2),
                torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=2),
            )

        else:
            scan_enc = gru_wrapper(self.scan_encoder, obs.scan.flatten(2))
            x = torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=2)

        x, _ = self.gru(x, hidden_states)

        x = gru_wrapper(self.gate_prop, x) + scan_enc * gru_wrapper(self.gate_scan, x)

        mean = gru_wrapper(self.actor, x)

        self.distribution = Normal(mean, torch.exp(self.log_std))

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

    def reset_std(self, std):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=self.log_std.device))
        self.log_std.data = new_log_std.data

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        self.hidden_states[:, dones] = 0.
