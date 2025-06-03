import torch
from torch import nn

from rsl_rl.modules.utils import gru_wrapper
from rsl_rl.modules.wmp.utils import ChannelLayerNorm


class SequenceModel(nn.Module):
    def __init__(self, env_cfg, wm_cfg):
        super().__init__()
        input_dim = wm_cfg.n_stoch * wm_cfg.n_discrete + env_cfg.num_actions
        hidden_size = wm_cfg.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU()
        )

        self.gru = nn.GRU(hidden_size, wm_cfg.n_deter)

    def forward(self, state_deter, state_stoch, prev_actions):
        x = gru_wrapper(self.mlp.forward, torch.cat([
            state_stoch.flatten(2),
            prev_actions.flatten(2)
        ], dim=2))

        state_deter_new, _ = self.gru(x, state_deter)
        return state_deter_new


class Encoder(nn.Module):
    def __init__(self, env_cfg, wm_cfg):
        super().__init__()

        self.n_stoch = wm_cfg.n_stoch
        self.n_discrete = wm_cfg.n_discrete

        self.mlp = nn.Sequential(
            nn.Linear(33, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU()
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1, bias=False),
            ChannelLayerNorm(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            ChannelLayerNorm(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            ChannelLayerNorm(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            ChannelLayerNorm(256),
            nn.SiLU(),
            nn.Flatten()
        )

        self.obs_stats_layers = nn.Sequential(
            nn.Linear(wm_cfg.n_deter + wm_cfg.n_mlp_enc + wm_cfg.n_cnn_enc, wm_cfg.hidden_size, bias=False),
            nn.LayerNorm(wm_cfg.hidden_size, eps=1e-3),
            nn.SiLU(),
            nn.Linear(wm_cfg.hidden_size, self.n_stoch * self.n_discrete)
        )

    def forward(self, state_deter, proprio, depth):
        x = torch.cat([state_deter, self.mlp(proprio), self.cnn(depth)], dim=-1)
        return self.obs_stats_layers(x).unflatten(-1, (self.n_stoch, self.n_discrete))


class Dynamics(nn.Module):
    def __init__(self, wm_cfg):
        super().__init__()
        self.n_stoch = wm_cfg.n_stoch
        self.n_discrete = wm_cfg.n_discrete
        hidden_size = wm_cfg.hidden_size

        self.imag_stats_layers = nn.Sequential(
            nn.Linear(wm_cfg.n_deter, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU(),
            nn.Linear(hidden_size, wm_cfg.n_stoch * wm_cfg.n_discrete, bias=True)
        )

    def forward(self, x):
        return self.imag_stats_layers(x).unflatten(1, (self.n_stoch, self.n_discrete))


class Decoder(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()

        self.net = nn.ModuleDict({

            "prop": nn.Sequential(
                nn.Linear(1536, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 33, bias=False),
            ),

            "depth": nn.Sequential(
                nn.Linear(1536, 4096, bias=True),
                nn.Unflatten(1, (256, 4, 4)),

                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                ChannelLayerNorm(128),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                ChannelLayerNorm(64),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
                ChannelLayerNorm(32),
                nn.SiLU(),
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, bias=True),
            ),

            "rew": nn.Sequential(
                nn.Linear(1536, 512, bias=False),
                nn.LayerNorm(512, eps=0.001, elementwise_affine=True),
                nn.SiLU(),
                nn.Linear(512, 512, bias=False),
                nn.LayerNorm(512, eps=0.001, elementwise_affine=True),
                nn.SiLU(),
                nn.Linear(512, 255, bias=True),
            ),

            "termination": nn.Sequential(
                nn.Linear(1536, 512, bias=False),
                nn.LayerNorm(512, eps=0.001, elementwise_affine=True),
                nn.SiLU(),
                nn.Linear(512, 512, bias=False),
                nn.LayerNorm(512, eps=0.001, elementwise_affine=True),
                nn.SiLU(),
                nn.Linear(512, 2, bias=True),
            )
        })

    def forward(self, deter, stoch):
        x = torch.cat([deter, stoch.flatten(1)], dim=1)

        return (
            self.net["prop"](x),
            self.net["depth"](x),
            self.net["rew"](x),
            self.net["termination"](x),
        )
