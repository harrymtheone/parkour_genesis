import torch
from torch import nn

from .utils import ChannelLayerNorm


class WMEncoder(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()

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

    def forward(self, proprio, depth):
        return torch.cat([self.mlp(proprio), self.cnn(depth)], dim=-1)
