import torch
from torch import nn

from .utils import ChannelLayerNorm


class WMDecoder(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()

        self.net = nn.ModuleDict({

            "ot1": nn.Sequential(
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
            )
        })

    def forward(self, deter, stoch):
        x = torch.cat([deter, stoch.flatten(1)], dim=1)

        return (
            self.net["ot1"](x),
            self.net["depth"](x),
            self.net["rew"](x),
        )
