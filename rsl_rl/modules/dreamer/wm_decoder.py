from torch import nn
from .utils import ChannelLayerNorm


class WMDecoder(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()

        self.cnn = nn.Sequential(
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
        )

        self.linear = nn.Sequential(
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
        )


        """
        
        MLP(
  (layers): Sequential(
    (Decoder_linear0): Linear(in_features=1536, out_features=1024, bias=False)
    (Decoder_norm0): LayerNorm((1024,), eps=0.001, elementwise_affine=True)
    (Decoder_act0): SiLU()
    (Decoder_linear1): Linear(in_features=1024, out_features=1024, bias=False)
    (Decoder_norm1): LayerNorm((1024,), eps=0.001, elementwise_affine=True)
    (Decoder_act1): SiLU()
    (Decoder_linear2): Linear(in_features=1024, out_features=1024, bias=False)
    (Decoder_norm2): LayerNorm((1024,), eps=0.001, elementwise_affine=True)
    (Decoder_act2): SiLU()
    (Decoder_linear3): Linear(in_features=1024, out_features=1024, bias=False)
    (Decoder_norm3): LayerNorm((1024,), eps=0.001, elementwise_affine=True)
    (Decoder_act3): SiLU()
    (Decoder_linear4): Linear(in_features=1024, out_features=1024, bias=False)
    (Decoder_norm4): LayerNorm((1024,), eps=0.001, elementwise_affine=True)
    (Decoder_act4): SiLU()
  )
  (mean_layer): ModuleDict(
    (prop): Linear(in_features=1024, out_features=33, bias=True)
  )
)
        
        """


