from torch import nn


class ChannelLayerNorm(nn.Module):
    def __init__(self, channel_size, eps=1e-3, ):
        super().__init__()
        self.norm = nn.LayerNorm(channel_size, eps=eps)

    def forward(self, x):
        assert x.ndim == 4
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
