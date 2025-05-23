import torch
from torch import nn

net = nn.GRU(128, 512)

x = torch.randn(10, 2, 128)
hidden = None


out, hidden_new = net(x, hidden)
