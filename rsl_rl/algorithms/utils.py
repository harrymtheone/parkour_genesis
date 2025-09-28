import math

import torch
from torch import nn as nn


def symlog(x):
    # This one is for label transformation
    if isinstance(x, float):
        if abs(x) < 1e-8:
            return 0.
        else:
            return math.copysign(1., x) * math.log(abs(x) + 1.0)
    else:
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)  # (batch_size, 1)


def symexp(x):
    # Inverse of symlog
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class SymexpTwoHotLoss(nn.Module):
    def __init__(self, low, high, num_buckets=255, device='cuda'):
        super().__init__()
        self.low = low
        self.high = high
        self.num_buckets = num_buckets
        self.device = device

        # Buckets evenly spaced in symlog space
        self.buckets = torch.linspace(symlog(low), symlog(high), steps=num_buckets, device=device)

    def forward(self, data, target):
        """
        logits: (batch_size, num_buckets) raw outputs from network (before softmax)
        target: (batch_size,) continuous ground truth values (not transformed)
        """

        # 1. Transform target with symlog and convert to soft one-hot vector over buckets
        x = symlog(target).unsqueeze(-1)

        # Find bucket indices just below and above target values
        below = torch.sum(self.buckets <= x, dim=-1, dtype=torch.long) - 1
        above = torch.sum(self.buckets > x, dim=-1, dtype=torch.long)

        # Clamp indices to valid range
        below = torch.clamp(below, 0, self.num_buckets - 1)
        above = torch.clamp(above, 0, self.num_buckets - 1)

        equal = (below == above)

        dist_to_below = torch.where(equal, torch.tensor(1.0, device=self.device), torch.abs(self.buckets[below] - x.squeeze(-1)))
        dist_to_above = torch.where(equal, torch.tensor(1.0, device=self.device), torch.abs(self.buckets[above] - x.squeeze(-1)))
        total_dist = dist_to_below + dist_to_above

        weight_below = dist_to_above / total_dist
        weight_above = dist_to_below / total_dist

        target_soft = (
                nn.functional.one_hot(below, num_classes=self.num_buckets) * weight_below.unsqueeze(-1) +
                nn.functional.one_hot(above, num_classes=self.num_buckets) * weight_above.unsqueeze(-1)
        ).float().squeeze(2)

        # 2. Compute log probabilities from logits using log_softmax for numerical stability
        log_probs = nn.functional.log_softmax(data, dim=-1)

        # 3. Compute soft negative log likelihood loss
        loss = -torch.sum(target_soft * log_probs, dim=-1)  # (batch_size,)

        return loss.mean()


class SymlogMSELoss(nn.Module):
    def __init__(self, tolerance=1e-8, reduction='mean'):
        super().__init__()
        self.tolerance = tolerance
        self.reduction = reduction

    def forward(self, data, target):
        loss = nn.functional.mse_loss(data, symlog(target), reduction=self.reduction)
        return loss.clip(min=self.tolerance)


def masked_MSE(input_, target, mask):
    return ((input_ - target) * mask).square().sum() / (input_.numel() / mask.numel() * mask.sum())


def masked_L1(input_, target, mask):
    return ((input_ - target) * mask).abs().sum() / (input_.numel() / mask.numel() * mask.sum())


def masked_mean(input_, mask):
    return (input_ * mask).sum() / (input_.numel() / mask.numel() * mask.sum())
