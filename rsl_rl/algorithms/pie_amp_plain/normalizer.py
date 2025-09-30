#  Copyright (c) 2020 Preferred Networks, Inc.
#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, num_repeats=1, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until * num_repeats
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.num_repeats = num_repeats
        self.count = 0

    def state_dict(self):
        data_dict = super().state_dict()
        data_dict.update({'count': self.count})
        return data_dict

    def load_state_dict(self, state_dict):
        if 'count' in state_dict:
            self.count = state_dict['count']
            state_dict.pop('count')
        super().load_state_dict(state_dict)

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        repeat_mean = self._mean.repeat(1,self.num_repeats)
        repeat_std = self._std.repeat(1,self.num_repeats)
        return (x - repeat_mean) / (repeat_std + self.eps)

    # @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until * self.num_repeats:
            return

        x_clone = x.clone()
        batch_size = x_clone.shape[0]
        x_reshaped = x_clone.reshape(batch_size,self.num_repeats,-1).reshape(batch_size*self.num_repeats,-1)
        count_x = x_reshaped.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x_reshaped, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x_reshaped, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    # @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean