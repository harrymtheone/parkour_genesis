# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import get_activation, make_linear_layers


class Actor(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        activation = nn.ELU()

        # construct actor network
        channel_size = 16

        self.prop_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=2 * channel_size, kernel_size=8, stride=4),
            activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=6, stride=1),
            activation,
            nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=6, stride=1),
            activation,
            nn.Flatten()
        )

        self.scan_enc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2 * channel_size, kernel_size=5, stride=2, padding=2),
            activation,
            nn.Conv2d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(in_channels=8 * channel_size, out_channels=4 * channel_size, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Flatten()
        )

        self.actor_backbone = nn.Sequential(
            nn.Linear(2 * 8 * channel_size, 4 * channel_size),
            activation,
            nn.Linear(4 * channel_size, env_cfg.num_actions),
        )

    def forward(self, obs, **kwargs):
        prop_enc = self.prop_enc(obs.proprio.permute((0, 2, 1)))
        scan_enc = self.scan_enc(obs.scan.unsqueeze(1))
        return self.actor_backbone(torch.cat((prop_enc, scan_enc), dim=1))


class ActorCriticRMA(nn.Module):
    is_recurrent = False

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.actor = Actor(env_cfg, policy_cfg)

        # Value function
        self.critic = make_linear_layers(env_cfg.num_critic_obs * env_cfg.len_critic_his + env_cfg.n_scan, *policy_cfg.critic_hidden_dims, 1,
                                         activation_func=nn.ELU())
        self.critic.pop(-1)

        # Action noise
        self.std = nn.Parameter(policy_cfg.init_noise_std * torch.ones(env_cfg.num_actions))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def load(self, state_dict):
        self.load_state_dict(state_dict)

    def reset(self, dones=None):
        pass

    def detach_hidden_states(self):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs, **kwargs):
        mean = self.actor(obs)
        self.distribution = Normal(mean, mean * 0. + self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    def act_inference(self, obs, **kwargs):
        return self.actor(obs)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations.to_tensor())

    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data
