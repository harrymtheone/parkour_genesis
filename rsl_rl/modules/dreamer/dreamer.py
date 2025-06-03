import torch
from torch import nn

from . import RSSM, Decoder, Actor, Critic
from ..utils import gru_wrapper


class Dreamer(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()

        self.rssm = RSSM(env_cfg, train_cfg)
        self.decoder = Decoder(env_cfg, train_cfg)

        self.actor = Actor(env_cfg)
        self.critic = Critic(env_cfg)

    def step(self, proprio, depth, prev_actions):
        self.rssm.step(proprio, depth, prev_actions)

        state = torch.cat([self.rssm.state_deter, self.rssm.state_stoch.flatten(1)], dim=1)
        return self.actor.act(state)

    def train_step(self, proprio, depth, prev_actions, state_deter, state_stoch, is_first_step):
        prior_digits, post_digits, state_deter, state_stoch = self.rssm.train_step(proprio, depth, prev_actions, state_deter, state_stoch, is_first_step)
        prop, depth, rew, terminated = gru_wrapper(self.decoder, state_deter, state_stoch)
        return prior_digits, post_digits, prop, depth, rew, terminated

    def get_deter(self):
        return self.rssm.get_deter()

    def get_stoch(self):
        return self.rssm.get_stoch()

    def reset(self, dones):
        self.rssm.reset(dones)
