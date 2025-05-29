import torch
from torch import nn

from .networks import WMEncoder, SequenceModel, Dynamics, WMDecoder
from .utils import UniMixOneHotCategorical
from ..utils import gru_wrapper


class RSSM(nn.Module):
    state_stoch: torch.Tensor

    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        self.cfg = train_cfg.world_model
        self.unimix_ratio = self.cfg.unimix_ratio
        self.initial = self.cfg.state_initial

        self.sequence_model = SequenceModel(env_cfg, self.cfg)

        self.encoder = WMEncoder(env_cfg, self.cfg)
        self.dynamics = Dynamics(self.cfg)
        self.decoder = WMDecoder(env_cfg, train_cfg)

        # stochastic state of world model
        self.register_buffer('state_stoch', torch.zeros(env_cfg.num_envs, self.cfg.n_stoch, self.cfg.n_discrete))

    def step(self, proprio, depth, prev_actions, dones, sample=True):
        if torch.any(dones):
            self.reset(dones)

        state_deter = self.sequence_model.inference_forward(self.state_stoch, prev_actions)

        post_digits = self.encoder(state_deter, proprio, depth)

        if sample:
            self.state_stoch[:] = self.get_dist(post_digits).sample()
        else:
            self.state_stoch[:] = self.get_dist(post_digits).mode

    def train_step(self, prop, depth, action_his, state_deter, state_stoch):
        state_deter_new = self.sequence_model(state_deter, state_stoch, action_his)

        prior_digits = gru_wrapper(self.dynamics.forward, state_deter_new)

        post_digits = gru_wrapper(self.encoder.forward, state_deter_new, prop, depth)

        prop, depth, rew = gru_wrapper(self.decoder, state_deter_new, self.get_dist(post_digits).sample())

        return prior_digits, post_digits, prop, depth, rew

    def reset(self, dones):
        self.sequence_model.init_model_state(dones)

        if self.initial == "zeros":
            self.state_stoch[:] = 0.
        elif self.initial == "learned":
            prior_digits = self.dynamics(self.sequence_model.state_deter)
            self.state_stoch[:] = self.get_dist(prior_digits).mode
        else:
            raise NotImplementedError(self._initial)

    def get_deter(self):
        return self.sequence_model.state_deter

    def get_stoch(self):
        return self.state_stoch

    def get_dist(self, logits):
        return torch.distributions.Independent(
            UniMixOneHotCategorical(logits=logits, unimix_ratio=self.unimix_ratio),
            reinterpreted_batch_ndims=1
        )
