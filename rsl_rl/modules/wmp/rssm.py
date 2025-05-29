import torch
from torch import nn

from .networks import WMEncoder, SequenceModel, Dynamics, WMDecoder
from .utils import UniMixOneHotCategorical
from ..utils import gru_wrapper


class RSSM(nn.Module):
    state_deter: torch.Tensor
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

        # recurrent and stochastic state of world model
        self.register_buffer('state_deter', torch.zeros(env_cfg.num_envs, self.cfg.n_deter, requires_grad=False))
        self.register_buffer('state_stoch', torch.zeros(env_cfg.num_envs, self.cfg.n_stoch, self.cfg.n_discrete, requires_grad=False))

        if self.initial == "learned":
            self.W = nn.Parameter(torch.zeros(1, self.cfg.n_deter))  # TODO: how is this one updated?

        # self.init_model_state(torch.ones(env_cfg.num_envs, dtype=torch.bool))

    def step(self, proprio, depth, prev_actions, is_first_step):
        if torch.any(is_first_step):
            self.reset(is_first_step)

        self.state_deter[:] = self.sequence_model(
            self.state_deter.unsqueeze(0),
            self.state_stoch.unsqueeze(0),
            prev_actions.unsqueeze(0),
        ).squeeze(0)

        post_digits = self.encoder(self.state_deter, proprio, depth)
        self.state_stoch[:] = self.get_dist(post_digits).sample()

    def train_step(self, prop, depth, action_his, state_deter, state_stoch, is_first_step):
        if torch.any(is_first_step):
            state_deter = state_deter.clone()
            state_stoch = state_stoch.clone()
            self.reset(is_first_step.squeeze(2), state_deter=state_deter, state_stoch=state_stoch)

        state_deter_new = self.sequence_model(state_deter[0].unsqueeze(0), state_stoch, action_his)

        prior_digits = gru_wrapper(self.dynamics.forward, state_deter_new)

        post_digits = gru_wrapper(self.encoder.forward, state_deter_new, prop, depth)

        prop, depth, rew = gru_wrapper(self.decoder, state_deter_new, self.get_dist(post_digits).sample())

        return prior_digits, post_digits, prop, depth, rew

    def play_step(self, proprio, depth, prev_actions, is_first_step, sample=True):
        if torch.any(is_first_step):
            self.reset(is_first_step)

        self.state_deter[:] = self.sequence_model(
            self.state_deter.unsqueeze(0),
            self.state_stoch.unsqueeze(0),
            prev_actions.unsqueeze(0),
        ).squeeze(0)

        post_digits = self.encoder(self.state_deter, proprio, depth)

        if sample:
            stoch = self.get_dist(post_digits).sample()
        else:
            stoch = self.get_dist(post_digits).mode

        prop, depth, rew = self.decoder(self.state_deter, stoch)

        if sample:
            self.state_stoch[:] = self.get_dist(post_digits).sample()
        else:
            self.state_stoch[:] = self.get_dist(post_digits).mode

        return {'wm_prop': prop, 'wm_depth': depth, 'wm_rew': rew}

    def reset(self, dones, state_deter=None, state_stoch=None):
        if state_deter is None:
            state_deter = self.state_deter
            state_stoch = self.state_stoch

        if self.initial == "zeros":
            state_deter[dones] = 0.
            state_stoch[dones] = 0.
        elif self.initial == "learned":
            deter = torch.tanh(self.W)
            prior_digits = self.dynamics(deter)
            state_deter[dones] = deter.to(state_deter)
            state_stoch[dones] = self.get_dist(prior_digits).mode.to(state_stoch)
        else:
            raise NotImplementedError(self.initial)

    def get_feature(self, vanilla_dreamer=False):
        if vanilla_dreamer:
            # For DreamerV3
            return torch.cat([self.state_deter, self.state_stoch.flatten(1)], dim=1)
        else:
            # For WMP
            return self.state_deter.clone()

    def get_deter(self):
        return self.state_deter.clone()

    def get_stoch(self):
        return self.state_stoch.clone()

    def get_dist(self, logits):
        return torch.distributions.Independent(
            UniMixOneHotCategorical(logits=logits, unimix_ratio=self.unimix_ratio),
            reinterpreted_batch_ndims=1
        )

    def load(self, state_dict):
        for k, v in state_dict.items():
            if k.endswith('state_stoch'):
                state_dict[k] = self.state_dict()[k]

            elif k.endswith('state_deter'):
                state_dict[k] = self.state_dict()[k]

        self.load_state_dict(state_dict)
