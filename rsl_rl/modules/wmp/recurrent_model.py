import torch
from torch import nn


class RecurrentModel(nn.Module):
    state_logit: torch.Tensor
    state_stoch: torch.Tensor
    state_deter: torch.Tensor

    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        self.n_stoch = 32
        self.n_discrete = 32
        n_deter = 512
        n_embed = 4096 + 1024
        rssm_period = 5
        hidden_size = 512
        self._initial = 'learned'
        self._unimix_ratio = 0.01

        input_dim = self.n_stoch * self.n_discrete + rssm_period * env_cfg.num_actions
        self.imag_in_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU()
        )

        # self.cell = nn.GRUCell(hidden_size, n_deter)
        self.cell = GRUCell(hidden_size, n_deter)

        self.imag_out_layers = nn.Sequential(
            nn.Linear(n_deter, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU()
        )
        self.obs_out_layers = nn.Sequential(
            nn.Linear(n_deter + n_embed, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU()
        )

        self.imag_stats_layer = nn.Linear(hidden_size, self.n_stoch * self.n_discrete)
        self.obs_stats_layer = nn.Linear(hidden_size, self.n_stoch * self.n_discrete)

        # recurrent and stochastic state of world model
        self.register_buffer('state_logit', torch.zeros(env_cfg.num_envs, self.n_stoch, self.n_discrete))
        self.register_buffer('state_stoch', torch.zeros(env_cfg.num_envs, self.n_stoch, self.n_discrete))
        self.register_buffer('state_deter', torch.zeros(env_cfg.num_envs, n_deter))

        if self._initial == "learned":
            self.W = nn.Parameter(torch.zeros(1, n_deter))

        # self.init_model_state(torch.ones(env_cfg.num_envs, dtype=torch.bool))

    def imagination_step(self, prev_actions, sample=True):
        x = torch.cat([self.state_stoch.flatten(1), prev_actions.flatten(1)], dim=1)
        x = self.imag_in_layers(x)

        x, deter = self.cell(x, [self.state_deter])
        self.state_deter[:] = deter[0]

        x = self.imag_out_layers(x)

        logits = self._suff_stats_layer("imag", x)

        if sample:
            self.state_stoch[:] = self.get_dist(logits).sample()
        else:
            self.state_stoch[:] = self.get_dist(logits).mode

    def observation_step(self, obs_enc, sample=True):
        x = torch.cat([self.state_deter, obs_enc], dim=-1)
        x = self.obs_out_layers(x)

        self.state_logit[:] = self._suff_stats_layer("obs", x)

        if sample:
            self.state_stoch[:] = self.get_dist(self.state_logit).sample()
        else:
            self.state_stoch[:] = self.get_dist(self.state_logit).mode

        return self.state_deter

    def init_model_state(self, dones):
        if self._initial == "zeros":
            self.state_logit[dones] = 0.
            self.state_stoch[dones] = 0.
            self.state_deter[dones] = 0.

        elif self._initial == "learned":
            # self.model_state['deter'][dones] = torch.tanh(self.W)
            deter = torch.tanh(self.W)

            x = self.imag_stats_layer(self.imag_out_layers(deter))
            logit = x.repeat(dones.sum(), 1).unflatten(1, (self.n_stoch, self.n_discrete))
            stoch = self.get_dist(logit).mode

            self.state_deter[dones] = deter
            self.state_stoch[dones] = stoch
        else:
            raise NotImplementedError(self._initial)

    def get_dist(self, logit):
        return torch.distributions.Independent(
            OneHotDist(logits=logit, unimix_ratio=self._unimix_ratio), 1)

    def _suff_stats_layer(self, name, x):
        if name == "imag":
            x = self.imag_stats_layer(x)
        elif name == "obs":
            x = self.obs_stats_layer(x)
        else:
            raise NotImplementedError

        return x.unflatten(1, (self.n_stoch, self.n_discrete))


class OneHotDist(torch.distributions.OneHotCategorical):
    def __init__(self, probs=None, logits=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = torch.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    @property
    def mode(self):
        _mode = nn.functional.one_hot(
            torch.argmax(super().logits, dim=-1),
            super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]
