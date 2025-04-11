import torch
from torch import nn


class RecurrentModel(nn.Module):
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

        self.cell = nn.GRUCell(hidden_size, n_deter)

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
        self.register_buffer('_logit', torch.zeros(env_cfg.num_envs, self.n_stoch, self.n_discrete))
        self.register_buffer('_stoch', torch.zeros(env_cfg.num_envs, self.n_stoch, self.n_discrete))
        self.register_buffer('_deter', torch.zeros(env_cfg.num_envs, n_deter))
        self.model_state = {'logit': self._logit, 'stoch': self._stoch, 'deter': self._deter}

        if self._initial == "learned":
            self.W = nn.Parameter(torch.zeros(1, n_deter))

        self.init_model_state(torch.ones(env_cfg.num_envs, dtype=torch.bool))

    def observation_step(self, prev_actions, ):
        if self.wm_state is None:
            # initialize all prev_state
            self.wm_state = self._init_wm_state(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(
                self._device
            )

        prev_state = self._init_wm_state(len(is_first))
        prev_action = torch.zeros((len(is_first), self._num_actions)).to(
            self._device
        )

    def init_model_state(self, dones):
        if self._initial == "zeros":
            self.model_state['logit'][dones] = 0.
            self.model_state['stoch'][dones] = 0.
            self.model_state['deter'][dones] = 0.

        elif self._initial == "learned":
            # self.model_state['deter'][dones] = torch.tanh(self.W)
            deter = torch.tanh(self.W)

            x = self.imag_stats_layer(self.imag_out_layers(deter))
            logit = x.repeat(dones.sum(), 1).unflatten(1, (self.n_stoch, self.n_discrete))
            stoch = self.get_dist(logit).mode

            self.model_state['deter'][dones] = deter
            self.model_state['stoch'][dones] = stoch
        else:
            raise NotImplementedError(self._initial)

    def get_dist(self, logit):
        return torch.distributions.Independent(OneHotDist(logits=logit, unimix_ratio=self._unimix_ratio), 1)

    def _suff_stats_layer(self, name, x):
        if name == "imag":
            x = 1
        elif name == "obs":
            x = self._obs_stat_layer(x)
        else:
            raise NotImplementedError
        logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
        return {"logit": logit}


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
