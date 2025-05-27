import torch
from torch import nn

from rsl_rl.modules.utils import gru_wrapper
from rsl_rl.modules.wmp.utils import OneHotDist


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

        self.gru = nn.GRU(hidden_size, n_deter)

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
        self.register_buffer('state_deter', torch.zeros(env_cfg.num_envs, n_deter, requires_grad=False))

        if self._initial == "learned":
            self.W = nn.Parameter(torch.zeros(1, n_deter))

        # self.init_model_state(torch.ones(env_cfg.num_envs, dtype=torch.bool))

    def step(self, obs_enc, prev_actions, sample=True):  # called during rollout
        x = torch.cat([self.state_stoch.flatten(1), prev_actions.flatten(1)], dim=1)
        x = self.imag_in_layers(x)
        x, _ = self.gru(x.unsqueeze(0), self.state_deter.unsqueeze(0))
        self.state_deter[:] = x.squeeze(0)

        x = torch.cat([self.state_deter, obs_enc], dim=-1)
        x = self.obs_out_layers(x)

        self.state_logit[:] = self._suff_stats_layer("obs", x)

        if sample:
            self.state_stoch[:] = self.get_dist(self.state_logit).sample()
        else:
            self.state_stoch[:] = self.get_dist(self.state_logit).mode

    def imagination_step(self, state_deter, state_stoch, prev_actions, sample=True):  # called during training
        x = torch.cat([state_stoch.flatten(2), prev_actions.flatten(2)], dim=2)
        x = gru_wrapper(self.imag_in_layers.forward, x)

        state_deter_new, _ = self.gru(x, state_deter[0:1])

        x = gru_wrapper(self.imag_out_layers.forward, state_deter_new)

        logits = self._suff_stats_layer("imag", x.flatten(0, 1)).unflatten(0, x.shape[:2])

        if sample:
            return state_deter_new, self.get_dist(logits).sample()
        else:
            return state_deter_new, self.get_dist(logits).mode

    def observation_step(self, state_deter, obs_enc, sample=True):  # called during training
        x = torch.cat([state_deter, obs_enc], dim=-1)
        x = self.obs_out_layers(x)

        logits = self._suff_stats_layer("obs", x)

        if sample:
            return self.get_dist(logits).sample()
        else:
            return self.get_dist(logits).mode

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
