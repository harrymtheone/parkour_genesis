import torch
from torch import nn


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


class RSSM(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        n_stoch = 32
        n_discrete = 32
        n_deterministic = 512
        n_embed = 4096 + 1024
        rssm_period = 5
        hidden_size = 512
        self.initial = 'learned'

        input_dim = n_stoch * n_discrete + rssm_period * env_cfg.num_actions

        self.input_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU()
        )

        self.cell = GRUCell(hidden_size, n_deterministic, True)

        self.img_out_layers = nn.Sequential(
            nn.Linear(n_deterministic, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU()
        )
        self.obs_out_layers = nn.Sequential(
            nn.Linear(n_deterministic + n_embed, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU()
        )

        self.imagined_stats_layer = nn.Linear(hidden_size, n_stoch * n_discrete)
        self.obs_stats_layer = nn.Linear(hidden_size, n_stoch * n_discrete)

        if self.initial == 'learned':
            self.W = nn.Parameter(torch.zeros(1, n_deterministic))
