import torch
from torch import nn

from rsl_rl.modules.utils import gru_wrapper
from rsl_rl.modules.wmp.utils import ChannelLayerNorm


class WMEncoder(nn.Module):
    def __init__(self, env_cfg, wm_cfg):
        super().__init__()

        self.n_stoch = wm_cfg.n_stoch
        self.n_discrete = wm_cfg.n_discrete

        self.mlp = nn.Sequential(
            nn.Linear(33, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.SiLU()
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1, bias=False),
            ChannelLayerNorm(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            ChannelLayerNorm(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            ChannelLayerNorm(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            ChannelLayerNorm(256),
            nn.SiLU(),
            nn.Flatten()
        )

        self.obs_stats_layers = nn.Sequential(
            nn.Linear(wm_cfg.n_deter + wm_cfg.n_mlp_enc + wm_cfg.n_cnn_enc, wm_cfg.hidden_size, bias=False),
            nn.LayerNorm(wm_cfg.hidden_size, eps=1e-3),
            nn.SiLU(),
            nn.Linear(wm_cfg.hidden_size, self.n_stoch * self.n_discrete)
        )

    def forward(self, state_deter, proprio, depth):
        obs_enc = torch.cat([self.mlp(proprio), self.cnn(depth)], dim=-1)
        x = torch.cat([state_deter, obs_enc], dim=-1)
        return self.obs_stats_layers(x).unflatten(-1, (self.n_stoch, self.n_discrete))


class SequenceModel(nn.Module):
    state_deter: torch.Tensor

    def __init__(self, env_cfg, wm_cfg):
        super().__init__()
        self.n_stoch = wm_cfg.n_stoch
        self.n_discrete = wm_cfg.n_discrete
        n_deter = wm_cfg.n_deter
        hidden_size = wm_cfg.hidden_size
        self._initial = wm_cfg.state_initial

        input_dim = self.n_stoch * self.n_discrete + wm_cfg.step_interval * env_cfg.num_actions
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU()
        )

        self.gru = nn.GRU(hidden_size, n_deter)

        # recurrent state of world model
        self.register_buffer('state_deter', torch.zeros(env_cfg.num_envs, n_deter, requires_grad=False))

        if self._initial == "learned":
            self.W = nn.Parameter(torch.zeros(1, n_deter))

        # self.init_model_state(torch.ones(env_cfg.num_envs, dtype=torch.bool))

    def inference_forward(self, state_stoch, prev_actions):  # called during rollout
        x = torch.cat([state_stoch.flatten(1), prev_actions.flatten(1)], dim=1)
        x = self.mlp(x)
        x, _ = self.gru(x.unsqueeze(0), self.state_deter.unsqueeze(0))
        self.state_deter[:] = x.squeeze(0)
        return self.state_deter

    def forward(self, state_deter, state_stoch, prev_actions):  # called during training
        x = torch.cat([state_stoch.flatten(2), prev_actions.flatten(2)], dim=2)
        x = gru_wrapper(self.mlp.forward, x)
        state_deter_new, _ = self.gru(x, state_deter[0].unsqueeze(0))
        return state_deter_new

    def init_model_state(self, dones):
        if self._initial == "zeros":
            self.state_deter[dones] = 0.
        elif self._initial == "learned":
            self.state_deter[dones] = torch.tanh(self.W)
        else:
            raise NotImplementedError(self._initial)


class Dynamics(nn.Module):
    def __init__(self, wm_cfg):
        super().__init__()
        self.n_stoch = wm_cfg.n_stoch
        self.n_discrete = wm_cfg.n_discrete
        hidden_size = wm_cfg.hidden_size

        self.imag_stats_layers = nn.Sequential(
            nn.Linear(wm_cfg.n_deter, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU(),
            nn.Linear(hidden_size, wm_cfg.n_stoch * wm_cfg.n_discrete, bias=True)
        )

    def forward(self, x):
        return self.imag_stats_layers(x).unflatten(1, (self.n_stoch, self.n_discrete))


class WMDecoder(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()

        self.net = nn.ModuleDict({

            "prop": nn.Sequential(
                nn.Linear(1536, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 1024, bias=False),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Linear(1024, 33, bias=False),
            ),

            "depth": nn.Sequential(
                nn.Linear(1536, 4096, bias=True),
                nn.Unflatten(1, (256, 4, 4)),

                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                ChannelLayerNorm(128),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                ChannelLayerNorm(64),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
                ChannelLayerNorm(32),
                nn.SiLU(),
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, bias=True),
            ),

            "rew": nn.Sequential(
                nn.Linear(1536, 512, bias=False),
                nn.LayerNorm(512, eps=0.001, elementwise_affine=True),
                nn.SiLU(),
                nn.Linear(512, 512, bias=False),
                nn.LayerNorm(512, eps=0.001, elementwise_affine=True),
                nn.SiLU(),
                nn.Linear(512, 255, bias=True),
            )
        })

    def forward(self, deter, stoch):
        x = torch.cat([deter, stoch.flatten(1)], dim=1)

        return (
            self.net["prop"](x),
            self.net["depth"](x),
            self.net["rew"](x),
        )
