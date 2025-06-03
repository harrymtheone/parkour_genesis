import torch
from torch import nn

from rsl_rl.modules.wmp.utils import ChannelLayerNorm
from .utils import UniMixOneHotCategorical, gru_wrapper, make_linear_layers


class WMEncoder(nn.Module):
    def __init__(self, env_cfg, wm_cfg):
        super().__init__()

        self.n_stoch = wm_cfg.n_stoch
        self.n_discrete = wm_cfg.n_discrete

        self.mlp = nn.Sequential(
            nn.Linear(env_cfg.n_proprio - env_cfg.num_actions, 64, bias=False),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128, bias=False),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 256, bias=False),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 512, bias=False),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1024, bias=False),
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
        x = torch.cat([state_deter, self.mlp(proprio), self.cnn(depth)], dim=-1)
        return self.obs_stats_layers(x).unflatten(-1, (self.n_stoch, self.n_discrete))


class SequenceModel(nn.Module):
    def __init__(self, env_cfg, wm_cfg):
        super().__init__()
        input_dim = wm_cfg.n_stoch * wm_cfg.n_discrete + env_cfg.num_actions
        hidden_size = wm_cfg.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU()
        )

        self.gru = nn.GRU(hidden_size, wm_cfg.n_deter)

    def forward(self, state_deter, state_stoch, prev_actions):
        x = gru_wrapper(self.mlp.forward, torch.cat([
            state_stoch.flatten(2),
            prev_actions.flatten(2)
        ], dim=2))

        state_deter_new, _ = self.gru(x, state_deter)
        return state_deter_new


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
                nn.Linear(1024, env_cfg.n_proprio - env_cfg.num_actions, bias=False),
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


class RSSM(nn.Module):
    state_deter: torch.Tensor
    state_stoch: torch.Tensor

    def __init__(self, env_cfg, train_cfg):
        super().__init__()
        self.num_actions = env_cfg.num_actions
        self.cfg = train_cfg.rssm
        self.unimix_ratio = self.cfg.unimix_ratio
        self.initial = self.cfg.state_initial

        self.sequence_model = SequenceModel(env_cfg, self.cfg)

        self.encoder = WMEncoder(env_cfg, self.cfg)
        # self.dynamics = Dynamics(self.cfg)
        self.decoder = WMDecoder(env_cfg, train_cfg)

        # recurrent and stochastic state of world model
        self.register_buffer('state_deter', torch.zeros(env_cfg.num_envs, self.cfg.n_deter, requires_grad=False))
        self.register_buffer('state_stoch', torch.zeros(env_cfg.num_envs, self.cfg.n_stoch, self.cfg.n_discrete, requires_grad=False))

        # initialize RSSM
        if self.initial == "learned":
            self.W = nn.Parameter(torch.zeros(1, self.cfg.n_deter))

    def step(self, proprio, depth, is_first_step):
        prop_rssm = proprio[:, :-self.num_actions].clone()
        prop_rssm[:, 6: 6 + 5] = 0.
        prev_actions = proprio[:, -self.num_actions:]

        if torch.any(is_first_step):
            self.reset(is_first_step)

        self.state_deter[:] = self.sequence_model(
            self.state_deter.unsqueeze(0),
            self.state_stoch.unsqueeze(0),
            prev_actions.unsqueeze(0),
        ).squeeze(0)

        post_digits = self.encoder(self.state_deter, prop_rssm, depth)
        self.state_stoch[:] = self.get_dist(post_digits).sample()

        return self.get_feature(proprio)

    def train_step(self, prop, depth, state_deter, state_stoch, is_first_step):
        prop_rssm = prop[:, :, :-self.num_actions].clone()
        prop_rssm[:, :, 6: 6 + 5] = 0.
        prev_actions = prop[:, :, -self.num_actions:]

        if torch.any(is_first_step):
            state_deter = state_deter.clone()
            state_stoch = state_stoch.clone()
            self.reset(is_first_step, state_deter=state_deter, state_stoch=state_stoch)

        state_deter_new = self.sequence_model(
            state_deter[0].contiguous().unsqueeze(0), state_stoch, prev_actions
        )

        # prior_digits = gru_wrapper(self.dynamics.forward, state_deter_new)

        post_digits = gru_wrapper(self.encoder.forward, state_deter_new, prop_rssm, depth)
        state_stoch_new = self.get_dist(post_digits).sample()

        prop_pred, depth_pred, rew_pred = gru_wrapper(self.decoder, state_deter_new, state_stoch_new)

        # return prior_digits, post_digits, prop, depth, rew
        return prop_pred, depth_pred, rew_pred
        # return self.get_feature(prop, state_deter_new, state_stoch_new).detach(), prop_pred, depth_pred, rew_pred

    def play_step(self, proprio, depth, is_first_step, sample=True):
        prop_rssm = proprio[:, :-self.num_actions]
        prev_actions = proprio[:, -self.num_actions:]

        if torch.any(is_first_step):
            self.reset(is_first_step)

        self.state_deter[:] = self.sequence_model(
            self.state_deter.unsqueeze(0),
            self.state_stoch.unsqueeze(0),
            prev_actions.unsqueeze(0),
        ).squeeze(0)

        post_digits = self.encoder(self.state_deter, prop_rssm, depth)

        if sample:
            self.state_stoch[:] = self.get_dist(post_digits).sample()
        else:
            self.state_stoch[:] = self.get_dist(post_digits).mode

        prop, depth, rew = self.decoder(self.state_deter, self.state_stoch)

        return self.get_feature(proprio), {'wm_prop': prop, 'wm_depth': depth, 'wm_rew': rew}

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

    def get_feature(self, proprio, state_deter=None, state_stoch=None):
        if state_deter is None:
            state_deter = self.state_deter
        if state_stoch is None:
            state_stoch = self.state_stoch

        if self.cfg.actor_input_type == 0:
            return state_deter.clone()
        elif self.cfg.actor_input_type == 1:
            return torch.cat([state_deter, state_stoch.flatten(1)], dim=1)
        elif self.cfg.actor_input_type == 2:
            return torch.cat([proprio, state_deter, state_stoch.flatten(1)], dim=1)
        else:
            raise NotImplementedError

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


class Actor(nn.Module):
    is_recurrent = False

    def __init__(self, task_cfg):
        super().__init__()
        activation = nn.ELU()
        rssm_cfg = task_cfg.rssm
        self.actor_input_type = rssm_cfg.actor_input_type

        if self.actor_input_type == 0:
            self.actor = make_linear_layers(rssm_cfg.n_deter, *task_cfg.policy.actor_hidden_dims, task_cfg.env.num_actions,
                                            activation_func=activation,
                                            output_activation=False)

        elif self.actor_input_type == 1:
            state_dim = rssm_cfg.n_deter + rssm_cfg.n_stoch * rssm_cfg.n_discrete
            self.actor = make_linear_layers(state_dim, *task_cfg.policy.actor_hidden_dims, task_cfg.env.num_actions,
                                            activation_func=activation,
                                            output_activation=False)
        else:
            state_dim = rssm_cfg.n_deter + rssm_cfg.n_stoch * rssm_cfg.n_discrete
            self.actor = nn.Sequential(
                nn.Linear(task_cfg.env.n_proprio + state_dim, 512),
                activation,
                nn.Linear(512, 256),
                activation,
                nn.Linear(256, 128),
                activation,
                nn.Linear(128, task_cfg.env.num_actions),
            )

        self.log_std = nn.Parameter(torch.ones(task_cfg.env.num_actions), requires_grad=False)
        self.distribution = None

        # disable args validation for speedup
        torch.distributions.Normal.set_default_validate_args = False

    def act(self, state, eval_=False, **kwargs):
        mean = self.actor(state)

        if eval_:
            return mean

        self.distribution = torch.distributions.Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, state):
        return self.act(state)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset_std(self, std, device):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=device))
        self.log_std.data = new_log_std.data
