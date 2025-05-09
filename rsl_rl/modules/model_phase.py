import torch
from torch import nn
from torch.distributions import Normal

from .utils import make_linear_layers, gru_wrapper


class Modulator(nn.Module):
    def __init__(self, env_cfg, task_cfg, blind_actor):
        super().__init__()
        self.blind_actor = blind_actor
        self.is_recurrent = blind_actor.is_recurrent

        activation = nn.ELU()

        self.mlp_prop = make_linear_layers(
            env_cfg.n_proprio + env_cfg.num_actions, 128, 256,
            activation_func=activation
        )

        self.mlp_hmap = make_linear_layers(
            2 * env_cfg.n_scan, 512, 256,
            activation_func=activation
        )

        self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=2)
        self.hidden_states = None

        self.mlp_out = make_linear_layers(
            512, 256,
            activation_func=activation,
        )
        self.mlp_out_actions = nn.Linear(256, env_cfg.num_actions)
        self.mlp_out_clock = nn.Sequential(
            nn.Linear(256, 1 + 2),
            nn.Tanh()
        )

        # Action noise
        self.log_std_action = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.dist_action = None

        self.log_std_clock = nn.Parameter(torch.zeros(3))
        self.dist_clock = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def act(self, obs, eval_=False, **kwargs):
        with torch.no_grad():
            blind_actions = self.blind_actor.act(obs, eval_=True)

        prop_enc = self.mlp_prop(torch.cat([obs.proprio, blind_actions], dim=1))
        hmap_enc = self.mlp_hmap(obs.scan.flatten(1))

        gru_input = torch.cat([prop_enc, hmap_enc], dim=1)
        gru_out, self.hidden_states = self.gru(gru_input.unsqueeze(0), self.hidden_states)

        action_clock = self.mlp_out(gru_out.squeeze(0))
        action_mean = self.mlp_out_actions(action_clock) + blind_actions
        clock_mean = self.mlp_out_clock(action_clock)

        if eval_:
            return torch.cat([action_mean, clock_mean], dim=-1)

        self.dist_action = Normal(action_mean, torch.exp(self.log_std_action))
        self.dist_clock = Normal(clock_mean, torch.exp(self.log_std_clock))
        return self.dist_action.sample(), self.dist_clock.sample()

    def train_act(self, obs, hidden_states, **kwargs):
        hidden_states, modulator_hidden_states = hidden_states
        with torch.no_grad():
            self.blind_actor.train_act(obs, hidden_states)
            blind_actions = self.blind_actor.action_mean

        prop_enc = gru_wrapper(self.mlp_prop.forward, torch.cat([obs.proprio, blind_actions], dim=2))
        hmap_enc = gru_wrapper(self.mlp_hmap.forward, obs.scan.flatten(2))

        gru_input = torch.cat([prop_enc, hmap_enc], dim=2)
        gru_out, _ = self.gru(gru_input, modulator_hidden_states)

        action_clock = gru_wrapper(self.mlp_out, gru_out)
        action_mean = gru_wrapper(self.mlp_out_actions, action_clock) + blind_actions
        clock_mean = gru_wrapper(self.mlp_out_clock, action_clock)

        self.dist_action = Normal(action_mean, torch.exp(self.log_std_action))
        self.dist_clock = Normal(clock_mean, torch.exp(self.log_std_clock))
        # self.dist_clock = Normal(clock_mean, torch.exp(0.1 * torch.ones(3, device=self.log_std_action.device)))

    @property
    def action_mean(self):
        return torch.cat([self.dist_action.mean, self.dist_clock.mean], dim=-1)

    @property
    def action_std(self):
        return torch.cat([self.dist_action.stddev, self.dist_clock.stddev], dim=-1)

    @property
    def entropy(self):
        return self.dist_action.entropy().sum(dim=-1), self.dist_clock.entropy().sum(dim=-1)

    def get_log_prob(self, action_clock):
        log_prob_action = self.dist_action.log_prob(action_clock[..., :-3])
        log_prob_clock = self.dist_clock.log_prob(action_clock[..., -3:])
        return torch.cat([log_prob_action, log_prob_clock], dim=-1).sum(dim=-1, keepdim=True)

    def reset_std(self, std, device):
        new_log_std = torch.log(0.5 * torch.ones_like(self.log_std_action.data, device=device))
        self.log_std_action.data = new_log_std.data

        new_log_std = torch.log(0.5 * torch.ones_like(self.log_std_clock.data, device=device))
        self.log_std_clock.data = new_log_std.data

    def get_hidden_states(self):
        if self.hidden_states is None:
            return self.blind_actor.get_hidden_states(), None
        return self.blind_actor.get_hidden_states(), self.hidden_states.detach()

    def reset(self, dones):
        self.hidden_states[:, dones] = 0
