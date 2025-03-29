import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import make_linear_layers, gru_wrapper

gru_hidden_size = 128
encoder_output_size = 3 + 64  # v_t, z_t


class VAE(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()

        self.mlp_mu = make_linear_layers(gru_hidden_size, 128, encoder_output_size,
                                         activation_func=activation,
                                         output_activation=False)

        self.mlp_logvar = make_linear_layers(gru_hidden_size, 128, encoder_output_size,
                                             activation_func=activation,
                                             output_activation=False)

        self.decoder = make_linear_layers(encoder_output_size, 64, env_cfg.n_proprio,
                                          activation_func=activation,
                                          output_activation=False)

    def forward(self, obs_enc, mu_only=False):
        if mu_only:
            return self.mlp_mu(obs_enc)

        est_mu = self.mlp_mu(obs_enc)
        est_logvar = self.mlp_logvar(obs_enc)
        ot1 = self.decoder(self.reparameterize(est_mu, est_logvar))
        return ot1, est_mu, est_logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class Actor(nn.Module):
    is_recurrent = False

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.activation = nn.ELU()

        # construct actor network
        channel_size = 16

        self.obs_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=2 * channel_size, kernel_size=8, stride=4),
            self.activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=6, stride=1),
            self.activation,
            nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=6, stride=1),  # (8 * channel_size, 1)
            self.activation,
            nn.Flatten()
        )
        self.vae = VAE(env_cfg, policy_cfg)

        # Action noise
        self.actor_backbone = make_linear_layers(env_cfg.n_proprio + encoder_output_size,
                                                 *policy_cfg.actor_hidden_dims,
                                                 env_cfg.num_actions,
                                                 activation_func=self.activation,
                                                 output_activation=False)
        self.log_std = nn.Parameter(torch.log(policy_cfg.init_noise_std * torch.ones(env_cfg.num_actions)))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def act(self, obs, eval_=False, **kwargs):
        obs_enc = self.obs_enc(obs.prop_his.transpose(1, 2))
        est_mu = self.vae(obs_enc, mu_only=True)
        actor_input = torch.cat((obs.proprio, self.activation(est_mu)), dim=1)
        mean = self.actor_backbone(actor_input)

        if eval_:
            return mean

        self.distribution = Normal(mean, mean * 0. + torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, **kwargs):
        obs_enc = self.obs_enc(obs.prop_his.transpose(1, 2))
        est_mu = self.vae(obs_enc, mu_only=True)
        actor_input = torch.cat((obs.proprio, self.activation(est_mu)), dim=1)
        mean = self.actor_backbone(actor_input)
        self.distribution = Normal(mean, mean * 0. + torch.exp(self.log_std))

    # def estimate(self, obs):
    #     obs_enc = self.obs_enc(obs.prop_his.transpose(1, 2))
    #     return self.vae(obs_enc, mu_only=False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    def reset_std(self, std, device):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=device))
        self.log_std.data = new_log_std.data


class ActorGRU(nn.Module):
    is_recurrent = True

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.activation = nn.ELU()

        gru_num_layers = 1
        self.gru = nn.GRU(input_size=env_cfg.n_proprio, hidden_size=gru_hidden_size, num_layers=gru_num_layers)
        self.hidden_states = None

        self.vae = VAE(env_cfg, policy_cfg)

        self.actor_backbone = make_linear_layers(encoder_output_size + env_cfg.n_proprio,
                                                 *policy_cfg.actor_hidden_dims,
                                                 env_cfg.num_actions,
                                                 activation_func=self.activation,
                                                 output_activation=False)

        # Action noise
        self.log_std = nn.Parameter(torch.log(policy_cfg.init_noise_std * torch.ones(env_cfg.num_actions)))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def act(self, obs, eval_=False, **kwargs):
        # inference forward
        obs_enc, self.hidden_states = self.gru(obs.proprio.unsqueeze(0), self.hidden_states)
        est_mu = self.vae(obs_enc.squeeze(0), mu_only=True)
        actor_input = torch.cat([obs.proprio, self.activation(est_mu)], dim=1)
        mean = self.actor_backbone(actor_input)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, hidden_states, **kwargs):
        obs_enc, _ = self.gru(obs.proprio, hidden_states)
        est_mu = gru_wrapper(self.vae.forward, obs_enc, mu_only=True)

        actor_input = torch.cat([obs.proprio, self.activation(est_mu)], dim=2)
        mean = gru_wrapper(self.actor_backbone.forward, actor_input)

        self.distribution = Normal(mean, mean * 0. + torch.exp(self.log_std))

    def estimate(self, obs, hidden_states, **kwargs):
        obs_enc, _ = self.gru(obs.proprio, hidden_states)
        ot1, est_mu, est_logvar = gru_wrapper(self.vae.forward, obs_enc, mu_only=False)
        return ot1, est_mu[..., :3], est_mu, est_logvar

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    def reset_std(self, std, device):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=device))
        self.log_std.data = new_log_std.data

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        self.hidden_states[:, dones] = 0


class Critic(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        channel_size = 16
        activation = nn.ELU()

        self.priv_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.num_critic_obs, out_channels=2 * channel_size, kernel_size=6, stride=4),
            activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=6, stride=2),
            activation,
            nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=4, stride=1),
            activation,
            nn.Flatten()
        )

        self.critic = make_linear_layers(8 * channel_size + env_cfg.n_scan, *policy_cfg.critic_hidden_dims, 1,
                                         activation_func=nn.ELU(),
                                         output_activation=False)

    def evaluate(self, obs):
        if obs.priv_his.ndim == 3:
            priv_his = obs.priv_his.transpose(1, 2)
            scan = obs.scan.flatten(1)
            priv_latent = self.priv_enc(priv_his)
            value = self.critic(torch.cat([priv_latent, scan], dim=1))
            return value
        else:
            n_steps = obs.priv_his.size(0)
            priv_his = obs.priv_his.flatten(0, 1).transpose(1, 2)
            scan = obs.scan.flatten(0, 1).flatten(1)
            priv_latent = self.priv_enc(priv_his)
            value = self.critic(torch.cat([priv_latent, scan], dim=1))
            return value.unflatten(0, (n_steps, -1))
