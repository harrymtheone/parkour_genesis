import torch
from torch import nn as nn
from torch.distributions import Normal

from rsl_rl.modules.utils import make_linear_layers, gru_wrapper


class EstimatorVAE(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()
        hidden_size = policy_cfg.estimator_gru_hidden_size

        self.mlp_mu = make_linear_layers(hidden_size, hidden_size, hidden_size,
                                         activation_func=activation, output_activation=False)

        self.mlp_logvar = make_linear_layers(hidden_size, hidden_size, hidden_size,
                                             activation_func=activation, output_activation=False)

        self.ot1_predictor = make_linear_layers(hidden_size, 128, env_cfg.n_proprio,
                                                activation_func=activation, output_activation=False)

        self.len_estimation = policy_cfg.len_estimation
        self.len_hmap_latent = policy_cfg.len_hmap_latent
        self.hmap_recon = make_linear_layers(self.len_hmap_latent, 256, env_cfg.n_scan,
                                             activation_func=activation, output_activation=False)

    def forward(self, gru_out):
        return self.mlp_mu(gru_out)

    def estimate(self, gru_out):
        vae_mu = self.mlp_mu(gru_out)
        vae_logvar = self.mlp_logvar(gru_out)
        ot1 = self.ot1_predictor(self.reparameterize(vae_mu, vae_logvar))
        hmap = self.hmap_recon(vae_mu[..., -self.len_hmap_latent:])
        return vae_mu, vae_logvar, vae_mu[..., :self.len_estimation], ot1, hmap

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class Actor(nn.Module):
    is_recurrent = True

    def __init__(self, task_cfg):
        super().__init__()
        env_cfg = task_cfg.env
        policy_cfg = task_cfg.policy
        scan_shape = task_cfg.env.scan_shape

        self.scan_encoder = make_linear_layers(2 * scan_shape[0] * scan_shape[1], 256, 128,
                                               activation_func=nn.ELU())

        # belief encoder
        self.gru = nn.GRU(env_cfg.n_proprio + 128, policy_cfg.actor_gru_hidden_size, num_layers=2)
        self.hidden_states = None

        self.actor = make_linear_layers(policy_cfg.actor_gru_hidden_size,
                                        *policy_cfg.actor_hidden_dims,
                                        env_cfg.num_actions,
                                        activation_func=nn.ELU(),
                                        output_activation=False)

        # Action noise
        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions), requires_grad=True)
        self.distribution = None

    def act(self, obs, recon, use_estimated_values, eval_=False, **kwargs):
        if torch.any(use_estimated_values):
            scan_enc = torch.where(
                use_estimated_values,
                self.scan_encoder(recon.flatten(1)),
                self.scan_encoder(obs.scan.flatten(1))
            )

            x = torch.cat([obs.proprio, scan_enc], dim=1)

        else:
            scan_enc = self.scan_encoder(obs.scan.flatten(1))
            x = torch.cat([obs.proprio, scan_enc], dim=1)

        x, self.hidden_states = self.gru(x.unsqueeze(0), self.hidden_states)
        x = x.squeeze(0)

        mean = self.actor(x)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, scan, hidden_states, use_estimated_values):
        if torch.any(use_estimated_values):
            scan_enc = torch.where(
                use_estimated_values,
                gru_wrapper(self.scan_encoder, scan.flatten(2)),
                gru_wrapper(self.scan_encoder, obs.scan.flatten(2)),
            )

            x = torch.cat([obs.proprio, scan_enc], dim=2)

        else:
            scan_enc = gru_wrapper(self.scan_encoder, obs.scan.flatten(2))
            x = torch.cat([obs.proprio, scan_enc], dim=2)

        x, _ = self.gru(x, hidden_states)

        mean = gru_wrapper(self.actor, x)

        self.distribution = Normal(mean, torch.exp(self.log_std))

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1, keepdim=True)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    def reset_std(self, std):
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=self.log_std.device))
        self.log_std.data = new_log_std.data

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        if self.hidden_states is not None:
            self.hidden_states[:, dones] = 0.
