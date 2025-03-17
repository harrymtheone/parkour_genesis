import torch
import torch.nn as nn

from .utils import make_linear_layers, gru_wrapper


class EstimatorGRU(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()

        self.prop_his_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=32, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4),
            activation,
            nn.Flatten()
        )

        self.depth_enc = nn.Sequential(
            nn.Conv2d(in_channels=env_cfg.len_depth_his, out_channels=16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        hidden_size = policy_cfg.estimator_gru_hidden_size
        self.gru = nn.GRU(input_size=256, hidden_size=hidden_size, num_layers=1)
        self.hidden_states = None

    def inference_forward(self, prop_his, depth_his):
        # inference forward
        prop_latent = self.prop_his_enc(prop_his.transpose(1, 2))
        depth_latent = self.depth_enc(depth_his)

        gru_input = torch.cat((prop_latent, depth_latent), dim=1)
        # TODO: transformer here?
        gru_out, self.hidden_states = self.gru(gru_input.unsqueeze(0), self.hidden_states)
        return gru_out.squeeze(0)

    def forward(self, prop_his, depth_his, hidden_states):
        # update forward
        prop_latent = gru_wrapper(self.prop_his_enc.forward, prop_his.transpose(2, 3))

        depth_latent = gru_wrapper(self.depth_enc.forward, depth_his)

        gru_input = torch.cat((prop_latent, depth_latent), dim=2)
        gru_out, _ = self.gru(gru_input, hidden_states)
        return gru_out

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def reset(self, dones):
        self.hidden_states[:, dones] = 0.


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

        self.len_estimation = policy_cfg.len_base_vel + policy_cfg.len_latent_feet + policy_cfg.len_latent_body
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
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.actor = make_linear_layers(env_cfg.n_proprio + policy_cfg.estimator_gru_hidden_size,
                                        *policy_cfg.actor_hidden_dims, env_cfg.num_actions,
                                        activation_func=nn.ELU(), output_activation=False)

    def forward(self, proprio, priv):
        return self.actor(torch.cat([proprio, priv], dim=1))


class Policy(nn.Module):
    is_recurrent = True
    from legged_gym.envs.T1.t1_pie_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.estimator = EstimatorGRU(env_cfg, policy_cfg)
        self.vae = EstimatorVAE(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.log(policy_cfg.init_noise_std * torch.ones(env_cfg.num_actions)))
        self.distribution = None

    def act(self,
            obs: ActorObs,
            eval_=False,
            ):  # <-- my mood be like
        # encode history proprio
        gru_out = self.estimator.inference_forward(obs.prop_his, obs.depth)
        vae_mu = self.vae(gru_out)

        # compute action
        mean = self.actor(obs.proprio, vae_mu)

        if eval_:
            # vae_mu, vae_logvar, est, ot1, hmap = self.estimator(obs.prop_his, obs.depth)
            # mean = self.actor(obs.proprio, vae_mu)
            # hmap = hmap.unflatten(1, (32, 16))
            # return mean, hmap, hmap
            return mean

        # sample action from distribution
        self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())
        return self.distribution.sample()

    def train_act(
            self,
            obs: ActorObs,
            hidden_states,
    ):  # <-- my mood be like
        gru_out = self.estimator(obs.prop_his, obs.depth, hidden_states)
        vae_mu, vae_logvar, est, ot1, recon = self.vae.estimate(gru_out)

        # compute action
        mean = gru_wrapper(self.actor, obs.proprio, vae_mu)

        # sample action from distribution
        self.distribution = torch.distributions.Normal(mean, mean * 0. + self.log_std.exp())
        return vae_mu, vae_logvar, est, ot1, recon

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
        new_log_std = torch.log(std * torch.ones_like(self.std.data, device=device))
        self.log_std.data = new_log_std.data

    def get_hidden_states(self):
        return self.estimator.get_hidden_states()

    def reset(self, dones):
        self.estimator.reset(dones)
