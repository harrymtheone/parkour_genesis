import torch
import torch.nn as nn

from .model_zju_gru import ObsGRU, ReconGRU, Actor
from .utils import make_linear_layers, gru_wrapper


# class ReconGRU(nn.Module):
#     def __init__(self, env_cfg, policy_cfg):
#         super().__init__()
#         assert policy_cfg.recon_gru_hidden_size == 512
#
#         self.recon_half = UNet(in_channel=4, out_channel=2)  # his_len * (mean + std)
#
#         self.adaptive_avg = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.gru = nn.GRU(input_size=64 + 128, hidden_size=512, num_layers=2)
#         self.hidden_state = None
#
#         self.recon_full = UNet(in_channel=2, out_channel=2)
#
#     def inference_forward(self, depth_scan_his, prop_latent):
#         # inference forward
#         recon_half, depth_scan_enc = self.recon_half(depth_scan_his.flatten(1, 2))
#
#         gru_input = torch.cat([prop_latent, self.adaptive_avg(depth_scan_enc).flatten(1)], dim=1)
#         enc_gru, self.hidden_state = self.gru(gru_input.unsqueeze(0), self.hidden_state)
#
#         recon_full_in = torch.cat([recon_half, enc_gru.view(recon_half.shape)], dim=2)
#         recon_full, _ = self.recon_full(recon_full_in)
#         return recon_half, recon_full
#
#     def forward(self, depth_scan_his, prop_latent, hidden_state):
#         # update forwardn_half, depth_scan_enc = gru_wrapper(self.recon_half.forward, depth_scan_his.flatten(2, 3))
#
#         gru_input = torch.cat([prop_latent, self.adaptive_avg(depth_scan_enc).flatten(2)], dim=2)
#         enc_gru, hidden_state = self.gru(gru_input, hidden_state)
#
#         recon_full_in = torch.cat([recon_half, enc_gru.view(recon_half.shape)], dim=3)
#         recon_full, _ = gru_wrapper(self.recon_full.forward, recon_full_in)
#         return recon_half, recon_full, hidden_state
#
#     def get_hidden_state(self):
#         if self.hidden_state is None:
#             return None
#         return self.hidden_state.detach()
#
#     def reset(self, dones):
#         self.hidden_state[:, dones] = 0.


class Mixer(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        predictor_hidden_dim = [128, 64]
        activation = nn.ReLU(inplace=True)

        obs_gru_hidden_size = policy_cfg.obs_gru_hidden_size
        mixer_embed_dim = policy_cfg.transformer_embed_dim
        self.len_latent = policy_cfg.len_latent
        vae_output_dim = (policy_cfg.len_latent
                          + policy_cfg.len_base_vel
                          + policy_cfg.len_latent_feet
                          + policy_cfg.len_latent_body)

        # patch embedding
        self.cnn_scan = nn.Sequential(  # 32, 16
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=1),  # 16, 8
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # 8, 4
            activation,
            nn.Conv2d(in_channels=32, out_channels=mixer_embed_dim, kernel_size=3, stride=2, padding=1),  # 4, 2
            activation,
            nn.AdaptiveAvgPool2d((1, 1)),  # 1, 1
            nn.Flatten()
        )

        # observation embedding
        self.mlp_obs = nn.Sequential(
            nn.Linear(obs_gru_hidden_size, mixer_embed_dim),
            activation,
        )

        self.mixer = make_linear_layers(mixer_embed_dim * 2, mixer_embed_dim * 2, mixer_embed_dim * 2,
                                        activation_func=activation)

        # output VAE
        self.mlp_mu = nn.Sequential(
            nn.Linear(mixer_embed_dim * 2, mixer_embed_dim),
            activation,
            nn.Linear(mixer_embed_dim, vae_output_dim),
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(mixer_embed_dim * 2, mixer_embed_dim),
            activation,
            nn.Linear(mixer_embed_dim, self.len_latent)
        )

        # Ot+1 predictor
        self.predictor = nn.Sequential(
            nn.Linear(vae_output_dim, predictor_hidden_dim[0]),
            activation,
            nn.Linear(predictor_hidden_dim[0], predictor_hidden_dim[1]),
            activation,
            nn.Linear(predictor_hidden_dim[1], env_cfg.n_proprio)
        )

    def inference_forward(self, scan, latent_obs):
        latent_scan = self.cnn_scan(scan)  # (1, 1, 32, 16) -> (1, embed_dim, 8, 4)
        latent_obs = self.mlp_obs(latent_obs)
        out = self.mixer(torch.cat([latent_obs, latent_scan], dim=1))

        # decode the vector
        est_mu = self.mlp_mu(out)
        return est_mu[..., :self.len_latent], est_mu[..., self.len_latent:]

    def forward(self, scan, latent_obs):
        latent_scan = self.cnn_scan(scan)  # (1, 1, 32, 16) -> (1, embed_dim, 8, 4)
        latent_obs = self.mlp_obs(latent_obs)
        out = self.mixer(torch.cat([latent_obs, latent_scan], dim=1))

        # decode the vector
        est_mu = self.mlp_mu(out)
        est_latent = est_mu[..., :self.len_latent]
        est = est_mu[..., self.len_latent:]

        est_logvar = self.mlp_logvar(out)

        # reparameterize and predict O_t+1
        z = self.reparameterize(est_latent, est_logvar)
        ot1 = self.predictor(torch.cat([z, est.detach()], dim=1))

        return est_latent, est, est_logvar, ot1

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class EstimatorNoRecon(nn.Module):
    is_recurrent = True
    from legged_gym.envs.T1.t1_zju_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.obs_gru = ObsGRU(env_cfg, policy_cfg)
        self.reconstructor = ReconGRU(env_cfg, policy_cfg)
        self.mixer = Mixer(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

    def act(self,
            obs: ActorObs,
            eval_=False,
            **kwargs):  # <-- my mood be like

        # encode history proprio
        latent_obs = self.obs_gru.inference_forward(obs.prop_his)

        # cross-model mixing using transformer
        est_latent, _, _, _ = self.mixer(obs.scan, latent_obs)
        latent_input = torch.cat([est_latent, obs.priv_actor], dim=1)

        # compute action
        mean = self.actor(obs.proprio, latent_input)

        # output action
        if eval_:
            return mean
        else:
            # sample action from distribution
            self.distribution = torch.distributions.Normal(mean, self.log_std.exp())
            return self.distribution.sample()

    def train_act(
            self,
            obs: ActorObs,
            hidden_states,
            use_estimated_values: torch.Tensor,
    ):  # <-- my mood be like

        # encode history proprio
        obs_enc_hidden_states, _ = hidden_states
        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden_states)

        # cross-model mixing using transformer
        est_latent, est, _, _ = gru_wrapper(self.mixer.forward, obs.scan, latent_obs)
        latent_input = torch.cat([est_latent, obs.priv_actor], dim=2)

        # compute action
        mean = gru_wrapper(self.actor.forward, obs.proprio, latent_input)

        # output action
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp())

    def reconstruct(self, obs, obs_enc_hidden, *args):
        # encode history proprio
        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden)
        est_latent, est, est_logvar, ot1 = gru_wrapper(self.mixer.forward, obs.scan, latent_obs)
        return None, None, est_latent, est, est_logvar, ot1

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

    def get_hidden_state(self):
        return (self.obs_gru.get_hidden_state(),)

    def reset(self, dones):
        self.obs_gru.reset(dones)


class EstimatorGRU(nn.Module):
    is_recurrent = True
    from legged_gym.envs.T1.t1_zju_environment import ActorObs

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.obs_gru = ObsGRU(env_cfg, policy_cfg)
        self.reconstructor = ReconGRU(env_cfg, policy_cfg)
        self.mixer = Mixer(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

    def act(self,
            obs: ActorObs,
            use_estimated_values: torch.Tensor,
            eval_=False,
            **kwargs):  # <-- my mood be like
        use_estimated_values = use_estimated_values.squeeze(-1)

        # encode history proprio
        latent_obs = self.obs_gru.inference_forward(obs.prop_his)

        # compute reconstruction
        with torch.no_grad():
            recon_rough, recon_refine = self.reconstructor.inference_forward(
                obs.depth, latent_obs, use_estimated_values)

        # cross-model mixing using transformer
        recon_output = obs.scan.clone().to(recon_refine.dtype)
        recon_output[use_estimated_values] = recon_refine

        est_latent, est = self.mixer.inference_forward(recon_output, latent_obs)

        latent_input = torch.where(
            use_estimated_values.unsqueeze(1),
            torch.cat([est_latent, est.detach()], dim=1),
            torch.cat([est_latent, obs.priv_actor], dim=1)
        )

        # compute action
        mean = self.actor(obs.proprio, latent_input)

        if eval_:
            return mean, recon_rough.squeeze(1), recon_refine.squeeze(1), est

        # sample action from distribution
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp())
        return self.distribution.sample()

    def train_act(
            self,
            obs: ActorObs,
            hidden_states,
            use_estimated_values: torch.Tensor,
    ):  # <-- my mood be like
        use_estimated_values = use_estimated_values.squeeze(-1)

        # encode history proprio
        obs_enc_hidden_states, recon_hidden_states = hidden_states

        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden_states)

        # compute reconstruction
        with torch.no_grad():
            _, recon_refine, _ = self.reconstructor(
                obs.depth, latent_obs, recon_hidden_states, use_estimated_values)

        # cross-model mixing using transformer
        recon_output = obs.scan.clone().to(recon_refine.dtype)
        recon_output[use_estimated_values] = recon_refine

        est_latent, est = gru_wrapper(self.mixer.inference_forward, recon_output, latent_obs)

        latent_input = torch.where(
            use_estimated_values.unsqueeze(2),
            torch.cat([est_latent, est.detach()], dim=2),
            torch.cat([est_latent, obs.priv_actor], dim=2)
        )

        # compute action
        mean = gru_wrapper(self.actor.forward, obs.proprio, latent_input)

        # output action
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp())

    def reconstruct(self, obs, obs_enc_hidden, recon_hidden):
        # encode history proprio
        latent_obs, _ = self.obs_gru(obs.prop_his, obs_enc_hidden)
        recon_rough, recon_refine, _ = self.reconstructor(obs.depth, latent_obs.detach(), recon_hidden)

        est_latent, est, est_logvar, ot1 = gru_wrapper(self.mixer.forward, recon_refine.detach(), latent_obs)

        return recon_rough.squeeze(2), recon_refine.squeeze(2), est_latent, est, est_logvar, ot1

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

    def get_hidden_state(self):
        return self.obs_gru.get_hidden_state(), self.reconstructor.get_hidden_state()

    def reset(self, dones):
        self.obs_gru.reset(dones)
        self.reconstructor.reset(dones)
