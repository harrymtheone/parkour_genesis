try:
    import isaacgym, torch
except ImportError:
    import torch

import os

import torch_tensorrt
from torch import nn

from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.modules.model_zju_gru import UNet
from rsl_rl.modules.utils import make_linear_layers


class ObsGRU(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ReLU(inplace=True)

        if env_cfg.len_prop_his == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=64, kernel_size=4),
                activation,
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4),
                activation,
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4),  # (64, 1)
                activation,
                nn.Flatten()
            )
        else:
            raise NotImplementedError

        self.gru = nn.GRU(input_size=64, hidden_size=policy_cfg.obs_gru_hidden_size, num_layers=policy_cfg.obs_gru_num_layers)

    def forward(self, obs_his, hidden_state):
        # update forward
        out = self.conv_layers(obs_his.reshape(1, 10, 50).transpose(1, 2)).reshape(1, 1, 64)
        return self.gru(out, hidden_state)


class ReconGRU(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.LeakyReLU()

        self.cnn_depth = nn.Sequential(
            nn.Conv2d(in_channels=env_cfg.len_depth_his, out_channels=16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.gru = nn.GRU(input_size=128 + policy_cfg.obs_gru_hidden_size,
                          hidden_size=policy_cfg.recon_gru_hidden_size,
                          num_layers=policy_cfg.recon_gru_num_layers)
        self.hidden_state = None

        self.recon_rough = nn.Sequential(
            nn.Unflatten(1, (8, 8, 4)),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            activation,
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            activation,
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
        )

        self.recon_refine = UNet(in_channel=2, out_channel=2)

    def forward(self, depth_his, prop_latent, hidden_state, use_estimated_values=None):
        # update forward
        enc_depth = self.cnn_depth(depth_his).reshape(1, 1, 128)

        # concatenate the two latent vectors
        gru_input = torch.cat([enc_depth, prop_latent], dim=2)
        enc_gru, hidden_state = self.gru(gru_input, hidden_state)

        # we need to compute all memory but no need to reconstruct all hmap
        hmap_rough = self.recon_rough(enc_gru.reshape(1, 256))
        hmap_refine, _ = self.recon_refine(hmap_rough)
        return hmap_rough, hmap_refine, hidden_state


class Mixer(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        predictor_hidden_dim = [128, 64]
        activation = nn.ELU()

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
        # self.mlp_obs = nn.Sequential(
        #     nn.Linear(obs_gru_hidden_size, mixer_embed_dim),
        #     activation,
        # )

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

    def forward(self, scan, latent_obs):
        scan_enc = self.cnn_scan(scan)
        out = self.mixer(torch.cat([latent_obs, scan_enc], dim=1))

        # decode the vector
        return self.mlp_mu(out)


class LocoTransformer(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        num_heads = 4
        num_layers = 2  # 2
        predictor_hidden_dim = [128, 64]
        activation = nn.ReLU(inplace=True)

        obs_gru_hidden_size = policy_cfg.obs_gru_hidden_size
        transformer_embed_dim = policy_cfg.transformer_embed_dim
        self.len_latent = policy_cfg.len_latent
        # vae_output_dim = (policy_cfg.len_latent
        #                   + policy_cfg.len_base_vel
        #                   + policy_cfg.len_latent_feet
        #                   + policy_cfg.len_latent_body)
        vae_output_dim = 16 + 3 + 8 + 16
        self.len_latent = 16

        # patch embedding
        self.cnn_scan = nn.Conv2d(in_channels=2, out_channels=transformer_embed_dim, kernel_size=4, stride=4)
        self.layer_norm = nn.LayerNorm(transformer_embed_dim)

        # observation embedding
        self.mlp_obs = nn.Sequential(
            nn.Linear(obs_gru_hidden_size, transformer_embed_dim),
            activation,
        )

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 33, transformer_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.2)

        # transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.),
            num_layers=num_layers
        )

        # output VAE
        self.mlp_mu = nn.Sequential(
            nn.Linear(transformer_embed_dim * 2, transformer_embed_dim),
            activation,
            nn.Linear(transformer_embed_dim, vae_output_dim),
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(transformer_embed_dim * 2, transformer_embed_dim),
            activation,
            nn.Linear(transformer_embed_dim, self.len_latent)
        )

        # Ot+1 predictor
        self.predictor = nn.Sequential(
            nn.Linear(vae_output_dim, predictor_hidden_dim[0]),
            activation,
            nn.Linear(predictor_hidden_dim[0], predictor_hidden_dim[1]),
            activation,
            nn.Linear(predictor_hidden_dim[1], env_cfg.n_proprio)
        )

    def forward(self, scan, latent_obs):
        # patch embedding
        x = self.cnn_scan(scan)  # (1, 1, 32, 16) -> (1, embed_dim, 8, 4)
        x = x.flatten(2).transpose(1, 2)  # -> (1, 32, embed_dim)
        x = self.layer_norm(x)

        # convert latent_obs to a token
        latent_obs = self.mlp_obs(latent_obs).unsqueeze(1)
        x = torch.cat([latent_obs, x], dim=1)

        # position encoding
        x += self.pos_embed

        # transformer
        out = self.transformer(x)  # -> (1, 33, embed_dim)
        token_prop = out[:, 0]
        token_terrain = torch.mean(out[:, 1:], dim=1)
        out = torch.cat([token_prop, token_terrain], dim=1)

        # decode the vector
        return self.mlp_mu(out)


class Actor(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        # vae_output_dim = (policy_cfg.len_latent
        #                   + policy_cfg.len_base_vel
        #                   + policy_cfg.len_latent_feet
        #                   + policy_cfg.len_latent_body)
        vae_output_dim = 16 + 3 + 8 + 16

        self.actor = make_linear_layers(env_cfg.n_proprio + vae_output_dim,
                                        *policy_cfg.actor_hidden_dims,
                                        env_cfg.num_actions,
                                        activation_func=nn.ELU(),
                                        output_activation=False)

    def forward(self, obs, priv):
        return self.actor(torch.cat([obs, priv], dim=1))


class EstimatorGRU(nn.Module):
    is_recurrent = True

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.enable_VAE = policy_cfg.enable_VAE

        self.obs_gru = ObsGRU(env_cfg, policy_cfg)
        self.reconstructor = ReconGRU(env_cfg, policy_cfg)
        # self.mixer = Mixer(env_cfg, policy_cfg)
        self.transformer = LocoTransformer(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))

    def forward(self, proprio, prop_his, depth, hidden_obs_gru, hidden_recon):  # <-- my mood be like
        # encode history proprio
        latent_obs, hidden_obs_gru_out = self.obs_gru(prop_his.reshape(1, 1, 10, 50), hidden_obs_gru)

        # compute reconstruction
        recon_rough, recon_refine, hidden_recon_out = self.reconstructor(depth, latent_obs, hidden_recon)

        # cross-model mixing using transformer
        est_mu = self.transformer(recon_refine, latent_obs.reshape(1, 64))

        mean = self.actor(proprio, est_mu)

        return mean, recon_rough, recon_refine, hidden_obs_gru_out, hidden_recon_out, est_mu


def trace():
    # proj, cfg, exptid, checkpoint = 't1', 't1_zju', 't1_zju_002', 12100
    # proj, cfg, exptid, checkpoint = 't1', 't1_zju', 't1_zju_002r1', 11200
    proj, cfg, exptid, checkpoint = 't1', 't1_mc', 't1_mc_029', 17200

    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=cfg)

    device = torch.device('cuda')
    load_path = os.path.join(f'../../logs/{proj}/', task_cfg.runner.algorithm_name, exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    # state_dict = torch.load(load_path, map_location=device, weights_only=True)
    state_dict = torch.load('model_58700.pt', map_location=device, weights_only=True)

    model = EstimatorGRU(task_cfg.env, task_cfg.policy)
    model.load_state_dict(state_dict['actor_state_dict'])
    model = model.eval().cuda()

    # Save the traced actor
    func = torch.zeros

    proprio = func(1, task_cfg.env.n_proprio, device=device)
    prop_his = func(1, task_cfg.env.len_prop_his, task_cfg.env.n_proprio, device=device)
    depth_his = func(1, task_cfg.env.len_depth_his, *reversed(task_cfg.sensors.depth_0.resized), device=device)
    hidden_obs_gru = func(1, 1, task_cfg.policy.obs_gru_hidden_size, device=device)
    hidden_recon = func(2, 1, task_cfg.policy.recon_gru_hidden_size, device=device)

    action, recon_rough, recon_refine, hidden_obs_gru_out, hidden_recon_out, est_mu = model(
        proprio, prop_his, depth_his, hidden_obs_gru, hidden_recon)

    # inputs = [proprio, prop_his, depth_his, hidden_obs_gru, hidden_recon]
    inputs = [prop_his, hidden_obs_gru]
    trt_gm = torch_tensorrt.compile(model.obs_gru, ir="ts", inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", inputs=inputs)


if __name__ == '__main__':
    trace()
