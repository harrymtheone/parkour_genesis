# from rsl_rl.modules.model_zju_gru import ObsGRU, ReconGRU, LocoTransformer, Actor
from rsl_rl.modules.model_zju_exp import ObsGRU, ReconGRU, Mixer, Actor

try:
    import isaacgym, torch
except ImportError:
    import torch

import os

from torch import nn

from legged_gym.utils.task_registry import TaskRegistry


class EstimatorGRU(nn.Module):
    is_recurrent = True

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.len_latent = policy_cfg.len_latent
        self.est_dim = self.len_latent + policy_cfg.len_estimation

        self.obs_gru = ObsGRU(env_cfg, policy_cfg)
        self.reconstructor = ReconGRU(env_cfg, policy_cfg)
        # self.transformer = LocoTransformer(env_cfg, policy_cfg)
        self.mixer = Mixer(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))
        self.distribution = None

    def forward(self, proprio, prop_his, depth, hidden_obs_gru, hidden_recon):  # <-- my mood be like
        # encode history proprio
        latent_obs, hidden_obs_gru = self.obs_gru(prop_his.unsqueeze(0), hidden_obs_gru)

        # compute reconstruction
        recon_rough, recon_refine, hidden_recon = self.reconstructor(depth.unsqueeze(0), latent_obs, hidden_recon)

        # cross-model mixing using transformer
        # est_latent, est, est_logvar, ot1 = self.transformer(recon_refine.squeeze(0), latent_obs.squeeze(0))
        est_latent, est, est_logvar, ot1 = self.mixer(recon_refine.squeeze(0), latent_obs.squeeze(0))

        # compute action
        latent_input = torch.cat([est_latent, est.detach()], dim=1)
        mean = self.actor(proprio, latent_input)

        # output action
        return mean, recon_rough.squeeze(), recon_refine.squeeze(), hidden_obs_gru, hidden_recon


def trace():
    proj, cfg, exptid, checkpoint = 't1', 't1_mc', 't1_mc_045r22', 91000
    # proj, cfg, exptid, checkpoint = 't1', 't1_mc', 't1_mc_056', 39000

    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=cfg)

    device = torch.device('cpu')
    load_path = os.path.join(f'../../logs/{proj}/', task_cfg.runner.algorithm_name, exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = EstimatorGRU(task_cfg.env, task_cfg.policy)
    model.load_state_dict(state_dict['actor_state_dict'])
    model.eval()

    # define the trace function
    def trace_and_save(model, args):
        model(*args)
        model_jit = torch.jit.trace(model, args)
        model_jit(*args)
        model_path = os.path.join(trace_path, f'{exptid}_{checkpoint}_jit.pt')

        try:
            torch.jit.save(model_jit, model_path)
            print("Saving model to:", os.path.abspath(model_path))
        except Exception as e:
            print(f"Failed to save files: {e}")

    with torch.no_grad():
        env, policy = task_cfg.env, task_cfg.policy

        # Save the traced actor
        proprio = torch.zeros(1, env.n_proprio, device=device)
        prop_his = torch.zeros(1, env.len_prop_his, env.n_proprio, device=device)
        depth_his = torch.zeros(1, env.len_depth_his, *reversed(task_cfg.sensors.depth_0.resized), device=device)
        hidden_obs_gru = torch.zeros(1, 1, policy.obs_gru_hidden_size, device=device)
        hidden_recon = torch.zeros(2, 1, policy.recon_gru_hidden_size, device=device)

        trace_and_save(model, (proprio, prop_his, depth_his, hidden_obs_gru, hidden_recon))


if __name__ == '__main__':
    trace()
