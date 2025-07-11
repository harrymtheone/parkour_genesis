try:
    import isaacgym, torch
except ImportError:
    import torch

import os

from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.modules.model_pie import EstimatorGRU, EstimatorVAE, Actor
from torch import nn


class Policy(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.estimator = EstimatorGRU(env_cfg, policy_cfg)
        self.vae = EstimatorVAE(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.ones(env_cfg.num_actions))

    def forward(self, proprio, prop_his, depth, hidden_states):  # <-- my mood be like
        # encode history proprio
        gru_out = self.estimator(prop_his.unsqueeze(0), depth.unsqueeze(0), hidden_states)
        vae_mu, vae_logvar, est, ot1, hmap = self.vae.estimate(gru_out)
        mean = self.actor(proprio, vae_mu.squeeze(0))
        return mean, hidden_states, hmap.squeeze().view(32, 16)


def trace():
    proj, cfg, exptid, checkpoint = 't1', 't1_pie_stair', 't1_pie_004s2', 9400

    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=cfg)

    device = torch.device('cpu')
    load_path = os.path.join(f'../../logs/{proj}/{task_cfg.runner.algorithm_name}', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = Policy(task_cfg.env, task_cfg.policy)
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
        # Save the traced actor
        proprio = torch.zeros(1, task_cfg.env.n_proprio, device=device)
        prop_his = torch.zeros(1, task_cfg.env.len_prop_his, task_cfg.env.n_proprio, device=device)
        depth_his = torch.zeros(1, task_cfg.env.len_depth_his, *reversed(task_cfg.sensors.depth_0.resized), device=device)
        hidden_states = torch.zeros(1, 1, task_cfg.policy.estimator_gru_hidden_size, device=device)

        trace_and_save(model, (proprio, prop_his, depth_his, hidden_states))


if __name__ == '__main__':
    trace()
