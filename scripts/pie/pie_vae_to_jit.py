try:
    import isaacgym, torch
except ImportError:
    import torch

import os

from torch import nn

from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.algorithms.pie_amp.networks import Mixer, EstimatorVAE, Actor


class MixerJIT(Mixer):
    def forward(self, proprio, depth_his, hidden_states, **kwargs):
        # update forward
        prop_latent = self.prop_his_enc(proprio)

        depth_latent = self.depth_enc(depth_his)

        mixer_out, hidden_states = self.gru(
            torch.cat((prop_latent, depth_latent), dim=-1).unsqueeze(0),
            hidden_states
        )
        return mixer_out.squeeze(0), hidden_states


class EstimatorJIT(EstimatorVAE):
    def forward(self, mixer_out):
        vel = self.mlp_vel(mixer_out)
        z = self.mlp_z(mixer_out)

        ot1 = self.ot1_predictor(torch.cat([vel, z], dim=-1))
        hmap = self.hmap_recon(z[:, -self.len_latent_hmap:])

        return vel, z, ot1, hmap


class PolicyJIT(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.mixer = MixerJIT(env_cfg, policy_cfg)
        self.vae = EstimatorJIT(env_cfg, policy_cfg)
        self.actor = Actor(env_cfg, policy_cfg)

        self.log_std = nn.Parameter(torch.ones(env_cfg.num_actions))

    def forward(self, proprio, depth, mixer_hidden_states):  # <-- my mood be like
        mixer_out, hidden_states = self.mixer.forward(proprio, depth, mixer_hidden_states)
        vel, z, ot1, hmap = self.vae.forward(mixer_out)

        mean = self.actor(proprio, vel, z)

        return mean, hidden_states, hmap.squeeze().view(32, 16)


def trace():
    proj, cfg, exptid, checkpoint = 't1', 't1_pie_amp', 't1_pie_amp_026', 10100

    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=cfg)

    device = torch.device('cpu')
    load_path = os.path.join(f'../../logs/{proj}/{task_cfg.runner.algorithm_name}', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = PolicyJIT(task_cfg.env, task_cfg.policy)
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
        depth = torch.zeros(1, 1, *reversed(task_cfg.sensors.depth_0.resized), device=device)
        hidden_states = torch.zeros(1, 1, task_cfg.policy.estimator_gru_hidden_size, device=device)

        print("------------ input shape ------------")
        print("proprio", proprio.shape)
        print("depth", depth.shape)
        print("hidden_states", hidden_states.shape)

        trace_and_save(model, (proprio, depth, hidden_states))


if __name__ == '__main__':
    trace()
