try:
    import isaacgym, torch
except ImportError:
    pass

import os

from torch import nn

from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.algorithms.dreamwaq.networks import VAE, Actor


class Policy(nn.Module):
    def __init__(self, vae, actor):
        super().__init__()
        self.vae = vae
        self.actor = actor

    def forward(self, proprio, vae_hidden_states):
        proprio = proprio.unsqueeze(0)

        vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1 = self.vae(proprio, vae_hidden_states, sample=False)

        actions = self.actor(proprio, vel, z)

        return actions.squeeze(0), vae_hidden_states


def trace():
    proj, cfg, exptid, checkpoint = 't1', 't1_dreamwaq', 't1_dream_006', 0

    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=cfg)

    device = torch.device('cpu')
    load_path = os.path.join(f'../../logs/{proj}/{task_cfg.runner.algorithm_name}/', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    vae = VAE(task_cfg.env, task_cfg.policy)
    actor = Actor(task_cfg.env, task_cfg.policy)

    vae.load_state_dict(state_dict['vae'])
    actor.load_state_dict(state_dict['actor'])

    policy = Policy(vae, actor)
    policy.eval()

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
        vae_hidden_states = torch.zeros(1, 1, 128, device=device)

        trace_and_save(policy, (proprio, vae_hidden_states))


if __name__ == '__main__':
    trace()
