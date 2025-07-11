try:
    import isaacgym, torch
except ImportError:
    import torch

import os

from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.modules.model_odom import Actor, OdomTransformer
from torch import nn


class ActorJIT(Actor):
    def forward(self, proprio, recon, priv_est, hidden_states):
        recon_enc = self.scan_encoder(recon.flatten(1))
        x = torch.cat([proprio, recon_enc, priv_est], dim=1)

        x, hidden_states_new = self.gru(x.unsqueeze(0), hidden_states)
        x = x.squeeze(0)

        return self.actor(x), hidden_states_new


class OdometerToJit(OdomTransformer):

    def forward(self, prop, depth, hidden_states):
        enc = self.transformer_forward(prop, depth)

        out, hidden_states_new = self.gru(enc.unsqueeze(0), hidden_states)

        out = torch.cat([enc, out.squeeze(0)], dim=1)

        # reconstructor
        recon_rough = self.recon_rough(out)
        recon_refine, _ = self.recon_refine(recon_rough)

        # estimator
        est = self.estimator(out)

        return recon_rough, recon_refine, est, hidden_states_new


def trace_actor(proj, cfg, exptid, checkpoint):
    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=cfg)

    device = torch.device('cpu')
    load_path = os.path.join(f'../../logs/{proj}/{task_cfg.runner.algorithm_name}/', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = ActorJIT(task_cfg)
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
        recon = torch.zeros(1, 2, 32, 16, device=device)
        priv_est = torch.zeros(1, 3, device=device)
        hidden_states = torch.zeros(1, 1, task_cfg.policy.actor_gru_hidden_size, device=device)

        trace_and_save(model, (proprio, recon, priv_est, hidden_states))


def trace_odom(proj, cfg, exptid, checkpoint):
    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=cfg)

    device = torch.device('cpu')
    load_path = os.path.join(f'../../logs/{proj}/{task_cfg.runner.algorithm_name}/', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = OdometerToJit(
        task_cfg.env.n_proprio,
        task_cfg.odometer.odom_transformer_embed_dim,
        task_cfg.odometer.odom_gru_hidden_size,
        task_cfg.odometer.estimator_output_dim
    ).to(device)

    model.load_state_dict(state_dict['odometer_state_dict'])
    model.load_state_dict(torch.load('/home/harry/projects/parkour_genesis/logs/odom_online/odom_030r1/latest.pth', weights_only=True))
    model.eval()

    # define the trace function
    def trace_and_save(model, args):
        model(*args)
        model_jit = torch.jit.trace(model, args)
        model_jit(*args)
        model_path = os.path.join(trace_path, f'recon_jit.pt')

        try:
            torch.jit.save(model_jit, model_path)
            print("Saving model to:", os.path.abspath(model_path))
        except Exception as e:
            print(f"Failed to save files: {e}")

    with torch.no_grad():
        # Save the traced actor
        proprio = torch.zeros(1, task_cfg.env.n_proprio, device=device)
        depth = torch.zeros(1, 1, 64, 64, device=device)
        hidden_states = torch.zeros(2, 1, task_cfg.odometer.odom_gru_hidden_size, device=device)

        trace_and_save(model, (proprio, depth, hidden_states))


if __name__ == '__main__':
    kwargs = dict(proj='t1', cfg='t1_odom_finetune', exptid='t1_odom_030r1r1', checkpoint=22600)

    trace_actor(**kwargs)
    trace_odom(**kwargs)
