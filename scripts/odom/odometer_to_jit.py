try:
    import isaacgym, torch
except ImportError:
    import torch

import os

from rsl_rl.modules.odometer.recurrent import OdomRecurrentTransformer


class OdometerToJit(OdomRecurrentTransformer):

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


def trace():
    load_path = '/home/harry/projects/parkour_genesis/logs/odom_online/2025-06-27_18-33-19/latest.pth'
    device = torch.device('cpu')
    trace_path = os.path.join('./traced')

    model = OdometerToJit(50, 64, 128, 3)
    model.load_state_dict(torch.load(load_path, weights_only=True))

    def trace_and_save(model, args):
        model(*args)
        model_jit = torch.jit.trace(model, args)
        model_jit(*args)
        model_path = os.path.join(trace_path, f'odom_013_odometer1_jit.pt')

        try:
            torch.jit.save(model_jit, model_path)
            print("Saving model to:", os.path.abspath(model_path))
        except Exception as e:
            print(f"Failed to save files: {e}")

    proprio = torch.zeros(1, 50, device=device)
    depth = torch.zeros(1, 1, 64, 64, device=device)
    hidden_states = torch.zeros(2, 1, 128, device=device)

    with torch.no_grad():
        trace_and_save(model, (proprio, depth, hidden_states))


if __name__ == '__main__':
    trace()
