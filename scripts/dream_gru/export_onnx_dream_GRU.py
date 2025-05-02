try:
    import isaacgym, torch
except ImportError:
    import torch

import os

from torch import nn

from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.modules.model_dreamwaq import VAE
from rsl_rl.modules.utils import make_linear_layers

gru_hidden_size = 128
encoder_output_size = 3 + 64  # v_t, z_t


class ActorGRU(nn.Module):
    is_recurrent = True

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.activation = nn.ELU()

        gru_num_layers = 1
        self.gru = nn.GRU(input_size=env_cfg.n_proprio, hidden_size=gru_hidden_size, num_layers=gru_num_layers)

        self.vae = VAE(env_cfg, policy_cfg)

        self.actor_backbone = make_linear_layers(encoder_output_size + env_cfg.n_proprio,
                                                 *policy_cfg.actor_hidden_dims,
                                                 env_cfg.num_actions,
                                                 activation_func=self.activation)
        self.actor_backbone.pop(-1)

        # Action noise
        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))

    def forward(self, proprio, hidden_states):
        # inference forward
        obs_enc, hidden_states = self.gru(proprio.unsqueeze(0), hidden_states)
        est_mu = self.vae(obs_enc.squeeze(0), mu_only=True)
        actor_input = torch.cat([proprio, self.activation(est_mu)], dim=1)
        return self.actor_backbone(actor_input), hidden_states


def trace():
    # proj, cfg, exptid, checkpoint = 't1', 't1_dreamwaq', 't1_dream_003', 4900
    proj, cfg, exptid, checkpoint = 't1', 't1_dreamwaq', 't1_dream_002', 1500

    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=cfg)

    device = torch.device('cpu')
    load_path = os.path.join(f'../../logs/{proj}/{task_cfg.runner.algorithm_name}/', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = ActorGRU(task_cfg.env, task_cfg.policy)
    model.load_state_dict(state_dict['actor_state_dict'])
    model.eval()

    # model inputs
    proprio = torch.zeros(1, task_cfg.env.n_proprio, device=device)
    hidden_states = torch.zeros(1, 1, gru_hidden_size, device=device)

    test_action, test_hidden = model(proprio + 1, hidden_states + 1)

    torch.onnx.export(
        model,
        (proprio, hidden_states),
        "onnx/policy.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['proprio', 'hidden_states_in'],
        output_names=['action', 'hidden_states_out'],
    )


if __name__ == '__main__':
    trace()
