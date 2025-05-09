import isaacgym, torch
import os

from torch import nn

from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.modules.model_dreamwaq import VAE
from rsl_rl.modules.utils import make_linear_layers

encoder_output_size = 3 + 64  # v_t, z_t


class Actor(nn.Module):
    is_recurrent = False

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        self.activation = nn.ELU()

        # construct actor network
        channel_size = 16

        self.obs_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=2 * channel_size, kernel_size=8, stride=4),
            self.activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=6, stride=1),
            self.activation,
            nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=6, stride=1),  # (8 * channel_size, 1)
            self.activation,
            nn.Flatten()
        )
        self.vae = VAE(env_cfg, policy_cfg)

        # Action noise
        self.actor_backbone = make_linear_layers(env_cfg.n_proprio + encoder_output_size,
                                                 *policy_cfg.actor_hidden_dims,
                                                 env_cfg.num_actions,
                                                 activation_func=self.activation,
                                                 output_activation=False)
        self.log_std = nn.Parameter(torch.zeros(env_cfg.num_actions))

    def forward(self, proprio, prop_his):
        obs_enc = self.obs_enc(prop_his.transpose(1, 2))
        est_mu = self.vae(obs_enc, mu_only=True)
        actor_input = torch.cat((proprio, self.activation(est_mu)), dim=1)
        return self.actor_backbone(actor_input), est_mu[..., :3]


def trace():
    # proj, cfg, exptid, checkpoint = 't1', 't1_dreamwaq', 't1_dream_005', 22000
    proj, cfg, exptid, checkpoint = 't1', 't1_dreamwaq', 't1_dream_006', 18700

    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=cfg)

    device = torch.device('cpu')
    load_path = os.path.join(f'../../logs/{proj}/{task_cfg.runner.algorithm_name}/', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = Actor(task_cfg.env, task_cfg.policy)
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

        trace_and_save(model, (proprio, prop_his))


if __name__ == '__main__':
    trace()
