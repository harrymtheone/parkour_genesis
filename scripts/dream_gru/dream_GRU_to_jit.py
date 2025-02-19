import os

import isaacgym, torch
from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.modules.model_dreamwaq import VAE
from rsl_rl.modules.utils import make_linear_layers
from torch import nn

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
        self.log_std = nn.Parameter(torch.log(policy_cfg.init_noise_std * torch.ones(env_cfg.num_actions)))

    def forward(self, proprio, hidden_states):
        # inference forward
        obs_enc, hidden_states = self.gru(proprio.unsqueeze(0), hidden_states)
        est_mu = self.vae(obs_enc.squeeze(0), mu_only=True)
        actor_input = torch.cat([proprio, self.activation(est_mu)], dim=1)
        return self.actor_backbone(actor_input), hidden_states


def trace():
    # exptid, checkpoint = 'pdd_dream_gru_004', 20000
    exptid, checkpoint = 'pdd_dream_gru_005', 30000

    trace_path = os.path.join('../dream_gru/traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    env_cfg, train_cfg = task_registry.get_cfgs(name='pdd_dreamwaq_gru')

    device = torch.device('cpu')
    load_path = os.path.join('../../logs/parkour_genesis/', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = ActorGRU(env_cfg.env, train_cfg.policy)
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
        proprio = torch.zeros(1, env_cfg.env.n_proprio, device=device)
        hidden_states = torch.zeros(1, 1, gru_hidden_size, device=device)

        trace_and_save(model, (proprio, hidden_states))


if __name__ == '__main__':
    trace()
