import isaacgym, torch
import os

from torch import nn

from legged_gym.utils.task_registry import TaskRegistry


class Actor(nn.Module):
    is_recurrent = False

    def __init__(self, env_cfg, policy_cfg):
        super().__init__()

        activation = nn.ELU()

        # construct actor network
        channel_size = 16

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.n_proprio, out_channels=2 * channel_size, kernel_size=8, stride=4),
            activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=6, stride=1),
            activation,
            nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=6, stride=1),
            activation,
            nn.Flatten(),
            nn.Linear(8 * channel_size, 4 * channel_size),
            activation,
            nn.Linear(4 * channel_size, env_cfg.num_actions),
        )

        # Action noise
        self.log_std = nn.Parameter(policy_cfg.init_noise_std * torch.ones(env_cfg.num_actions))

    def forward(self, prop_his):
        return self.net(prop_his.transpose(1, 2))


def trace():
    # exptid = 'pdd_dream_006'
    # checkpoint = 2900
    # exptid, checkpoint = 'pdd_dream_006_st05', 5800
    # exptid, checkpoint = 'pdd_dream_006_st07', 10000
    exptid, checkpoint = 'pdd_dream_006_st08', 20000
    exptid, checkpoint = 'pdd_dream_007r1', 40000

    trace_path = os.path.join('./traced')
    if not os.path.exists(trace_path):
        os.mkdir(trace_path)

    task_registry = TaskRegistry()
    env_cfg, train_cfg = task_registry.get_cfgs(name='pdd_dreamwaq')

    device = torch.device('cpu')
    load_path = os.path.join('../../logs/humanoid_parkour/', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = Actor(env_cfg.env, train_cfg.policy)
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
        prop_his = torch.zeros(1, env_cfg.env.len_prop_his, env_cfg.env.n_proprio, device=device)

        trace_and_save(model, (prop_his,))


if __name__ == '__main__':
    trace()
