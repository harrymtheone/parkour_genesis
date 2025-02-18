import os

try:
    import isaacgym, torch
except ImportError:
    import torch

import time

from rich.live import Live

from legged_gym.simulator import SimulatorType
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import TaskRegistry
from vis import gen_info_panel

slowmo = 1


def play(args):
    exptid, checkpoint = 'pdd_dream_gru_003', 000

    args.proj_name = 'parkour_genesis'
    log_root = 'logs'
    # args.simulator = SimulatorType.Genesis
    args.simulator = SimulatorType.IsaacGym
    args.headless = False
    args.resume = True

    task_registry = TaskRegistry()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.play.control = False
    env_cfg.play.use_joystick = True
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s *= 10 if env_cfg.play.control else 1
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_difficulty = False
    env_cfg.terrain.max_init_terrain_level = 9
    # env_cfg.asset.disable_gravity = True

    # env_cfg.depth.position_range = [(-0.01, 0.01), (-0., 0.), (-0.0, 0.01)]  # front camera
    env_cfg.depth.position_range = [(-0., 0.), (-0, 0), (-0., 0.)]  # front camera
    env_cfg.depth.angle_range = [-1, 1]
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.push_duration = [0.05, 0.1, 0.15]

    env_cfg.terrain.terrain_dict = {
        'smooth_slope': 1,
        'rough_slope': 0,
        'stairs_up': 0,
        'stairs_down': 0,
        'discrete': 0,
        'stepping_stone': 0,
        'gap': 0,
        'pit': 0,
        'parkour': 0,
        'parkour_gap': 0,
        'parkour_box': 0,
        'parkour_step': 0,
        'parkour_stair': 0,
        'parkour_flat': 0,
    }
    env_cfg.terrain.num_cols = sum(env_cfg.terrain.terrain_dict.values())

    # prepare environment
    args.n_rendered_envs = env_cfg.env.num_envs
    env, _ = task_registry.make_env(args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    model = torch.jit.load(os.path.join("traced", f'{exptid}_{checkpoint}_jit.pt'))
    model = model.to(args.device)
    model.eval()

    hidden_states = torch.zeros(1, 1, 128, device=args.device)

    with Live(gen_info_panel(args, env), refresh_per_second=60) as live:
        for _ in range(10 * int(env.max_episode_length)):
            time_start = time.time()

            actions, hidden_states = model(obs.proprio, hidden_states)

            obs, _, rewards, dones, _ = env.step(actions)

            live.update(gen_info_panel(args, env))

            while time.time() - time_start < env.dt * slowmo:
                env.render()


if __name__ == '__main__':
    with torch.inference_mode():
        play(get_args())
