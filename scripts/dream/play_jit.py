import os
from argparse import Namespace

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


def play():
    args = Namespace()

    proj, cfg, exptid, checkpoint = 't1', 't1_dreamwaq', 't1_dream_006', 1800

    args.proj_name = proj
    args.device = 'cpu'
    args.task = cfg
    args.exptid = exptid
    args.checkpoint = checkpoint

    # args.simulator = SimulatorType.Genesis
    args.simulator = SimulatorType.IsaacGym
    args.headless = False
    args.resume = True

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=args.task)

    # override some parameters for testing
    task_cfg.play.control = False
    task_cfg.play.use_joystick = True
    task_cfg.env.num_envs = 1
    task_cfg.env.episode_length_s *= 10 if task_cfg.play.control else 1
    task_cfg.terrain.num_rows = 10
    task_cfg.terrain.curriculum = True
    task_cfg.terrain.max_difficulty = False
    task_cfg.terrain.max_init_terrain_level = 9
    # env_cfg.asset.disable_gravity = True

    task_cfg.domain_rand.push_robots = False
    task_cfg.domain_rand.push_interval_s = 6
    task_cfg.domain_rand.push_duration = [0.05, 0.1, 0.15]

    task_cfg.domain_rand.randomize_joint_armature = True
    task_cfg.domain_rand.joint_armature_range = {
        'default': dict(range=(0.1, 0.1), log_space=False),
    }

    task_cfg.terrain.terrain_dict = {
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
    task_cfg.terrain.num_cols = sum(task_cfg.terrain.terrain_dict.values())

    # prepare environment
    args.n_rendered_envs = task_cfg.env.num_envs
    env = task_registry.make_env(args=args, task_cfg=task_cfg)
    obs = env.get_observations()

    model = torch.jit.load(os.path.join("traced", f'{exptid}_{checkpoint}_jit.pt'))
    model = model.to(args.device)
    model.eval()

    hidden_states = torch.zeros(1, 1, 128, device=args.device)

    with Live(gen_info_panel(args, env), refresh_per_second=60) as live:
        for _ in range(10 * int(env.max_episode_length)):
            time_start = time.time()

            obs.proprio[:, 8:11] = 0.
            obs.proprio[:, 8] = 0.3
            print(env.commands[env.lookat_id].cpu().numpy())
            actions, hidden_states = model(obs.proprio, hidden_states)

            obs, _, rewards, dones, _ = env.step(actions)

            live.update(gen_info_panel(args, env))

            while time.time() - time_start < env.dt * slowmo:
                env.render()


if __name__ == '__main__':
    with torch.inference_mode():
        play()
