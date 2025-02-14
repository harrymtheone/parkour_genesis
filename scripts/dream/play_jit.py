import isaacgym, torch  # do not change this line

import os
import time

import numpy as np
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import TaskRegistry

slowmo = 1


def play(args):
    exptid = 'pdd_dream_001r'
    checkpoint = 3500

    args.proj_name = 'humanoid_parkour'
    args.headless = False
    args.resume = True

    task_registry = TaskRegistry()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.play.control = False
    env_cfg.play.use_joystick = True
    env_cfg.env.num_envs = 2
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
    env_cfg.domain_rand.push_interval_s = 4

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
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    model = torch.jit.load(os.path.join("traced", f'{exptid}_{checkpoint}_jit.pt'))
    model = model.to(args.device)
    model.eval()

    for _ in range(10 * int(env.max_episode_length)):
        time_start = time.time()

        actions = model(obs.proprio, obs.prop_his)
        obs, _, rewards, dones, _ = env.step(actions)

        cmd_vx, cmd_vy, cmd_yaw, _ = env.commands[env.lookat_id].cpu().numpy()
        real_vx, real_vy, _ = env.base_lin_vel[env.lookat_id].cpu().numpy()
        _, _, real_yaw = env.base_ang_vel[env.lookat_id].cpu().numpy()
        real_base_height = env.base_height.mean().cpu().numpy()

        print('\ntime: %.2f | cmd  vx %.2f | cmd  vy %.2f | cmd  yaw %.2f | target yaw %.2f | feet height %.1f, %.1f' % (
            env.episode_length_buf[env.lookat_id].item() / 50,
            cmd_vx,
            cmd_vy,
            cmd_yaw,
            env.target_yaw[env.lookat_id],
            *(env.feet_height[env.lookat_id] * 100)
        ))

        print('time: %.2f | real vx %.2f | real vy %.2f | real yaw %.2f | base height %.2f' % (
            env.episode_length_buf[env.lookat_id].item() / 50,
            real_vx,
            real_vy,
            real_yaw,
            real_base_height
        ))

        elapsed_time = time.time() - time_start
        if elapsed_time < env.dt * slowmo:
            time.sleep(env.dt * slowmo - elapsed_time)


if __name__ == '__main__':
    with torch.inference_mode():
        args = get_args()
        args.device = 'cpu'
        args.task = 'pdd_dreamwaq'
        play(args)


