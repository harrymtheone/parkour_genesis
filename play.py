try:
    import isaacgym, torch
except ImportError:
    import torch

import time

import numpy as np
from rich.live import Live
from rich.table import Table

from legged_gym.simulator import SimulatorType
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import TaskRegistry

slowmo = 1


def play(args):
    args.proj_name = 'parkour_genesis'
    log_root = 'logs'
    # args.simulator = SimulatorType.Genesis
    args.simulator = SimulatorType.IsaacGym
    args.headless = False
    args.resume = True

    # rtv = RealTimeVisualizer(3, 1, (-30.0, 30.0))
    # rtv = RealTimeVisualizer(2, 2, (-1., 1.0))
    # rtv = RealTimeVisualizer(6, 2, (-2.5, 2.))

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

    # load policy
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env, log_root, args=args, train_cfg=train_cfg)

    terrain_class, terrain_env_counts = torch.unique(env.env_class, return_counts=True)
    coefficient_variation = torch.ones_like(terrain_class)
    cur_reward_sum = torch.zeros(env_cfg.env.num_envs, dtype=torch.float, device=args.device)
    mean_env_reward = torch.zeros(env_cfg.env.num_envs, dtype=torch.float, device=args.device)

    with Live(gen_info_panel(args, env), refresh_per_second=60) as live:
        for _ in range(10 * int(env.max_episode_length)):
            time_start = time.time()

            rtn = runner.play_act(obs, use_estimated_values=False)
            # rtn = runner.play_act(obs, use_estimated_values=random.random() > 0.6)

            if type(rtn) is tuple:
                actions, recon_rough, recon_refine, est_mu = rtn

                if type(est_mu) is tuple:
                    est_mu, est_logvar, ot1 = est_mu

                # env.draw_height_samples(recon_rough, world_frame=False)
                env.draw_height_samples(recon_refine, world_frame=False)
                # env.draw_feet_hmap(est_mu[:, -16-16:-16])  # feet height map estimation
                # env.draw_body_hmap(est_mu[:, -16:])  # body height map estimation
            else:
                actions = rtn
            # print(actions[env.lookat_id])
            # scan = obs[1].scan if type(obs) is tuple else obs.scan
            # env.draw_height_samples(scan, world_frame=False)
            # env.draw_height_samples(scan - recon_refine - 1.0, world_frame=False)

            # # for testing reference motion
            # actions[env.lookat_id] = (env.ref_dof_pos - env.default_dof_pos_all)[env.lookat_id]
            # actions[env.lookat_id] /= env.cfg.control.action_scale

            obs, _, rewards, dones, _ = env.step(actions)

            # cur_reward_sum += rewards
            # new_ids = (dones > 0).nonzero(as_tuple=False)
            # mean_env_reward[new_ids] = 0.9 * mean_env_reward[new_ids] + 0.1 * cur_reward_sum[new_ids]
            # cur_reward_sum[new_ids] = 0
            #
            # for i, t in enumerate(terrain_class):
            #     rew_terrain = mean_env_reward[env.env_class == t]
            #     coefficient_variation[i] = rew_terrain.std() / (rew_terrain.mean().abs() + 1e-5)
            #
            # p_smpl = torch.sum(coefficient_variation * terrain_env_counts / terrain_env_counts.sum())
            # p_smpl = torch.tanh(p_smpl).item()
            # # print(p_smpl)

            live.update(gen_info_panel(args, env))

            # rtv.update([env.rew_buf[env.lookat_id] / env.dt, value[env.lookat_id, 0]])
            # rtv.update([*env.dof_vel[env.lookat_id, 3:6].cpu().numpy()])
            # rtv.update([*env.commands[0]])
            # rtv.update(actions[env.lookat_id])

            while time.time() - time_start < env.dt * slowmo:
                env.render()


def gen_info_panel(args, env):
    cmd_vx, cmd_vy, cmd_yaw, _ = env.commands[env.lookat_id].cpu().numpy()
    real_vx, real_vy, _ = env.base_lin_vel[env.lookat_id].cpu().numpy()
    _, _, real_yaw = env.base_ang_vel[env.lookat_id].cpu().numpy()
    real_base_height = env.base_height[env.lookat_id].cpu().numpy()

    if args.headless:
        perc_contact_forces = torch.mean(env.contact_forces_avg, dim=0).cpu().numpy()
        perc_feet_air_time = torch.mean(env.feet_air_time_avg, dim=0).cpu().numpy()
    else:
        perc_contact_forces = env.contact_forces_avg[env.lookat_id].cpu().numpy()
        perc_feet_air_time = env.feet_air_time_avg[env.lookat_id].cpu().numpy()

    perc_feet_air_time = perc_feet_air_time / np.sum(perc_feet_air_time + 1e-5)
    # print(f'env level: {env.env_levels[env.lookat_id].cpu().numpy()}')

    table11 = Table()
    table11.add_column(f'time: {env.episode_length_buf[env.lookat_id].item() / 50:.2f}')
    table11.add_column("vx")
    table11.add_column("vy")
    table11.add_column("yaw")
    table11.add_row("cmd", f'{cmd_vx: .2f}', f'{cmd_vy: .2f}', f'{cmd_yaw: .2f}')
    table11.add_row("real", f'{real_vx: .2f}', f'{real_vy: .2f}', f'{real_yaw: .2f}')

    table12 = Table()
    table12.add_column("target_yaw")
    table12.add_column("base_height")
    table12.add_row(f'{env.target_yaw[env.lookat_id]: .2f}', f'{real_base_height: .2f}')

    table21 = Table()
    table21.add_column("")
    table21.add_column("Left")
    table21.add_column("Right")
    table21.add_row("Contact forces", f'{perc_contact_forces[0]: .2f}', f'{perc_contact_forces[1]: .2f}')

    table22 = Table()
    table22.add_column("")
    table22.add_column("Left")
    table22.add_column("Right")
    table22.add_row("Feet air time", f'{perc_feet_air_time[0]: .2f}', f'{perc_feet_air_time[1]: .2f}')

    grid = Table.grid()
    grid.add_row(table11, table12)
    grid.add_row(table21, table22)

    return grid


if __name__ == '__main__':
    with torch.inference_mode():
        play(get_args())

        # from legged_gym.envs.base.legged_robot import LeggedRobot
        # from legged_gym.utils.math import point_cloud_to_voxel_grid_jit
        #
        # lp = LineProfiler()
        # lp.add_function(LeggedRobot._draw_test)
        # lp.add_function(point_cloud_to_voxel_grid_jit)
        # lp.add_function(LeggedRobot._post_physics_step)
        # wrapper = lp(play)
        # wrapper(get_args())
        # lp.print_stats()
