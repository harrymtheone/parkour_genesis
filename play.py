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
    log_root = 'logs'
    # args.simulator = SimulatorType.Genesis
    args.simulator = SimulatorType.IsaacGym
    args.headless = False
    args.resume = True

    task_registry = TaskRegistry()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.play.control = True
    env_cfg.env.num_envs = 4
    env_cfg.env.episode_length_s *= 10 if env_cfg.play.control else 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_difficulty = False
    env_cfg.terrain.max_init_terrain_level = 4
    # env_cfg.asset.disable_gravity = True

    # env_cfg.depth.position_range = [(-0.01, 0.01), (-0., 0.), (-0.0, 0.01)]  # front camera
    # env_cfg.depth.position_range = [(-0., 0.), (-0, 0), (-0., 0.)]  # front camera
    # env_cfg.depth.angle_range = [-1, 1]
    # env_cfg.domain_rand.action_delay_range = [(5, 5)]
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.push_duration = [0.05, 0.1, 0.15]

    env_cfg.terrain.terrain_dict = {
        'smooth_slope': 0,
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
        'parkour_stair': 1,
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

    with Live(gen_info_panel(args, env), refresh_per_second=60) as live:
        for _ in range(10 * int(env.max_episode_length)):
            time_start = time.time()

            rtn = runner.play_act(obs, use_estimated_values=False)
            # rtn = runner.play_act(obs, use_estimated_values=random.random() > 0.6)

            if type(rtn) is tuple:
                actions, recon_rough, recon_refine = rtn

                # env.draw_hmap(recon_rough)
                env.draw_hmap(recon_refine)
                # env.draw_feet_hmap(est_mu[:, -16-16:-16])  # feet height map estimation
                # env.draw_body_hmap(est_mu[:, -16:])  # body height map estimation
            else:
                actions = rtn

            # env.draw_hmap(obs.scan)
            # env.draw_hmap(scan - recon_refine - 1.0, world_frame=False)

            # for calibration of mirroring of dof
            # actions.zero_()
            # actions[env.lookat_id, 5] = env.joystick_handler.get_control_input()[0]
            # actions[env.lookat_id, 5 + 6] = env.joystick_handler.get_control_input()[0]

            # # for testing reference motion
            # actions[env.lookat_id] = (env.ref_dof_pos - env.init_state_dof_pos)[env.lookat_id, env.dof_activated]
            # actions[env.lookat_id] /= env.cfg.control.action_scale

            obs, _, rewards, dones, _ = env.step(actions)

            live.update(gen_info_panel(args, env))

            while time.time() - time_start < env.dt * slowmo:
                env.render()


if __name__ == '__main__':
    with torch.inference_mode():
        play(get_args())
