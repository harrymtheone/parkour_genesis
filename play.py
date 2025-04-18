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
    args.simulator = SimulatorType.Genesis
    # args.simulator = SimulatorType.IsaacGym
    args.headless = False
    args.resume = True

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=args.task)

    # override some parameters for testing
    task_cfg.play.control = True
    task_cfg.env.num_envs = 3
    task_cfg.env.episode_length_s *= 10 if task_cfg.play.control else 1
    task_cfg.terrain.num_rows = 5
    task_cfg.terrain.max_init_terrain_level = task_cfg.terrain.num_rows - 1
    task_cfg.terrain.curriculum = True
    # task_cfg.terrain.max_difficulty = True
    # task_cfg.asset.disable_gravity = True

    # task_cfg.depth.position_range = [(-0.01, 0.01), (-0., 0.), (-0.0, 0.01)]  # front camera
    # task_cfg.depth.position_range = [(-0., 0.), (-0, 0), (-0., 0.)]  # front camera
    # task_cfg.depth.angle_range = [-1, 1]
    task_cfg.domain_rand.action_delay = False
    task_cfg.domain_rand.action_delay_range = [(10, 10)]
    task_cfg.domain_rand.push_robots = False
    task_cfg.domain_rand.push_duration = [0.15]
    # task_cfg.domain_rand.push_interval_s = 6

    task_cfg.terrain.terrain_dict = {
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
    task_cfg.terrain.num_cols = sum(task_cfg.terrain.terrain_dict.values())

    # prepare environment
    args.n_rendered_envs = task_cfg.env.num_envs
    task_cfg = task_registry.get_cfg(name=args.task)
    env = task_registry.make_env(args=args, task_cfg=task_cfg)
    obs, obs_critic = env.get_observations(), env.get_critic_observations()
    env.sim.clear_lines = True

    # load policy
    task_cfg.runner.resume = True
    runner = task_registry.make_alg_runner(task_cfg, args, log_root)

    with Live(gen_info_panel(args, env)) as live:
        for _ in range(10 * int(env.max_episode_length)):
            time_start = time.time()

            rtn = runner.play_act(obs, obs_critic=obs_critic, use_estimated_values=True, eval_=True)
            # rtn = runner.play_act(obs, obs_critic=obs_critic, use_estimated_values=random.random() > 0.5, eval_=True)

            if type(rtn) is tuple:
                if len(rtn) == 2:
                    actions, _ = rtn
                elif len(rtn) == 4:
                    actions, recon_rough, recon_refine, est = rtn
                    hmap_rough, edge_rough = recon_rough[:, 0], recon_rough[:, 1]
                    hmap_refine, edge_refine = recon_refine[:, 0], recon_refine[:, 1]

                    if len(hmap_rough) > 0:
                        args.est = est[env.lookat_id, :3] / 2

                        # env.draw_hmap(hmap_rough)
                        env.draw_hmap(hmap_refine)
                        # env.draw_body_edge(edge_refine)
                        # env.draw_est_hmap(est)

            else:
                actions = rtn

            # env.draw_hmap(obs.scan[:, 0])
            # env.draw_body_edge(critic_obs.base_edge_mask)
            # env.draw_hmap(scan - recon_refine - 1.0, world_frame=False)

            # for calibration of mirroring of dof
            # actions[:] = 0.
            # actions[env.lookat_id, 5] = env.joystick_handler.get_control_input()[0]
            # actions[env.lookat_id, 5 + 6] = env.joystick_handler.get_control_input()[0]

            # # for testing reference motion
            # actions[:] = 0.
            # actions[env.lookat_id] = (env.ref_dof_pos - env.init_state_dof_pos)[env.lookat_id, env.dof_activated]
            # actions[env.lookat_id] /= env.cfg.control.action_scale

            obs, obs_critic, rewards, dones, _ = env.step(actions)

            live.update(gen_info_panel(args, env))

            while time.time() - time_start < env.dt * slowmo:
                env.render()
                env.refresh_graphics(clear_lines=False)
            env.refresh_graphics(clear_lines=True)


if __name__ == '__main__':
    with torch.inference_mode():
        play(get_args())
