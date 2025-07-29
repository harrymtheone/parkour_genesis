try:
    import isaacgym, torch  # NOQA
except ImportError:
    import torch

import time

import cv2
from rich.live import Live

import vis
from legged_gym.simulator import SimulatorType
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import TaskRegistry

slowmo = 1


def play(args):
    log_root = 'logs'
    # args.simulator = SimulatorType.Genesis
    args.simulator = SimulatorType.IsaacGym
    args.headless = False
    args.resume = True

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=args.task)

    # override some parameters for testing
    task_cfg.play.control = False
    task_cfg.env.num_envs = 8
    task_cfg.env.episode_length_s *= 10 if task_cfg.play.control else 1
    task_cfg.terrain.num_rows = 5
    task_cfg.terrain.max_init_terrain_level = task_cfg.terrain.num_rows - 1
    task_cfg.terrain.curriculum = True
    # task_cfg.terrain.max_difficulty = True
    # task_cfg.asset.disable_gravity = True

    task_cfg.commands.resampling_time = 8

    # task_cfg.depth.position_range = [(-0.01, 0.01), (-0., 0.), (-0.0, 0.01)]  # front camera
    # task_cfg.depth.position_range = [(-0., 0.), (-0, 0), (-0., 0.)]  # front camera
    # task_cfg.depth.angle_range = [-1, 1]
    task_cfg.domain_rand.push_robots = False
    task_cfg.domain_rand.push_interval_s = 3
    task_cfg.domain_rand.push_duration = [0.3]
    task_cfg.domain_rand.action_delay = True
    task_cfg.domain_rand.action_delay_range = [(2, 2)]
    task_cfg.domain_rand.add_dof_lag = True
    task_cfg.domain_rand.dof_lag_range = (3, 3)
    task_cfg.domain_rand.randomize_torques = False
    task_cfg.domain_rand.randomize_gains = False
    task_cfg.domain_rand.joint_armature_range = {
        'default': dict(range=(0.01, 0.05), log_space=False),
    }

    task_cfg.rewards.only_positive_rewards = False

    task_cfg.terrain.description_type = 'trimesh'
    task_cfg.terrain.terrain_dict = {
        'smooth_slope': 0,
        'rough_slope': 0,
        'stairs_up': 0,
        'stairs_down': 0,
        'huge_stair': 0,
        'discrete': 0,
        'stepping_stone': 0,
        'gap': 0,
        'pit': 0,
        'parkour_flat': 0,
        'parkour': 0,
        'parkour_gap': 0,
        'parkour_box': 0,
        'parkour_step': 0,
        'parkour_stair': 1,
        'parkour_stair_down': 1,
        'parkour_mini_stair': 1,
        'parkour_mini_stair_down': 1,
        'parkour_go_back_stair': 0,
    }
    task_cfg.terrain.num_cols = sum(task_cfg.terrain.terrain_dict.values())

    # prepare environment
    args.n_rendered_envs = task_cfg.env.num_envs
    task_cfg = task_registry.get_cfg(name=args.task)
    env = task_registry.make_env(args=args, task_cfg=task_cfg)
    obs, obs_critic = env.get_observations(), env.get_critic_observations()
    dones = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)

    # load policy
    env.sim.clear_lines = True
    task_cfg.runner.resume = True
    task_cfg.runner.logger_backend = None
    runner = task_registry.make_alg_runner(task_cfg, args, log_root)

    runner.odom.odom.load_state_dict(torch.load('/home/harry/projects/parkour_genesis/logs/odom_online/2025-07-29_10-10-09/latest.pth',
                                                map_location=args.device,
                                                weights_only=True))

    with Live(vis.gen_info_panel(args, env)) as live:
        for step_i in range(10 * int(env.max_episode_length)):
            time_start = time.time()

            rtn = runner.play_act(obs, obs_critic=obs_critic, use_estimated_values=False, eval_=False, dones=dones)
            # rtn = runner.play_act(obs, obs_critic=obs_critic, use_estimated_values=random.random() > 0.9, eval_=True, dones=dones)

            actions = rtn['actions']

            if 'wm_depth' in rtn:
                depth_img = rtn['wm_depth'][env.lookat_id, 0]

                img = torch.clip((depth_img + 0.5) * 255, 0, 255).to(torch.uint8)
                cv2.imshow("wm_depth", cv2.resize(img.cpu().numpy(), (img.shape[0] * 5, img.shape[1] * 5)))
                cv2.waitKey(1)

            if 'recon_refine' in rtn:
                # recon_rough = rtn['recon_rough']
                recon_refine = rtn['recon_refine']
                est = rtn['estimation']

                args.est_vel = est[env.lookat_id, :3] / 2
                args.est_height = est[env.lookat_id, 3]
                args.recon_loss = torch.nn.functional.l1_loss(obs.scan[env.lookat_id, 1], recon_refine[env.lookat_id, 1])

                # env._draw_body_hmap(recon_rough[env.lookat_id])
                # env.draw_recon(recon_refine[env.lookat_id])
                # env.draw_est_hmap(est)
                # env.draw_hmap(scan - recon_refine - 1.0, world_frame=False)
                # env.draw_recon(obs.scan[env.lookat_id])

                recon_refine = recon_refine[env.lookat_id].clone()
                # recon_refine[0] = - recon_refine[0] - 0.7
                # recon_refine[0] = recon_refine[0] - task_cfg.normalization.scan_norm_bias + est[env.lookat_id, 3]
                recon_refine[0] = recon_refine[0] - task_cfg.normalization.scan_norm_bias + obs_critic.priv_actor[env.lookat_id, 3]
                recon_refine[1] += 0.5
                env.draw_recon(recon_refine)

            # elif hasattr(obs, 'scan'):
            #     noisy_scan = obs.scan[env.lookat_id].clone()
            #     # noisy_scan[0] = - noisy_scan[0] - 0.7
            #     noisy_scan[0] = noisy_scan[0] - task_cfg.normalization.scan_norm_bias + env.base_height[env.lookat_id]
            #     env.draw_recon(noisy_scan)

            # # for calibration of mirroring of dof
            # actions[:] = 0.
            # actions[env.lookat_id, 5] = env.joystick_handler.get_control_input()[0]
            # actions[env.lookat_id, 5 + 6] = env.joystick_handler.get_control_input()[0]

            # # for testing reference motion
            # actions[:] = 0.
            # actions[env.lookat_id] = (env.ref_dof_pos - env.init_state_dof_pos)[env.lookat_id, env.dof_activated]
            # actions[env.lookat_id] /= env.cfg.control.action_scale

            obs, obs_critic, rewards, dones, extras = env.step(actions)

            live.update(vis.gen_info_panel(args, env))

            while time.time() - time_start < env.dt * slowmo:
                env.render()
                env.refresh_graphics(clear_lines=False)
            env.refresh_graphics(clear_lines=True)

            # t1_vis.plot(env)


if __name__ == '__main__':
    # t1_vis = vis.RewVisualizer()
    # t1_vis = vis.T1ActionsVisualizer()
    # t1_vis = vis.T1DofPosVisualizer()
    # t1_vis = vis.T1DofVelVisualizer()
    # t1_vis = vis.T1TorqueVisualizer()

    with torch.inference_mode():
        play(get_args())
