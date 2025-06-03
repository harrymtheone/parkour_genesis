import cv2

try:
    import isaacgym, torch
except ImportError:
    import torch

import time

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
    task_cfg.env.num_envs = 16
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
    task_cfg.domain_rand.action_delay_range = [(0, 5)]
    task_cfg.domain_rand.add_dof_lag = False
    task_cfg.domain_rand.dof_lag_range = (0, 10)
    task_cfg.domain_rand.push_robots = False
    task_cfg.domain_rand.push_duration = [0.3]
    task_cfg.domain_rand.push_interval_s = 8

    task_cfg.terrain.terrain_dict = {
        'smooth_slope': 1,
        'rough_slope': 1,
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
        'parkour_mini_stair': 0,
        'parkour_flat': 0,
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
    runner = task_registry.make_alg_runner(task_cfg, args, log_root)

    with Live(vis.gen_info_panel(args, env)) as live:
        for step_i in range(10 * int(env.max_episode_length)):
            time_start = time.time()

            rtn = runner.play_act(obs, obs_critic=obs_critic, eval_=True, dones=dones)
            # rtn = runner.play_act(obs, obs_critic=obs_critic, use_estimated_values=True, eval_=True)
            # rtn = runner.play_act(obs, obs_critic=obs_critic, use_estimated_values=random.random() > 0.5, eval_=True)

            actions = rtn['actions']

            if 'wm_depth' in rtn:
                depth_img = rtn['wm_depth'][env.lookat_id, 0]

                img = torch.clip((depth_img + 0.5) * 255, 0, 255).to(torch.uint8)
                cv2.imshow("wm_depth", cv2.resize(img.cpu().numpy(), (img.shape[0] * 5, img.shape[1] * 5)))
                cv2.waitKey(1)

            # if type(rtn) is tuple:
            #     if len(rtn) == 2:
            #         actions, _ = rtn
            #     elif len(rtn) == 4:
            #         actions, recon_rough, recon_refine, est = rtn
            #
            #         if len(recon_rough) > 0:
            #             args.est = est[env.lookat_id, :3] / 2
            #
            #             env.draw_recon(recon_rough)
            #             # env.draw_recon(recon_refine)
            #             # env.draw_est_hmap(est)
            # else:
            #     actions = rtn

            # env.draw_recon(obs.scan)
            # env.draw_hmap(scan - recon_refine - 1.0, world_frame=False)

            # # for calibration of mirroring of dof
            # actions[:] = 0.
            # actions[env.lookat_id, 8] = env.joystick_handler.get_control_input()[0]
            # actions[env.lookat_id, 8 + 3] = env.joystick_handler.get_control_input()[0]

            # # for testing reference motion
            # actions[:] = 0.
            # actions[env.lookat_id] = (env.ref_dof_pos - env.init_state_dof_pos)[env.lookat_id, env.dof_activated]
            # actions[env.lookat_id] /= env.cfg.control.action_scale

            obs, obs_critic, rewards, dones, _ = env.step(actions)

            live.update(vis.gen_info_panel(args, env))

            while time.time() - time_start < env.dt * slowmo:
                env.render()
                env.refresh_graphics(clear_lines=False)
            env.refresh_graphics(clear_lines=True)

            # # visualize FFT
            # joint_idx = 4
            # feet_act_his.append(actions[env.lookat_id, joint_idx].item())
            # if step_i % 50 == 49:
            #     X = np.fft.fft(feet_act_his)
            #     freqs = np.fft.fftfreq(len(feet_act_his), d=env.dt)
            #
            #     sorted_indices = np.argsort(freqs)
            #     freqs_sorted = freqs[sorted_indices]
            #     X_sorted = X[sorted_indices]
            #
            #     line0.set_data(range(len(feet_act_his)), env.actions_his_buf.get_all()[:, env.lookat_id, joint_idx].tolist()[::-1])
            #     line1.set_data(freqs_sorted, np.abs(X_sorted))
            #     line2.set_data(range(len(feet_act_his)), env.actions_filtered_his_buf.get_all()[:, env.lookat_id, joint_idx].tolist()[::-1])
            #
            #     axs[0].relim()
            #     axs[0].autoscale_view()
            #     axs[1].relim()
            #     axs[1].autoscale_view()
            #     axs[2].relim()
            #     axs[2].autoscale_view()
            #
            #     plt.draw()
            #     plt.pause(0.1)  # Pause for GUI update
            #     feet_act_his.clear()

            actions = actions.cpu().numpy()
            # t1_vis.plot({
            #     'Waist': actions[env.lookat_id, 0],
            #     'Left_Hip_Pitch': actions[env.lookat_id, 1],
            #     'Left_Hip_Roll': actions[env.lookat_id, 2],
            #     'Left_Hip_Yaw': actions[env.lookat_id, 3],
            #     'Left_Knee_Pitch': actions[env.lookat_id, 4],
            #     'Left_Ankle_Pitch': actions[env.lookat_id, 5],
            #     'Left_Ankle_Roll': actions[env.lookat_id, 6],
            #     'Right_Hip_Pitch': actions[env.lookat_id, 7],
            #     'Right_Hip_Roll': actions[env.lookat_id, 8],
            #     'Right_Hip_Yaw': actions[env.lookat_id, 9],
            #     'Right_Knee_Pitch': actions[env.lookat_id, 10],
            #     'Right_Ankle_Pitch': actions[env.lookat_id, 11],
            #     'Right_Ankle_Roll': actions[env.lookat_id, 12],
            # })

            dof_vel = env.sim.dof_vel.cpu().numpy()
            # t1_vis.plot1{
            #     'Left_Hip_Pitch': dof_vel[env.lookat_id, 11],
            #     'Left_Hip_Roll': dof_vel[env.lookat_id, 12],
            #     'Left_Hip_Yaw': dof_vel[env.lookat_id, 13],
            #     'Left_Knee_Pitch': dof_vel[env.lookat_id, 14],
            #     'Left_Ankle_Pitch': dof_vel[env.lookat_id, 15],
            #     'Left_Ankle_Roll': dof_vel[env.lookat_id, 16],
            #     'Right_Hip_Pitch': dof_vel[env.lookat_id, 17],
            #     'Right_Hip_Roll': dof_vel[env.lookat_id, 18],
            #     'Right_Hip_Yaw': dof_vel[env.lookat_id, 19],
            #     'Right_Knee_Pitch': dof_vel[env.lookat_id, 20],
            #     'Right_Ankle_Pitch': dof_vel[env.lookat_id, 21],
            #     'Right_Ankle_Roll': dof_vel[env.lookat_id, 22],
            # })

            # euler = env.projected_gravity.cpu().numpy()
            # t1_vis.plot({
            #     'X': euler[env.lookat_id, 0],
            #     'Y': euler[env.lookat_id, 1],
            #     'Z': euler[env.lookat_id, 2],
            # })


if __name__ == '__main__':
    # plt.ion()
    #
    # fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    # line0, = axs[0].plot([], [])
    # line1, = axs[1].plot([], [])
    # line2, = axs[2].plot([], [])

    # t1_vis = vis.T1ActionsVisualizer()
    # t1_vis = vis.T1DofVelVisualizer()
    # t1_vis = T1GravityVisualizer()

    with torch.inference_mode():
        play(get_args())
