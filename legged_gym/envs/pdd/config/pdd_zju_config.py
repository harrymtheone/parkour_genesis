import numpy as np

from .pdd_base_config import PddBaseCfg, PddBaseCfgPPO


class PddZJUCfg(PddBaseCfg):
    class env(PddBaseCfg.env):
        num_envs = 4096  # 6144

        n_proprio = 41
        len_prop_his = 10

        len_depth_his = 2
        scan_shape = (32, 16)
        n_scan = scan_shape[0] * scan_shape[1]

        num_critic_obs = 53 + 24  # +32 feet_hmap
        len_critic_his = 50

        num_actions = 10
        episode_length_s = 30  # episode length in seconds

    class sensors:
        activated = True

        class depth_0:
            position = [0.10, 0, 0.0]  # front camera
            position_range = [(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)]  # front camera
            pitch = 60  # positive pitch down (degree)
            pitch_range = [-1, 1]  # positive pitch down (degree)

            update_interval = 5  # 5 works without retraining, 8 worse
            delay_prop = (5, 1)  # Gaussian (mean, std)

            resolution = (106, 60)  # width, height
            resized = (87, 58)  # (87, 58)
            horizontal_fov = 87

            near_clip = 0
            far_clip = 2
            dis_noise_global = 0.1  # in meters
            dis_noise_gaussian = 0.05  # in meters

    class terrain(PddBaseCfg.terrain):
        scan_pts_x = np.linspace(-0.5, 1.1, 32)
        scan_pts_y = np.linspace(-0.4, 0.4, 16)
        body_pts_x = np.linspace(-0.2, 0.2, 4)
        body_pts_y = np.linspace(-0.2, 0.2, 4)
        feet_pts_x = np.linspace(-0.1, 0.1, 2)
        feet_pts_y = np.linspace(-0.1, 0.1, 2)

        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        num_cols = 20  # number of terrain cols (types)

        terrain_dict = {
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
            'parkour_stair': 4,
            'parkour_flat': 0,
        }

    class commands:
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 8.  # time before command are changed[s]

        lin_vel_clip = 0.1
        ang_vel_clip = 0.4

        sw_switch = True

        class flat_ranges:
            lin_vel_x = [-0.2, 0.5]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-1.0, 1.0]

        class stair_ranges:
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-1.0, 1.0]  # this value limits the max yaw velocity computed by goal
            heading = [-1.0, 1.0]

        class parkour_ranges:
            lin_vel_x = [0.4, 0.8]  # min value should be greater than lin_vel_clip
            ang_vel_yaw = [-1.0, 1.0]  # this value limits the max yaw velocity computed by goal

    class domain_rand(PddBaseCfg.domain_rand):
        switch = False

        randomize_start_pos = switch
        randomize_start_y = switch
        randomize_start_yaw = switch
        randomize_start_vel = switch
        randomize_start_pitch = switch

        randomize_start_dof_pos = True
        randomize_start_dof_vel = True

        randomize_friction = switch
        randomize_base_mass = switch
        randomize_link_mass = switch
        randomize_com = switch

        push_robots = switch
        action_delay = switch
        add_dof_lag = False
        add_imu_lag = False

        randomize_torque = switch
        randomize_gains = switch
        randomize_motor_offset = switch
        randomize_joint_stiffness = False  # for joints with spring behavior, (not implemented yet)
        randomize_joint_damping = False
        randomize_joint_friction = False
        randomize_joint_armature = switch
        randomize_coulomb_friction = switch

    class rewards:
        base_height_target = 0.6
        feet_height_target = 0.04
        feet_height_target_max = 0.07
        use_guidance_terrain = True
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 5
        soft_dof_pos_limit = 0.9
        EMA_update_alpha = 0.99

        cycle_time = 0.64  # 0.64
        target_joint_pos_scale = 0.19  # 0.3

        min_dist = 0.18
        max_dist = 0.50
        max_contact_force = 300

        rew_norm_factor = 1.0

        class scales:
            # gait
            joint_pos = 2.
            feet_clearance = 1.2  # 0.2
            feet_contact_number = 1.2
            feet_air_time = 1.
            foot_slip = -1.
            feet_distance = 0.2
            knee_distance = 0.2
            # feet_rotation = 0.3

            # contact
            feet_contact_forces = -0.1

            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5
            low_speed = 0.2
            track_vel_hard = 0.5

            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2

            # energy
            action_smoothness = -0.003
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.
            # stand_still = 2.0


class PddZJUCfgPPO(PddBaseCfgPPO):
    seed = -1
    runner_name = 'rl_dream'  # rl, distil, mixed
    algorithm_name = 'ppo_zju'

    class policy:
        # actor parameters
        actor_hidden_dims = [512, 256, 128]  # [128, 64, 32]
        init_noise_std = 1.0

        # critic parameters
        critic_hidden_dims = [512, 256, 128]

        use_recurrent_policy = True

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        continue_from_last_std = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 10
        num_mini_batches = 4  # mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 2.e-4  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

        use_amp = True

    class runner(PddBaseCfgPPO.runner):
        max_iterations = 50000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
