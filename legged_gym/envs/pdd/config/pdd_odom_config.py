import numpy as np

from .pdd_base_config import PddBaseCfg


class PddOdomCfg(PddBaseCfg):
    class env(PddBaseCfg.env):
        num_envs = 4096  # 6144
        n_proprio = 41
        len_prop_his = 50

        scan_shape = (32, 16)
        n_scan = scan_shape[0] * scan_shape[1]

        num_critic_obs = 61
        len_critic_his = 50

        num_actions = 10
        episode_length_s = 30  # episode length in seconds

    class sensors:
        activated = False

        class depth_0:
            link_attached_to = 'base_link'
            position = [0.07, 0, 0.09]  # front camera
            position_range = [(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)]  # front camera
            pitch = 0  # positive is looking down
            pitch_range = [-3, 3]

            data_format = 'depth'  # depth, cloud, hmap
            update_interval = 1
            delay_prop = None  # Gaussian (mean, std), or None
            history_length = 1

            resolution = (114, 64)  # width, height
            crop = (0, 2, 4, 4)  # top, bottom, left, right

            edge_process = True
            edge_noise = dict(blank_ratio=0.2, repeat_ratio=0.2)
            blank_ratio = 0.002

            near_clip = 0
            far_clip = 2
            dis_noise_global = 0.01  # in meters
            dis_noise_gaussian = 0.01  # in meters
            noise_scale_perlin = 1  # 0-1

            resized = (64, 64)
            horizontal_fov = 87

            bounding_box = (0.3, 1.1, -0.4, 0.4)  # x1, x2, y1, y2
            hmap_shape = (16, 16)  # x dim, y dim

    class commands:
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 8.  # time before command are changed[s]

        lin_vel_clip = 0.1
        ang_vel_clip = 0.2
        parkour_vel_tolerance = 0.3

        cycle_time = 0.64  # 0.64
        target_joint_pos_scale = 0.19  # 0.19

        sw_switch = True
        phase_offset_l = 0.
        phase_offset_r = 0.5
        air_ratio = 0.4
        delta_t = 0.02

        class flat_ranges:
            lin_vel_x = [-0.4, 0.6]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-1., 1.]

        class stair_ranges:
            lin_vel_x = [-0.3, 0.6]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-1., 1.]  # this value limits the max yaw velocity computed by goal
            heading = [-1.5, 1.5]

        class parkour_ranges:
            lin_vel_x = [0.3, 0.6]  # min value should be greater than lin_vel_clip
            ang_vel_yaw = [-1.0, 1.0]  # this value limits the max yaw velocity computed by goal

    class terrain(PddBaseCfg.terrain):
        body_pts_x = np.linspace(-0.6, 1.2, 32)
        body_pts_y = np.linspace(-0.6, 0.6, 16)

        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        num_cols = 20  # number of terrain cols (types)

        terrain_dict = {
            'smooth_slope': 1,
            'rough_slope': 1,
            'stairs_up': 2,
            'stairs_down': 2,
        }

    class noise(PddBaseCfg.noise):
        add_noise = True

    class domain_rand(PddBaseCfg.domain_rand):
        switch = True

        randomize_start_pos = switch
        randomize_start_y = False
        randomize_start_yaw = switch
        randomize_start_vel = switch
        randomize_start_pitch = switch

        randomize_start_dof_pos = False
        randomize_start_dof_vel = False

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
        joint_armature_range = {
            'default': dict(range=(0.005, 0.05), log_space=True),
            'ankle': dict(dof_ids=(4, 9), range=(0.001, 0.05), log_space=True)
        }
        randomize_coulomb_friction = switch

    class rewards:
        base_height_target = 0.6
        feet_height_target = 0.04
        feet_height_target_max = 0.05
        use_guidance_terrain = True
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_until_epoch = 100
        tracking_sigma = 5
        EMA_update_alpha = 0.99

        min_dist = 0.18
        max_dist = 0.50
        max_contact_force = 300

        rew_norm_factor = 1.0

        class scales:
            # gait
            joint_pos = (2.0, 0.3, 10, 200)
            feet_contact_number = (1.2, 0.6, 10, 200)
            feet_clearance = (1., 0.5, 10, 200)
            feet_distance = -1.
            knee_distance = -1.
            feet_rotation = -0.3

            # vel tracking
            tracking_lin_vel = 1.5
            tracking_goal_vel = 2.5
            tracking_ang_vel = 1.0

            # contact
            feet_slip = -0.1
            feet_contact_forces = -1e-3
            feet_stumble = -2.
            foothold = -0.1
            # feet_edge = -0.1

            # base pos
            default_dof_pos = -0.04
            default_dof_pos_yr = (0., -1., 10, 100)
            orientation = -10.0
            # base_height = -10.
            base_acc = -1.
            lin_vel_z = -1.
            ang_vel_xy = (0., -0.05, 10, 100)

            # energy
            action_smoothness = (0., -1e-3, 10, 100)
            # dof_vel_smoothness = -1e-3
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1.e-7
            collision = -1.
            dof_pos_limits = -10.
            dof_torque_limits = -0.01

    class policy:
        actor_gru_hidden_size = 128
        actor_hidden_dims = [512, 256, 128]

        critic_hidden_dims = [512, 256, 128]

    class odometer:
        odometer_type = 'recurrent'  # recurrent, auto-regression
        # odometer_type = 'auto-regression'  # recurrent, auto-regression

        # odometer parameters
        odom_transformer_embed_dim = 64
        odom_gru_hidden_size = 128
        estimator_output_dim = 4
        update_since = 100000000
        batch_size = 258
        learning_rate = 1e-3

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 8
        num_mini_batches = 5  # mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 2.e-4  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

        use_amp = True
        continue_from_last_std = True
        init_noise_std = 1.0

    class runner(PddBaseCfg.runner):
        runner_name = 'rl_odom'
        algorithm_name = 'ppo_odom'

        initial_smpl = 1.0
        lock_smpl_until = 1e10

        load_latest_interval = -1
        odometer_path = ''

        max_iterations = 50000  # number of policy updates
