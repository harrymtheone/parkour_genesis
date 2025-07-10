from .t1_base_config import T1BaseCfg


class T1_PIE_Cfg(T1BaseCfg):
    class env(T1BaseCfg.env):
        num_envs = 4096  # 6144

        n_proprio = 50
        len_prop_his = 10

        len_depth_his = 2
        scan_shape = (32, 16)
        n_scan = scan_shape[0] * scan_shape[1]

        num_critic_obs = 86
        len_critic_his = 50

        num_actions = 13
        episode_length_s = 40  # episode length in seconds

    class sensors:
        activated = True

        class depth_0:
            link_attached_to = 'H2'
            position = [0.07, 0, 0.09]  # front camera
            position_range = [(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)]  # front camera
            pitch = 0  # positive is looking down
            pitch_range = [-3, 3]

            data_format = 'depth'  # depth, cloud, hmap
            update_interval = 1
            delay_prop = (5, 1)  # Gaussian (mean, std)

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
        resampling_time = 4.  # time before command are changed[s]

        lin_vel_clip = 0.2
        ang_vel_clip = 0.2
        parkour_vel_tolerance = 0.3

        cycle_time = 0.7  # 0.64
        target_joint_pos_scale = 0.2

        sw_switch = True
        phase_offset_l = 0.
        phase_offset_r = 0.5
        air_ratio = 0.4
        delta_t = 0.1

        class flat_ranges:
            lin_vel_x = [-0.8, 1.2]
            lin_vel_y = [-0.8, 0.8]
            ang_vel_yaw = [-1., 1.]

        class stair_ranges:
            lin_vel_x = [-0.8, 1.2]
            lin_vel_y = [-0.8, 0.8]
            ang_vel_yaw = [-1., 1.]  # this value limits the max yaw velocity computed by goal
            heading = [-1.5, 1.5]

        class parkour_ranges:
            lin_vel_x = [0.3, 1.2]  # min value should be greater than lin_vel_clip
            ang_vel_yaw = [-1.0, 1.0]  # this value limits the max yaw velocity computed by goal

    class terrain(T1BaseCfg.terrain):
        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        num_cols = 20  # number of terrain cols (types)

        terrain_dict = {
            'smooth_slope': 2,
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
            'parkour_flat': 0,
        }

    class noise(T1BaseCfg.noise):
        add_noise = True

    class domain_rand(T1BaseCfg.domain_rand):
        randomize_start_pos = True
        randomize_start_z = False
        randomize_start_yaw = True
        randomize_start_vel = True
        randomize_start_pitch = True

        randomize_start_dof_pos = False
        randomize_start_dof_vel = False

        randomize_friction = True
        randomize_base_mass = True
        randomize_link_mass = True
        randomize_com = True

        push_robots = True
        push_duration = [0.1]
        action_delay = True
        action_delay_range = [(0, 4)]
        add_dof_lag = True
        dof_lag_range = (0, 6)
        add_imu_lag = False

        randomize_torque = True
        randomize_gains = True
        randomize_motor_offset = True
        randomize_joint_stiffness = False  # for joints with spring behavior, (not implemented yet)
        randomize_joint_damping = False
        randomize_joint_friction = False

        randomize_joint_armature = True
        joint_armature_range = {
            'default': dict(range=(0.01, 0.05), log_space=False),
            'ankle': dict(dof_ids=(15, 16, 21, 22), range=(0.01, 0.05), log_space=False)
        }

        randomize_coulomb_friction = True

    class rewards:
        base_height_target = 0.64
        feet_height_target = 0.05
        feet_height_target_max = 0.07
        use_guidance_terrain = True
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_until_epoch = 1000  # after the epoch, turn off only_positive_reward
        tracking_sigma = 5
        soft_dof_pos_limit = 0.9
        EMA_update_alpha = 0.99

        min_dist = 0.2
        max_dist = 0.5
        max_contact_force = 300

        rew_norm_factor = 1.0

        class scales:
            # gait
            joint_pos = 2.
            feet_contact_number = 1.2
            feet_clearance = 1.
            feet_distance = 0.2
            knee_distance = 0.2
            feet_rotation = 0.5

            # vel tracking
            tracking_lin_vel = 2.5
            tracking_goal_vel = 3.0
            tracking_ang_vel = 2.5

            # contact
            feet_slip = -1.
            feet_contact_forces = -1e-3
            feet_stumble = -1.
            foothold = -1.

            # base pos
            default_joint_pos = 1.0
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            vel_mismatch_exp = 0.5

            # energy
            action_smoothness = -3e-3
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

            dof_torque_limits = -0.01
            dof_pos_limits = -10.

    class policy:
        # actor parameters
        actor_hidden_dims = [512, 256, 128]  # [128, 64, 32]

        # critic parameters
        critic_hidden_dims = [512, 256, 128]

        use_recurrent_policy = True
        estimator_gru_hidden_size = 256

        len_estimation = 3
        len_hmap_latent = 128

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 4
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

    class runner(T1BaseCfg.runner):
        runner_name = 'rl_dream'  # rl, distil, mixed
        algorithm_name = 'ppo_pie'

        lock_smpl_until = 6000
        max_iterations = 6000  # number of policy updates


class T1_PIE_Stair_Cfg(T1_PIE_Cfg):
    class terrain(T1_PIE_Cfg.terrain):
        num_rows = 10  # number of terrain rows (levels), spread is beneficial!
        num_cols = 20  # number of terrain cols (types)

        terrain_dict = {
            'smooth_slope': 2,
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
            'parkour_stair': 1,
            'parkour_flat': 0,
        }

    class domain_rand(T1_PIE_Cfg.domain_rand):
        randomize_start_yaw = False

        push_robots = False

        action_delay = True
        action_delay_range = [(0, 4)]

    class rewards(T1_PIE_Cfg.rewards):
        only_positive_rewards = False

    class algorithm(T1_PIE_Cfg.algorithm):
        continue_from_last_std = False
        init_noise_std = 0.6

    class runner(T1_PIE_Cfg.runner):
        max_iterations = 50000  # number of policy updates
