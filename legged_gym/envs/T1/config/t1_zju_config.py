import numpy as np

from .t1_base_config import T1BaseCfg, T1BaseCfgPPO


class T1ZJUCfg(T1BaseCfg):
    class env(T1BaseCfg.env):
        num_envs = 2048  # 6144

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
            link_attached_to = 'Waist'
            position = [0.1, 0, 0.]  # front camera
            position_range = [(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)]  # front camera
            pitch = 30  # positive is looking down
            pitch_range = [-1, 1]

            data_format = 'depth'  # depth, cloud
            update_interval = 5  # 5 works without retraining, 8 worse
            delay_prop = (5, 1)  # Gaussian (mean, std)

            resolution = (106, 60)  # width, height
            resized = (87, 58)  # (87, 58)
            horizontal_fov = 87

            near_clip = 0
            far_clip = 2
            dis_noise_global = 0.01  # in meters
            dis_noise_gaussian = 0.01  # in meters

    class commands:
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 8.  # time before command are changed[s]

        lin_vel_clip = 0.1
        ang_vel_clip = 0.2
        parkour_vel_tolerance = 0.3

        sw_switch = True

        class flat_ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-1., 1.]

        class stair_ranges:
            lin_vel_x = [0.6, 1.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1., 1.]  # this value limits the max yaw velocity computed by goal
            heading = [-1.5, 1.5]

        class parkour_ranges:
            lin_vel_x = [0.8, 1.2]  # min value should be greater than lin_vel_clip
            ang_vel_yaw = [-1.0, 1.0]  # this value limits the max yaw velocity computed by goal

    class terrain(T1BaseCfg.terrain):
        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        num_cols = 20  # number of terrain cols (types)

        terrain_dict = {
            'smooth_slope': 1,
            'rough_slope': 2,
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
            'parkour_stair': 6,  # First train a policy without stair for 2000 epochs
            'parkour_flat': 0,
        }

    class noise(T1BaseCfg.noise):
        add_noise = True

    class domain_rand(T1BaseCfg.domain_rand):
        switch = True

        randomize_start_pos = False
        randomize_start_y = switch
        randomize_start_yaw = switch
        randomize_start_vel = switch
        randomize_start_pitch = switch

        randomize_start_dof_pos = True
        randomize_start_dof_vel = True

        randomize_friction = False
        randomize_base_mass = switch
        randomize_link_mass = switch
        randomize_com = switch

        push_robots = False
        action_delay = True
        add_dof_lag = False
        add_imu_lag = False

        randomize_torque = switch
        randomize_gains = switch
        randomize_motor_offset = switch
        randomize_joint_stiffness = False  # for joints with spring behavior, (not implemented yet)
        randomize_joint_damping = False
        randomize_joint_friction = False
        randomize_joint_armature = True
        randomize_coulomb_friction = False

    class rewards:
        base_height_target = 0.64
        feet_height_target = 0.04
        feet_height_target_max = 0.06
        use_guidance_terrain = True
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_until_epoch = 100  # after the epoch, turn off only_positive_reward
        tracking_sigma = 5
        soft_dof_pos_limit = 0.9
        EMA_update_alpha = 0.99

        cycle_time = 0.7  # 0.64
        target_joint_pos_scale = 0.3  # 0.19

        min_dist = 0.2
        max_dist = 0.50
        max_contact_force = 300

        rew_norm_factor = 1.0

        class scales:
            # gait
            joint_pos = 2.
            feet_contact_number = 1.2
            feet_clearance = 0.2  # 0.2
            feet_air_time = 1.
            feet_slip = -1.
            feet_distance = 0.2
            knee_distance = 0.2
            feet_rotation = 0.5

            # contact
            feet_contact_forces = -0.01
            feet_stumble = -1.0
            feet_edge = -3.0

            # vel tracking
            tracking_lin_vel = 2.0
            tracking_goal_vel = 3.0
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5

            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2

            # energy
            action_smoothness = -3e-3
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.


class T1ZJUCfgPPO(T1BaseCfgPPO):
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
        enable_reconstructor = False

        obs_gru_hidden_size = 64
        recon_gru_hidden_size = 256

        len_latent = 16
        len_base_vel = 3
        len_latent_feet = 8
        len_latent_body = 16
        len_base_height = 0
        transformer_embed_dim = 64

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
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
        continue_from_last_std = True

    class runner(T1BaseCfgPPO.runner):
        max_iterations = 50000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
