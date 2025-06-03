import numpy as np

from .go1_base_config import Go1BaseCfg


class Go1_WMP_Cfg(Go1BaseCfg):
    class env(Go1BaseCfg.env):
        num_envs = 4096  # 6144

        n_proprio = 45  # 3 + 3 + 3 + 3 * 12
        len_prop_his = 5

        len_depth_his = 1
        scan_shape = (32, 16)
        n_scan = scan_shape[0] * scan_shape[1]

        num_critic_obs = 91
        len_critic_his = 50

        num_actions = 12
        episode_length_s = 30  # episode length in seconds

    class sensors:
        activated = True

        class depth_0:
            link_attached_to = 'base'
            position = [0.27, 0, 0.086]  # front camera
            position_range = [(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)]  # front camera
            pitch = 28  # positive pitch down (degree)
            pitch_range = [-1, 1]  # positive pitch down (degree)

            data_format = 'depth'  # depth, cloud, hmap
            update_interval = 5  # 5 works without retraining, 8 worse
            delay_prop = (5, 1)  # Gaussian (mean, std)

            resolution = (64, 64)  # width, height
            resized = (64, 64)
            horizontal_fov = 58

            near_clip = 0
            far_clip = 2
            dis_noise_global = 0.0  # in meters
            dis_noise_gaussian = 0.0  # in meters

    class terrain(Go1BaseCfg.terrain):
        base_pts_x = np.linspace(-0.2, 0.6, 16)
        base_pts_y = np.linspace(-0.2, 0.2, 8)

        scan_pts_x = np.linspace(-0.5, 1.1, 32)
        scan_pts_y = np.linspace(-0.4, 0.4, 16)
        body_pts_x = np.linspace(-0.2, 0.2, 4)
        body_pts_y = np.linspace(-0.2, 0.2, 4)
        feet_pts_x = np.linspace(-0.1, 0.1, 2)
        feet_pts_y = np.linspace(-0.1, 0.1, 2)

        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        num_cols = 20  # number of terrain cols (types)

        terrain_dict = {
            'smooth_slope': 3,
            'rough_slope': 1,
            'stairs_up': 1,
            'stairs_down': 1,
            'discrete': 0,
            'stepping_stone': 0,
            'gap': 0,
            'pit': 0,
            'parkour': 0,
            'parkour_gap': 1,
            'parkour_box': 1,
            'parkour_step': 1,
            'parkour_stair': 1,
            'parkour_flat': 0,
        }

    class noise(Go1BaseCfg.noise):
        add_noise = True

    class domain_rand(Go1BaseCfg.domain_rand):
        randomize_start_pos = True
        randomize_start_z = True
        randomize_start_yaw = True
        randomize_start_vel = True
        randomize_start_pitch = True

        randomize_start_dof_pos = True
        randomize_start_dof_vel = True

        randomize_friction = True
        randomize_base_mass = True
        randomize_link_mass = True
        randomize_com = True

        push_robots = False
        action_delay = True
        action_delay_range = [(0, 2)]
        add_dof_lag = False
        add_imu_lag = False

        randomize_torque = True
        randomize_gains = True
        randomize_motor_offset = True
        randomize_joint_stiffness = False  # for joints with spring behavior, (not implemented yet)
        randomize_joint_damping = False
        randomize_joint_friction = False
        randomize_joint_armature = True
        randomize_coulomb_friction = False

    class rewards:
        base_height_target = 0.3
        feet_height_target = 0.05
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_until_epoch = 100  # after the epoch, turn off only_positive_reward
        use_guidance_terrain = True
        tracking_sigma = 5
        soft_dof_pos_limit = 0.9

        EMA_update_alpha = 0.99
        rew_norm_factor = 1.0

        class scales:
            tracking_lin_vel = 1.5
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5

            base_stumble = -1.0
            feet_clearance = -0.01
            feet_stumble = -5.0  # -1.0
            feet_edge = -0.3  # -0.3

            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = 0.5  # -0.2
            torques = -1e-5
            action_smoothness = -3e-3  # -0.01
            dof_acc = -2.5e-7
            # dof_error = -0.04
            default_joint_pos = 0.1
            hip_pos = -0.5
            collision = -1.0
            dof_pos_limits = -10.0
            base_height = -1.0

    class world_model:
        step_interval = 5

        n_stoch = 32
        n_discrete = 32
        n_deter = 512
        n_mlp_enc = 1024
        n_cnn_enc = 4096
        hidden_size = 512
        state_initial = 'learned'

        unimix_ratio = 0.01

    class policy:
        # actor parameters
        actor_hidden_dims = [512, 256, 128]  # [128, 64, 32]
        init_noise_std = 1.0

        # critic parameters
        critic_hidden_dims = [512, 256, 128]

        use_recurrent_policy = True

        obs_gru_hidden_size = 64
        recon_gru_hidden_size = 256

    class algorithm:
        # PPO parameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2.e-4  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

        use_amp = True
        continue_from_last_std = True

    class runner(Go1BaseCfg.runner):
        runner_name = 'rl_dreamer'
        algorithm_name = 'ppo_wmp'

        max_iterations = 100000  # number of policy updates
