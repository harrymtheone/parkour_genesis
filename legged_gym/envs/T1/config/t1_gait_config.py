import numpy as np

from .t1_base_config import T1BaseCfg, T1BaseCfgPPO


class T1GaitCfg(T1BaseCfg):
    class env(T1BaseCfg.env):
        num_envs = 4096
        n_proprio = 50
        len_prop_his = 50

        scan_shape = (32, 16)
        n_scan = scan_shape[0] * scan_shape[1]

        num_critic_obs = 62
        len_critic_his = 50

        num_actions = 13
        episode_length_s = 30  # episode length in seconds

    class terrain(T1BaseCfg.terrain):
        scan_pts_x = np.linspace(-0.5, 1.1, 32)
        scan_pts_y = np.linspace(-0.4, 0.4, 16)
        body_pts_x = np.linspace(-0.2, 0.2, 4)
        body_pts_y = np.linspace(-0.2, 0.2, 4)
        feet_pts_x = np.linspace(-0.1, 0.1, 2)
        feet_pts_y = np.linspace(-0.1, 0.1, 2)

        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        num_cols = 20  # number of terrain cols (types)

        curriculum = False

        terrain_dict = {
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
            'parkour_flat': 1,
        }

    class noise(T1BaseCfg.noise):
        add_noise = False

    class domain_rand(T1BaseCfg.domain_rand):
        switch = False

        randomize_start_pos = False
        randomize_start_y = switch
        randomize_start_yaw = switch
        randomize_start_vel = switch
        randomize_start_pitch = switch

        randomize_start_dof_pos = False
        randomize_start_dof_vel = False

        randomize_friction = switch
        randomize_base_mass = switch
        randomize_link_mass = switch
        randomize_com = switch

        push_robots = False
        action_delay = False
        add_dof_lag = False
        add_imu_lag = False

        randomize_torque = switch
        randomize_gains = switch
        randomize_motor_offset = switch
        randomize_joint_stiffness = False  # for joints with spring behavior, (not implemented yet)
        randomize_joint_damping = False
        randomize_joint_friction = switch
        randomize_joint_armature = True
        randomize_coulomb_friction = False

    class rewards:
        base_height_target = 0.65
        feet_height_target = 0.05
        use_guidance_terrain = True
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 5
        soft_dof_pos_limit = 0.9
        EMA_update_alpha = 0.99

        cycle_time = 0.7  # 0.64

        min_dist = 0.2
        max_dist = 0.6
        max_contact_force = 300

        rew_norm_factor = 1.0

        class scales:
            # tracking_lin_vel = 1.5
            # tracking_ang_vel = 0.5
            # orientation = -2.0
            # energy = -2.5e-7
            #
            # dof_vel = -1e-4
            # dof_acc = -2e-6
            # dof_pos_limits = -10
            # torques = -1e-7
            # contact_forces = -3e-4
            # collision = -10.
            #
            # action_rate = -6e-3
            # # arm_dof_err = -0.3
            # waist_dof_err = -0.1
            # leg_yaw_roll = -0.1
            # feet_away = 0.4
            #
            # base_height = -1.0
            # feet_distance = 0.2

            # tracking rewards
            tracking_lin_vel = 1.5
            tracking_goal_vel = 2.0
            tracking_yaw = 0.5
            orientation = -0.1
            dof_acc = -2.5e-7

            lin_vel_z = -1.0 / 50
            ang_vel_xy = -0.05 / 50
            collision = -10.0 / 50
            action_rate = -0.1 / 50
            delta_torques = -1.0e-7 / 50
            torques = -0.00001 / 50
            dof_error = -0.15

            # feet_stumble = -1.0 / 50
            # feet_edge = -1.0 / 50

            feet_distance = 0.2
            contact_forces = -2e-3
            # dof_pos_limits = -10
            # dof_torque_limits = -0.1


class T1GaitCfgPPO(T1BaseCfgPPO):
    seed = -1
    runner_name = 'rl_dream'  # rl, distil, mixed
    algorithm_name = 'ppo_gait'

    class policy:
        # actor parameters
        actor_hidden_dims = [512, 256, 128]  # [128, 64, 32]

        # critic parameters
        critic_hidden_dims = [512, 256, 128]

        use_recurrent_policy = True
        estimator_gru_hidden_size = 256

        len_base_vel = 3
        len_latent_feet = 8
        len_latent_body = 16
        len_hmap_latent = 128

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 2.e-4  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

        use_amp = True
        init_noise_std = 1.0
        continue_from_last_std = False

    class runner(T1BaseCfgPPO.runner):
        max_iterations = 50000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
