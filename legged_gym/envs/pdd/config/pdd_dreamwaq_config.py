import numpy as np

from .pdd_base_config import PddBaseCfg, PddBaseCfgPPO


class PddDreamWaqCfg(PddBaseCfg):
    class env(PddBaseCfg.env):
        num_envs = 4096  # 6144
        n_proprio = 41
        len_prop_his = 50

        scan_shape = (32, 16)
        n_scan = scan_shape[0] * scan_shape[1]

        num_critic_obs = 53
        len_critic_his = 50

        num_actions = 10
        episode_length_s = 30  # episode length in seconds

    class terrain(PddBaseCfg.terrain):
        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        # num_cols = 20  # number of terrain cols (types)
        num_cols = 1  # number of terrain cols (types)

        scan_pts_x = np.linspace(-0.5, 1.1, 32)
        scan_pts_y = np.linspace(-0.4, 0.4, 16)

        curriculum = False

        terrain_dict = {
            'smooth_slope': 0,
            'rough_slope': 0,
            'stairs_up': 1,
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

    class noise(PddBaseCfg.noise):
        add_noise = False

    class domain_rand(PddBaseCfg.domain_rand):
        randomize_base_mass = True
        randomize_link_mass = True
        randomize_com = True
        randomize_torque = True
        randomize_motor_offset = True
        randomize_gains = True

        randomize_start_pos = True
        randomize_start_y = True
        randomize_start_yaw = True
        randomize_start_vel = True
        randomize_start_pitch = True

        randomize_start_dof_pos = True
        randomize_start_dof_vel = True

    class rewards:
        base_height_target = 0.6
        feet_height_target = 0.04
        feet_height_target_max = 0.07
        use_guidance_terrain = True
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 5
        soft_dof_pos_limit = 0.9
        EMA_update_alpha = 0.99

        cycle_time = 0.64  # 0.64
        target_joint_pos_scale = 0.19  # 0.19

        min_dist = 0.18
        max_dist = 0.50
        max_contact_force = 300

        rew_norm_factor = 1.0

        class scales:
            joint_pos = 2.0
            feet_clearance = 0.2
            feet_contact_number = 1.6
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # feet_rotation = 0.3
            # contact
            feet_contact_forces = -0.1
            # vel tracking
            tracking_lin_vel = 1.6
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
            torques = -2e-7
            dof_vel = -2e-4
            dof_acc = -5e-8
            collision = -1.
            stand_still = 2.0
            # feet_rotation = 0.3


class PddDreamWaqCfgPPO(PddBaseCfgPPO):
    seed = -1
    runner_name = 'rl_dream'
    algorithm_name = 'ppo_dreamwaq'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

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

    class runner(PddBaseCfgPPO.runner):
        max_iterations = 3000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations


class PddDreamWaqDRCfg(PddDreamWaqCfg):
    class terrain(PddDreamWaqCfg.terrain):
        curriculum = True

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
            'parkour_stair': 0,
            'parkour_flat': 0,
        }

    class noise(PddDreamWaqCfg.noise):
        add_noise = True

    class domain_rand(PddDreamWaqCfg.domain_rand):
        switch = True

        push_robots = switch
        action_delay = True
        add_dof_lag = False
        add_imu_lag = False

        randomize_base_mass = switch
        randomize_link_mass = switch
        randomize_com = switch
        randomize_friction = switch
        randomize_coulomb_friction = False
        randomize_torque = switch
        randomize_motor_offset = switch
        randomize_gains = switch


class PddDreamWaqDRCfgPPO(PddDreamWaqCfgPPO):
    class runner(PddDreamWaqCfgPPO.runner):
        max_iterations = 10000  # number of policy updates
