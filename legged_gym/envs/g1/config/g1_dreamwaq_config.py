import numpy as np

from .g1_base_config import G1BaseCfg


class G1DreamWaqCfg(G1BaseCfg):
    class env(G1BaseCfg.env):
        num_envs = 4096  # 6144

        n_proprio = 50
        len_prop_his = 50

        scan_shape = (32, 16)
        n_scan = scan_shape[0] * scan_shape[1]

        num_critic_obs = 62
        len_critic_his = 50

        num_actions = 13
        episode_length_s = 30  # episode length in seconds

    class terrain(G1BaseCfg.terrain):
        description_type = 'plane'

    class noise(G1BaseCfg.noise):
        add_noise = True

    class domain_rand(G1BaseCfg.domain_rand):
        switch = True

        randomize_start_pos = switch
        randomize_start_z = False
        randomize_start_yaw = switch
        randomize_start_vel = False
        randomize_start_pitch = switch

        randomize_start_dof_pos = False
        randomize_start_dof_vel = False

        randomize_friction = switch
        randomize_base_mass = switch
        randomize_link_mass = switch
        randomize_com = switch

        push_robots = switch
        action_delay = switch
        add_dof_lag = switch
        add_imu_lag = False

        randomize_torque = switch
        randomize_gains = switch
        randomize_motor_offset = switch
        randomize_joint_stiffness = False  # for joints with spring behavior, not usually used
        randomize_joint_damping = False
        randomize_joint_friction = False
        randomize_joint_armature = switch
        randomize_coulomb_friction = switch

    class rewards:
        base_height_target = 0.7
        feet_height_target = 0.04
        feet_height_target_max = 0.06
        use_guidance_terrain = True
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_until_epoch = 1e10  # after the epoch, turn off only_positive_reward
        tracking_sigma = 5
        soft_dof_pos_limit = 0.9
        EMA_update_alpha = 0.99

        cycle_time = 0.7
        target_joint_pos_scale = 0.3  # 0.19

        min_dist = 0.18
        max_dist = 0.50
        max_contact_force = 300

        rew_norm_factor = 1.0

        class scales:
            # gait
            joint_pos = 2.
            feet_contact_number = 1.2
            feet_clearance = 0.2  # 0.2
            feet_distance = 0.2
            knee_distance = 0.2
            feet_rotation = 0.5

            # contact
            feet_slip = -1.
            feet_contact_forces = -0.001

            # vel tracking
            tracking_lin_vel = 1.6
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5

            # base pos
            default_joint_pos = 1.5
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

    class policy:
        use_recurrent_policy = True
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
        init_noise_std = 1.0

    class runner(G1BaseCfg.runner):
        runner_name = 'rl_dream'
        algorithm_name = 'ppo_dreamwaq'

        max_iterations = 1000  # number of policy updates


class T1DreamWaqPhase2Cfg(G1DreamWaqCfg):
    class terrain(G1DreamWaqCfg.terrain):
        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        num_cols = 20  # number of terrain cols (types)

        scan_pts_x = np.linspace(-0.5, 1.1, 32)
        scan_pts_y = np.linspace(-0.4, 0.4, 16)

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
            'parkour_mini_stair': 0,
            'parkour_flat': 0,
        }

    class domain_rand(G1DreamWaqCfg.domain_rand):
        push_robots = True
        push_duration = [0.2, 0.3]

        action_delay_range = [(0, 10), (5, 15), (5, 20)]
        action_delay_update_steps = 2000 * 24

    class control(G1DreamWaqCfg.control):
        # PD Drive parameters:
        stiffness = {
            'Head': 30,
            'Shoulder_Pitch': 300, 'Shoulder_Roll': 200, 'Elbow_Pitch': 200, 'Elbow_Yaw': 100,  # not used yet, set randomly
            'Waist': 100,
            'Hip_Pitch': 55, 'Hip_Roll': 55, 'Hip_Yaw': 30, 'Knee_Pitch': 100, 'Ankle_Pitch': 30, 'Ankle_Roll': 30,
        }

        damping = {
            'Head': 1,
            'Shoulder_Pitch': 3, 'Shoulder_Roll': 3, 'Elbow_Pitch': 3, 'Elbow_Yaw': 3,  # not used yet, set randomly
            'Waist': 3,
            'Hip_Pitch': 3, 'Hip_Roll': 3, 'Hip_Yaw': 4, 'Knee_Pitch': 5, 'Ankle_Pitch': 0.3, 'Ankle_Roll': 0.3,
        }

    class runner(G1DreamWaqCfg.runner):
        max_iterations = 20000  # number of policy updates
