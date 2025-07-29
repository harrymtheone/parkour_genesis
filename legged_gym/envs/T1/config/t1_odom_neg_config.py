import numpy as np

from .t1_base_config import T1BaseCfg


class T1_Odom_Neg_Cfg(T1BaseCfg):
    class env(T1BaseCfg.env):
        num_envs = 4096  # 6144

        n_proprio = 50
        len_prop_his = 10

        scan_shape = (32, 16)
        n_scan = scan_shape[0] * scan_shape[1]

        priv_actor_len = 3

        num_critic_obs = 70
        len_critic_his = 50

        num_actions = 13
        episode_length_s = 40  # episode length in seconds

    class sensors:
        activated = False

        class depth_0:
            # link_attached_to = 'H2'
            # position = [0.07, 0, 0.09]  # front camera
            # pitch = 0  # positive is looking down
            link_attached_to = 'Trunk'
            position = [0.17, 0, 0.0]  # front camera
            pitch = 60  # positive is looking down

            position_range = [(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)]  # front camera
            pitch_range = [-3, 3]

            data_format = 'cloud'  # depth, cloud, hmap
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
        delta_t = 0.02

        class flat_ranges:
            lin_vel_x = [-0.8, 1.2]
            lin_vel_y = [-0.8, 0.8]
            ang_vel_yaw = [-1., 1.]

        class stair_ranges:
            lin_vel_x = [-0.5, 0.8]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1., 1.]  # this value limits the max yaw velocity computed by goal
            heading = [-1.5, 1.5]

        class parkour_ranges:
            lin_vel_x = [0.3, 0.8]  # min value should be greater than lin_vel_clip
            ang_vel_yaw = [-1.0, 1.0]  # this value limits the max yaw velocity computed by goal

    class terrain(T1BaseCfg.terrain):
        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        num_cols = 20  # number of terrain cols (types)

        terrain_dict = {
            'smooth_slope': 1,
            'rough_slope': 1,
            'stairs_up': 2,
            'stairs_down': 2,
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
            # 'ankle': dict(dof_ids=(15, 16, 21, 22), range=(0.01, 0.05), log_space=False)
        }

        randomize_coulomb_friction = True

    class rewards:
        base_height_target = 0.64
        feet_height_target = 0.04
        feet_height_target_max = 0.06
        use_guidance_terrain = True
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_until_epoch = 500  # after the epoch, turn off only_positive_reward
        tracking_sigma = 5
        EMA_update_alpha = 0.99

        min_dist = 0.25
        max_dist = 0.50
        max_contact_force = 300

        rew_norm_factor = 1.0

        class scales:  # float or (start, end, span, start_it)
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
            action_smoothness = -1e-3
            # dof_vel_smoothness = -1e-3
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1.e-7
            collision = -1.
            dof_pos_limits = -10.
            dof_torque_limits = -0.01

    class policy:
        # actor parameters
        actor_gru_hidden_size = 128
        actor_hidden_dims = [512, 256, 128]  # [128, 64, 32]

        # critic parameters
        critic_hidden_dims = [512, 256, 128]

    class odometer:
        odometer_type = 'priv_recon'  # recurrent, auto-regression, priv_recon

        # odometer parameters
        odom_transformer_embed_dim = 64
        odom_gru_hidden_size = 128
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

        continue_from_last_std = True
        init_noise_std = 0.6

        use_amp = True

    class runner(T1BaseCfg.runner):
        runner_name = 'rl_odom'
        algorithm_name = 'ppo_odom'

        initial_smpl = 1.0
        lock_smpl_until = 1e10

        load_latest_interval = -1
        odometer_path = ''

        max_iterations = 20000  # number of policy updates


# -----------------------------------------------------------------------------------------------
# ------------------------------------------- Stair -------------------------------------------
# -----------------------------------------------------------------------------------------------

class T1_Odom_Stair_Neg_Cfg(T1_Odom_Neg_Cfg):
    class domain_rand(T1_Odom_Neg_Cfg.domain_rand):
        push_robots = True
        push_duration = [0.3]

        action_delay = True
        action_delay_range = [(0, 4), (0, 6)]
        action_delay_update_steps = 5000 * 24

        add_dof_lag = True
        dof_lag_range = (0, 6)

        randomize_joint_armature = True
        joint_armature_range = {
            'default': dict(range=(0.01, 0.05), log_space=False),
            'ankle': dict(dof_ids=(15, 16, 21, 22), range=(0.0001, 0.05), log_space=True)
        }

    class terrain(T1_Odom_Neg_Cfg.terrain):
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)

        terrain_dict = {
            'smooth_slope': 1,
            'rough_slope': 1,
            'stairs_up': 1,
            'stairs_down': 1,
            'parkour_stair': 1,
            'parkour_stair_down': 1,
            'parkour_mini_stair': 1,
            'parkour_mini_stair_down': 1,
        }

    class rewards(T1_Odom_Neg_Cfg.rewards):
        only_positive_rewards = True
        only_positive_rewards_until_epoch = 22000 + 200

        class scales(T1_Odom_Neg_Cfg.rewards.scales):  # start, end, span, start_it
            joint_pos = 0.3
            feet_contact_number = 0.3
            feet_clearance = 1.
            feet_distance = -1.
            knee_distance = -1.
            feet_rotation = -0.3

            # vel tracking
            tracking_lin_vel = 1.5
            tracking_goal_vel = 2.5
            tracking_ang_vel = 2.0

            # contact
            feet_slip = -0.1
            feet_contact_forces = -1e-3
            feet_stumble = -2.
            foothold = -0.1
            feet_edge = -0.1

            # base pos
            default_dof_pos = -0.04
            default_dof_pos_yr = -1.
            orientation = -10.
            # base_height = -10.
            base_acc = -1.
            lin_vel_z = -1.
            ang_vel_xy = -0.05

            # energy
            action_smoothness = -1e-3
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1.e-7
            collision = -1.

            # dof_vel_smoothness = -1e-4
            dof_pos_limits = -10.
            # dof_vel_limits = -0.5
            dof_torque_limits = -0.1

    class control(T1BaseCfg.control):
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

    class algorithm(T1_Odom_Neg_Cfg.algorithm):
        continue_from_last_std = False
        init_noise_std = 0.6

    class runner(T1_Odom_Neg_Cfg.runner):
        max_iterations = 100000  # number of policy updates


# -----------------------------------------------------------------------------------------------
# ------------------------------------------- Finetune -------------------------------------------
# -----------------------------------------------------------------------------------------------


class T1_Odom_Neg_Finetune_Cfg(T1_Odom_Stair_Neg_Cfg):
    class sensors(T1_Odom_Stair_Neg_Cfg.sensors):
        activated = True

    class domain_rand(T1_Odom_Stair_Neg_Cfg.domain_rand):
        action_delay = True
        action_delay_range = [(0, 6)]

    class algorithm(T1_Odom_Neg_Cfg.algorithm):
        continue_from_last_std = True

    class runner(T1_Odom_Stair_Neg_Cfg.runner):
        max_iterations = 100000

        load_latest_interval = 100
        odometer_path = ''

        initial_smpl = 1.0
        lock_smpl_until = 0

        # runner_name = 'rl_dream'
        # algorithm_name = 'ppo_odom_roa'
