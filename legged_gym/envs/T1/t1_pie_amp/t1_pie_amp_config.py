import numpy as np

from legged_gym.envs.T1.config.t1_base_config import T1BaseCfg


class Obs_scales:
    dof_pos = 1.0
    dof_vel = 1.0
    lin_vel = 1.0
    ang_vel = 1.0
    quat = 1.0


class T1_PIE_AMP_Cfg(T1BaseCfg):
    class env(T1BaseCfg.env):
        num_envs = 4096  # 6144

        n_proprio = 48
        len_prop_his = 10

        scan_shape = (32, 16)
        n_scan = scan_shape[0] * scan_shape[1]

        num_critic_obs = 67
        len_critic_his = 50

        num_actions = 13
        episode_length_s = 40  # episode length in seconds

    class sensors:
        activated = True

        class depth_0:
            link_attached_to = 'H2'
            position = [0.07, 0, 0.09]  # front camera
            pitch = 0  # positive is looking down
            # link_attached_to = 'Trunk'
            # position = [0.17, 0, 0.0]  # front camera
            # pitch = 60  # positive is looking down
            yaw = 0

            position_range = [(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)]  # front camera
            pitch_range = [-3, 3]

            data_format = 'depth'  # depth, cloud, hmap
            update_interval = 5
            delay_prop = (5, 1)  # Gaussian (mean, std), or None
            history_length = 2

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

        # class depth_1:
        #     link_attached_to = 'Trunk'
        #     position = [-0.06, 0, 0.17]  # front camera
        #     pitch = 60  # positive is looking down
        #     yaw = 180
        #
        #     position_range = [(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)]  # front camera
        #     pitch_range = [-3, 3]
        #
        #     data_format = 'depth'  # depth, cloud, hmap
        #     update_interval = 5
        #     delay_prop = (5, 1)  # Gaussian (mean, std), or None
        #     history_length = 2
        #
        #     resolution = (114, 64)  # width, height
        #     crop = (0, 2, 4, 4)  # top, bottom, left, right
        #
        #     edge_process = True
        #     edge_noise = dict(blank_ratio=0.2, repeat_ratio=0.2)
        #     blank_ratio = 0.002
        #
        #     near_clip = 0
        #     far_clip = 2
        #     dis_noise_global = 0.01  # in meters
        #     dis_noise_gaussian = 0.01  # in meters
        #     noise_scale_perlin = 1  # 0-1
        #
        #     resized = (64, 64)
        #     horizontal_fov = 87
        #
        #     bounding_box = (0.3, 1.1, -0.4, 0.4)  # x1, x2, y1, y2
        #     hmap_shape = (16, 16)  # x dim, y dim

    class commands:
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 4.  # time before command are changed[s]

        lin_vel_clip = 0.2
        ang_vel_clip = 0.2
        parkour_vel_tolerance = 0.3

        cycle_time = 1.  # 0.64
        target_joint_pos_scale = 0.5

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
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [0.3, 0.8]
            ang_vel_yaw = [-1., 1.]  # this value limits the max yaw velocity computed by goal
            heading = [-1.5, 1.5]

        class parkour_ranges:
            lin_vel_x = [0.3, 1.2]  # min value should be greater than lin_vel_clip
            ang_vel_yaw = [-1.0, 1.0]  # this value limits the max yaw velocity computed by goal

    class terrain(T1BaseCfg.terrain):
        body_pts_x = np.linspace(-0.6, 1.2, 32)
        body_pts_y = np.linspace(-0.6, 0.6, 16)

        num_rows = 10  # number of terrain rows (levels)   spreaded is beneficial !
        num_cols = 20  # number of terrain cols (types)

        terrain_dict = {
            'smooth_slope': 1,
            'rough_slope': 1,
            'stairs_up': 1,
            'stairs_down': 1,
            'parkour_stair': 1,
            'parkour_stair_down': 1,
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
        action_delay_range = [(0, 5), (0, 10), (0, 15)]
        add_dof_lag = True
        dof_lag_range = (0, 10)
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
            'ankle': dict(dof_ids=(15, 16, 21, 22), range=(0.0001, 0.05), log_space=True)
        }

        randomize_coulomb_friction = True

    class rewards:
        base_height_target = 0.64
        feet_height_target = 0.04
        feet_height_target_max = 0.06
        use_guidance_terrain = True
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_until_epoch = 200  # after the epoch, turn off only_positive_reward
        tracking_sigma = 5
        EMA_update_alpha = 0.99

        min_dist = 0.25
        max_dist = 0.50
        max_contact_force = 300

        rew_norm_factor = 1.0

        class scales:  # float or (start, end, span, start_it)
            # vel tracking
            tracking_lin_vel = 2.5
            tracking_goal_vel = 3.0
            tracking_ang_vel = 1.5

            # contact
            feet_slip = -0.1
            feet_contact_forces = -1e-3
            feet_stumble = -1.
            foothold = -0.1
            feet_clearance = 0.1

            # base pos
            default_joint_pos = -0.04
            orientation = -10.

            # energy
            action_smoothness = -1e-3
            dof_vel_smoothness = -1e-3
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

        estimator_gru_hidden_size = 256
        len_latent_z = 32
        len_latent_hmap = 32

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 4
        num_mini_batches = 5  # mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 1.e-4  # 5.e-4
        amp_lr = 5e-5
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

        continue_from_last_std = True
        init_noise_std = 1.0
        noise_range = (0.3, 1.0)

        use_amp = True

    class runner(T1BaseCfg.runner):
        runner_name = 'rl_amp'
        algorithm_name = 'ppo_pie_amp'
        # algorithm_name = 'ppo_pie_amp_plain'

        lock_smpl_to = 1

        max_iterations = 200000  # number of policy updates

    class amp:
        # 数据加载相关
        motion_file = "data/T1_walk"
        preload = True
        num_preload_data = 400000

        # 根据amp_obs_dict加在参考数据以及生成数据
        amp_obs_dict = {
            "dof_pos":
                {"using": True,
                 "size": 13,
                 "obs_scale": [Obs_scales.dof_pos],
                 },
            "dof_vel":
                {"using": True,
                 "size": 13,
                 "obs_scale": [Obs_scales.dof_vel],
                 },
            "base_lin_vel":
                {"using": True,
                 "size": 3,
                 "obs_scale": [Obs_scales.lin_vel],
                 },
            "base_ang_vel":
                {"using": True,
                 "size": 3,
                 "obs_scale": [Obs_scales.ang_vel],
                 },
            "base_height":
                {"using": True,
                 "size": 1,
                 "obs_scale": [1.0],
                 },
            "projected_gravity":
                {"using": True,
                 "size": 3,
                 "obs_scale": [1.0],
                 "interpolate": "slerp",
                 },
            "torso_projected_gravity":
                {"using": False,
                 "size": 3,
                 "obs_scale": [1.0],
                 "interpolate": "slerp"
                 },
            "shoulder_pos_to_base":
                {"using": False,
                 "size": 3 * 2,
                 "obs_scale": [1.0],
                 },
            "knee_pos_to_base":
                {"using": True,
                 "size": 3 * 2,
                 "obs_scale": [1.0],
                 },
            "feet_pos_to_base":
                {"using": True,
                 "size": 3 * 2,
                 "obs_scale": [1.0],
                 },
            "elbow_pos_to_base":
                {"using": False,
                 "size": 3 * 2,
                 "obs_scale": [1.0],
                 },
            "wrist_pos_to_base":
                {"using": False,
                 "size": 3 * 2,
                 "obs_scale": [1.0],
                 },
        }
        num_single_amp_obs = 0
        amp_obs_hist_steps = 6
        for key, value in amp_obs_dict.items():
            if value["using"]:
                num_single_amp_obs += value["size"]
        num_amp_obs = int(amp_obs_hist_steps * num_single_amp_obs)

        # 构建判别器相关
        amp_disc_cfg = {"num_input": num_amp_obs,
                        "hidden_dims": [1024, 512, 256],
                        "activation": 'relu',
                        "amp_reward_coef": 6.0,
                        "amp_type": 'least_square',  # 'least_square' , 'wasserstein', 'log', 'bce'
                        "lambda_schedule_dict":
                            {
                                "schedule_type": "inverse",  # linear, inverse, exp, None
                                "lambda1": [20, 50, 500, 0.05],  # init,low,high,ema
                            },
                        "task_rew_schedule_dict":
                            {
                                "using_schedule": True,
                                "buffer_size": 10000,
                                "update_step": 0.05,
                                "task_rew_coef_min": 0.7,
                                "update_threshold": 0.8,
                            },
                        }

        # 更新判别器相关
        amp_optim_cfg = {"amp_trunk_weight_decay": 10e-4,
                         "amp_head_weight_decay": 10e-2,
                         "amp_replay_buffer_size": 500000,
                         "amp_loss_coef": 5.0,
                         'amp_disc_lr': 5e-5,
                         'max_amp_disc_grad_norm': 0.05,
                         'amp_update_interval': 1,
                         }

        # 数据归一化相关
        amp_empirical_normalization = True
        amp_normal_update_until = 1e4

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
