import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_config import BaseConfig


class PddBaseCfg(BaseConfig):
    class play:
        control = False  # manually control or automatically
        use_joystick = False  # use joystick or keyboard
        plot_flag = False

    class env:
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        obs_type = "og"

        # additional visual inputs
        next_goal_threshold = 0.4
        reach_goal_delay = 0.05
        num_future_goal_obs = 2

    class depth:
        use_camera = False
        use_warp = True

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0

        scan_norm_bias = 0.7

        clip_observations = 100.
        clip_actions = 100.

    class terrain:
        # description_type = 'trimesh'  # plane, heightfield or trimesh
        description_type = 'plane'  # plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1  # for fast

        y_range = [-0.4, 0.4]

        edge_width_thresh = 0.05
        horizontal_scale = 0.25  # [m] influence computation time by a lot
        vertical_scale = 0.005  # [m]
        border_size = 5  # [m]
        roughness_height = [0.0, 0.0]
        simplify_grid = True
        downsampled_scale = 0.075

        curriculum = True
        max_difficulty = False

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True

        base_pts_x = np.linspace(-0.2, 0.2, 8)
        base_pts_y = np.linspace(-0.2, 0.2, 8)

        height_update_interval = 5  # 1 * dt

        max_init_terrain_level = 9  # starting curriculum state
        terrain_size = [8., 8.]
        terrain_parkour_size = [18., 4.]

        # trimesh only:
        slope_treshold = 1.5  # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False

        num_goals = 8

    class commands:
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 15.  # time before command are changed[s]

        lin_vel_clip = 0.1
        ang_vel_clip = 0.2

        sw_switch = True

        class flat_ranges:
            lin_vel_x = [-0.5, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1., 1.]

        class stair_ranges:
            lin_vel_x = [0.2, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1., 1.]  # this value limits the max yaw velocity computed by goal
            heading = [-1.5, 1.5]

        class parkour_ranges:
            lin_vel_x = [0.6, 1.0]  # min value should be greater than lin_vel_clip
            ang_vel_yaw = [-1.0, 1.0]  # this value limits the max yaw velocity computed by goal

    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.02
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class domain_rand:
        randomize_start_pos = False
        randomize_start_pos_range = 0.3
        randomize_start_vel = False
        randomize_start_vel_range = 0.1
        randomize_start_yaw = False
        randomize_start_yaw_range = 1.2
        randomize_start_y = False
        randomize_start_y_range = 0.1  # (0, 0.05)
        randomize_start_pitch = False
        randomize_start_pitch_range = 0.09  # 5 degree

        randomize_start_dof_pos = False
        randomize_start_dof_pos_range = 0.15
        randomize_start_dof_vel = False
        randomize_start_dof_vel_range = 0.15

        push_robots = False
        push_force_max = ((-300, 600),
                          (-400, 400),
                          (-5, 5))
        push_torque_max = (-0, 0)
        push_interval_s = 6
        push_duration = [0.05, 0.1, 0.15]
        push_duration_update_steps = 1000 * 24

        action_delay = False
        randomize_action_delay = True  # if False, max delay will be used
        randomize_action_delay_each_step = False
        action_delay_range = (1, 20)
        # action_delay_range = (10, 70)

        add_dof_lag = False
        randomize_dof_lag = True  # if False, max delay will be used
        randomize_dof_lag_each_step = False
        dof_lag_range = (0, 40)

        add_imu_lag = False
        randomize_imu_lag = True  # if False, max delay will be used
        randomize_imu_lag_each_step = False
        imu_lag_range = (1, 10)

        randomize_base_mass = False
        added_mass_range = [-2.5, 2.5]

        randomize_link_mass = False
        link_mass_multiplier_range = [0.9, 1.1]

        randomize_com = False
        com_displacement_range = [-0.05, 0.05]

        randomize_friction = False
        friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        randomize_coulomb_friction = False
        joint_coulomb_range = [0.1, 1.0]
        joint_viscous_range = [0.1, 0.9]

        randomize_torque = False
        torque_multiplier_range = [0.8, 1.2]

        randomize_motor_offset = False
        motor_offset_range = [-0.035, 0.035]  # Offset to add to the motor angles

        randomize_gains = False
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]  # Factor

    class init_state:
        pos = [0.0, 0.0, 0.7]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        init_angle = 0.36
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'leg_l1_joint': -0.,
            'leg_l2_joint': -0.,
            'leg_l3_joint': -init_angle,
            'leg_l4_joint': init_angle * 2,
            'leg_l5_joint': -init_angle,
            'leg_r1_joint': 0.,
            'leg_r2_joint': 0.,
            'leg_r3_joint': -init_angle,
            'leg_r4_joint': init_angle * 2,
            'leg_r5_joint': -init_angle,
        }

    class control:
        # PD Drive parameters:
        stiffness = {'1_joint': 15, '2_joint': 5, '3_joint': 15, '4_joint': 25, '5_joint': 5}
        damping = {'1_joint': 1.0, '2_joint': 0.5, '3_joint': 1.5, '4_joint': 2.5, '5_joint': 0.1}
        # stiffness = {'1_joint': 30, '2_joint': 5, '3_joint': 15, '4_joint': 25, '5_joint': 5}
        # damping = {'1_joint': 3.0, '2_joint': 0.5, '3_joint': 1.5, '4_joint': 2.5, '5_joint': 0.1}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset:
        file = LEGGED_GYM_ROOT_DIR + '/resources/robots/pdd/urdf/pdd_bigfoot.urdf'
        # file = LEGGED_GYM_ROOT_DIR + '/resources/robots/pdd/urdf/pdd_with_fric_damp.urdf'

        name = "pdd"
        base_link_name = "base_link"
        foot_name = "5_link"
        knee_name = "4_link"
        penalize_contacts_on = ['base_link', "1_link", "2_link", "3_link", "4_link"]
        # terminate_after_contacts_on = ['base_link', "1_link", "2_link", "3_link", "4_link"]
        terminate_after_contacts_on = ['base_link']
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        links_to_keep = ['']
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # default_dof_drive_mode = 1  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.1
        thickness = 0.01

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class PddBaseCfgPPO(BaseConfig):
    seed = -1

    class runner:
        num_steps_per_env = 24  # per iteration
        resume = False
        checkpoint = -1  # -1 = last saved model
