import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_config import BaseConfig


class G1BaseCfg(BaseConfig):
    seed = -1

    class play:
        control = False

    class env:
        env_spacing = 3.
        send_timeouts = True

        next_goal_threshold = 0.4
        reach_goal_delay = 0.05
        num_future_goal_obs = 2

    class sensors:
        activated = False

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0

        scan_norm_bias = 0.7
        feet_height_correction = -0.027

        clip_observations = 100.
        clip_actions = 100.

    class terrain:
        # description_type = 'plane'  # plane, heightfield or trimesh
        # description_type = 'heightfield'  # plane, heightfield or trimesh
        description_type = 'trimesh'  # plane, heightfield or trimesh
        max_error = 0.1  # for fast

        y_range = [-0.2, 0.2]

        edge_width_thresh = 0.05
        horizontal_scale = 0.02  # [m] influence computation time by a lot
        horizontal_scale_downsample = 0.05  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 5  # [m]
        roughness_height = [0.0, 0.03]
        simplify_grid = False
        downsampled_scale = 0.075

        curriculum = True
        max_difficulty = False

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True

        base_pts_x = np.linspace(-0.2, 0.2, 8)
        base_pts_y = np.linspace(-0.2, 0.2, 8)
        scan_pts_x = np.linspace(-0.5, 1.1, 32)
        scan_pts_y = np.linspace(-0.4, 0.4, 16)
        body_pts_x = np.linspace(-0.2, 0.2, 4)
        body_pts_y = np.linspace(-0.2, 0.2, 4)
        feet_pts_x = np.linspace(-0.1, 0.1, 2)
        feet_pts_y = np.linspace(-0.1, 0.1, 2)

        height_update_interval = 1  # 1 * dt

        foothold_pts = [(-0.1, 0.12, 10), (-0.05, 0.05, 5), -0.03]  # (a, b, num points)
        foothold_contact_thresh = 0.01

        max_init_terrain_level = 9  # starting curriculum state
        terrain_size = [8., 8.]
        terrain_parkour_size = [18., 4.]

        # trimesh only:
        slope_treshold = 1.  # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False

    class commands:
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 8.  # time before command are changed[s]

        lin_vel_clip = 0.2
        ang_vel_clip = 0.2
        parkour_vel_tolerance = 0.3

        sw_switch = True

        class flat_ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1., 1.]

        class stair_ranges:
            lin_vel_x = [-0.4, 1.2]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1., 1.]  # this value limits the max yaw velocity computed by goal
            heading = [-1.5, 1.5]

        class parkour_ranges:
            lin_vel_x = [0.3, 1.2]  # min value should be greater than lin_vel_clip
            ang_vel_yaw = [-1.0, 1.0]  # this value limits the max yaw velocity computed by goal

    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.02
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.1
            height_measurements = 0.1

    class domain_rand:
        randomize_start_pos = False
        randomize_start_pos_range = 1.0
        randomize_start_vel = False
        randomize_start_vel_range = 0.1
        randomize_start_yaw = False
        randomize_start_yaw_range = 1.2
        randomize_start_z = False
        randomize_start_z_range = 0.1  # (0, 0.05)
        randomize_start_pitch = False
        randomize_start_pitch_range = 0.1  # 5 degree

        randomize_start_dof_pos = False
        randomize_start_dof_pos_range = 0.1
        randomize_start_dof_vel = False
        randomize_start_dof_vel_range = 0.1

        push_robots = False
        push_force_max = ((-200, 200),
                          (-200, 200),
                          (-5, 5))
        push_torque_max = (-50, 50)
        push_interval_s = 5
        push_duration = [0.1, 0.2, 0.3]
        push_duration_update_steps = 2000 * 24

        action_delay = False
        randomize_action_delay = True  # if False, max delay will be used
        action_delay_range = [(0, 5), (0, 10), (0, 15), (0, 20)]
        action_delay_update_steps = 2000 * 24

        add_dof_lag = False
        randomize_dof_lag = True  # if False, max delay will be used
        dof_lag_range = (0, 20)

        add_imu_lag = False
        randomize_imu_lag = True  # if False, max delay will be used
        imu_lag_range = (0, 10)

        randomize_base_mass = False
        added_mass_range = [-2.5, 2.5]

        randomize_link_mass = False
        link_mass_multiplier_range = [0.9, 1.1]

        randomize_com = False
        com_displacement_range = [-0.05, 0.05]

        randomize_friction = False
        friction_range = [0.1, 1.25]
        compliance_range = [0.5, 1.5]
        restitution_range = [0., 0.4]

        randomize_joint_stiffness = False
        joint_stiffness_range = [0., 0.]

        randomize_joint_damping = False
        joint_damping_multiplier_range = [0.3, 1.5]

        randomize_joint_friction = False
        joint_friction_range = [0.0, 2.]

        randomize_joint_armature = False
        joint_armature_range = {
            'default': dict(range=(0.01, 0.05), log_space=False),
            'ankle': dict(dof_ids=(7, 8, 13, 14), range=(0.01, 0.05), log_space=False)
        }

        randomize_coulomb_friction = False
        joint_coulomb_range = [0.1, 1.0]
        joint_viscous_range = [0.1, 0.9]

        randomize_motor_offset = False
        motor_offset_range = [-0.035, 0.035]  # Offset to add to the motor angles

        randomize_gains = False
        kp_multiplier_range = [0.8, 1.2]
        kd_multiplier_range = [0.8, 1.2]

        randomize_torque = False
        torque_multiplier_range = [0.8, 1.2]

    class init_state:
        pos = [0.0, 0.0, 0.85]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = {
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': 0.,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.,

            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': 0.,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.,

            'waist_yaw_joint': 0.,
            # 'waist_roll_joint': 0.,
            # 'waist_pitch_joint': 0.,
        }

    class control:
        # PD Drive parameters:
        stiffness = {'hip': 100, 'knee': 150, 'ankle': 20, 'waist': 200}  # [N*m/rad]
        damping = {'hip': 2, 'knee': 4, 'ankle': 0.2, 'waist': 4}  # [N*m*s/rad]
        activated = ['hip', 'knee', 'ankle', 'waist']

        action_scale = 0.25
        decimation = 10

    class asset:
        file = LEGGED_GYM_ROOT_DIR + '/robots/g1/g1_15dof.urdf'
        name = "g1"
        base_link_name = 'torso_link'
        foot_name = "ankle_roll"
        knee_name = 'knee'
        foot_dof_name = 'ankle'
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["torso", "arm", "pelvis"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        links_to_keep = ['']
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.  # joint props
        linear_damping = 0.  # joint props
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

        use_soft_limits = True
        sim_dof_limit_mul = 10.  # if we are using soft limit, we can relax the simulation limit

        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_dof_torque_limit = 0.9

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.002
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

    class runner:
        num_steps_per_env = 24
        inference_enabled = True

        logger_backend = 'tensorboard'

        resume = False
        checkpoint = -1  # -1 = last saved model

        # logging
        save_interval = 100
