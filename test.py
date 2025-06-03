{
  "env": {
    "value": {
      "n_scan": 512,
      "num_envs": 4096,
      "n_proprio": 45,
      "scan_shape": [
        32,
        16
      ],
      "env_spacing": 3,
      "num_actions": 12,
      "len_prop_his": 5,
      "len_depth_his": 1,
      "send_timeouts": true,
      "len_critic_his": 50,
      "num_critic_obs": 91,
      "episode_length_s": 30,
      "reach_goal_delay": 0.05,
      "next_goal_threshold": 0.4,
      "num_future_goal_obs": 2
    }
  },
  "sim": {
    "value": {
      "dt": 0.005,
      "physx": {
        "num_threads": 10,
        "rest_offset": 0,
        "solver_type": 1,
        "contact_offset": 0.01,
        "contact_collection": 2,
        "max_gpu_contact_pairs": 8388608,
        "num_position_iterations": 4,
        "num_velocity_iterations": 0,
        "bounce_threshold_velocity": 0.5,
        "max_depenetration_velocity": 1,
        "default_buffer_size_multiplier": 5
      },
      "gravity": [
        0,
        0,
        -9.81
      ],
      "up_axis": 1,
      "substeps": 1
    }
  },
  "play": {
    "value": {
      "control": false
    }
  },
  "seed": {
    "value": -1
  },
  "asset": {
    "value": {
      "file": "/root/extreme_parkour/legged_gym/robots/go1/urdf/go1.urdf",
      "density": 0.001,
      "armature": 0,
      "friction": 0,
      "foot_name": "foot",
      "stiffness": 0,
      "thickness": 0.01,
      "hip_dof_name": "hip",
      "calf_dof_name": "calf",
      "fix_base_link": false,
      "links_to_keep": [
        ""
      ],
      "base_link_name": "base",
      "linear_damping": 0,
      "thigh_dof_name": "thigh",
      "angular_damping": 0,
      "disable_gravity": false,
      "self_collisions": 1,
      "max_linear_velocity": 1000,
      "max_angular_velocity": 1000,
      "penalize_contacts_on": [
        "base",
        "thigh",
        "calf"
      ],
      "collapse_fixed_joints": true,
      "default_dof_drive_mode": 3,
      "flip_visual_attachments": true,
      "terminate_after_contacts_on": [],
      "replace_cylinder_with_capsule": true
    }
  },
  "noise": {
    "value": {
      "add_noise": true,
      "noise_level": 1,
      "noise_scales": {
        "ang_vel": 0.2,
        "dof_pos": 0.01,
        "dof_vel": 1.5,
        "gravity": 0.05,
        "lin_vel": 0.1,
        "height_measurements": 0.1
      }
    }
  },
  "_wandb": {
    "value": {
      "m": [],
      "t": {
        "1": [
          1,
          41,
          55
        ],
        "2": [
          1,
          41,
          55
        ],
        "3": [
          13,
          16,
          23,
          55,
          61
        ],
        "4": "3.8.10",
        "5": "0.19.11",
        "8": [
          5
        ],
        "12": "0.19.11",
        "13": "linux-x86_64"
      },
      "cli_version": "0.19.11",
      "python_version": "3.8.10"
    }
  },
  "policy": {
    "value": {
      "init_noise_std": 1,
      "actor_hidden_dims": [
        512,
        256,
        128
      ],
      "critic_hidden_dims": [
        512,
        256,
        128
      ],
      "obs_gru_hidden_size": 64,
      "use_recurrent_policy": true,
      "recon_gru_hidden_size": 256
    }
  },
  "runner": {
    "value": {
      "resume": false,
      "checkpoint": -1,
      "runner_name": "rl_dreamer",
      "save_interval": 100,
      "algorithm_name": "ppo_wmp",
      "max_iterations": 100000,
      "num_steps_per_env": 24
    }
  },
  "viewer": {
    "value": {
      "pos": [
        10,
        0,
        6
      ],
      "lookat": [
        11,
        5,
        3
      ],
      "ref_env": 0
    }
  },
  "control": {
    "value": {
      "damping": {
        "joint": 0.5
      },
      "activated": [
        "hip",
        "thigh",
        "calf"
      ],
      "stiffness": {
        "joint": 20
      },
      "decimation": 4,
      "action_scale": 0.25
    }
  },
  "rewards": {
    "value": {
      "scales": {
        "dof_acc": -2.5e-7,
        "hip_pos": -0.5,
        "torques": -0.00001,
        "collision": -1,
        "feet_edge": -0.3,
        "lin_vel_z": -1,
        "ang_vel_xy": -0.05,
        "base_height": -1,
        "orientation": 0.5,
        "base_stumble": -1,
        "feet_stumble": -5,
        "tracking_yaw": 0.5,
        "dof_pos_limits": -10,
        "feet_clearance": -0.01,
        "tracking_lin_vel": 1.5,
        "action_smoothness": -0.003,
        "default_joint_pos": 0.1,
        "tracking_goal_vel": 1.5
      },
      "tracking_sigma": 5,
      "rew_norm_factor": 1,
      "EMA_update_alpha": 0.99,
      "base_height_target": 0.3,
      "feet_height_target": 0.05,
      "soft_dof_pos_limit": 0.9,
      "use_guidance_terrain": true,
      "only_positive_rewards": false,
      "only_positive_rewards_until_epoch": 100
    }
  },
  "sensors": {
    "value": {
      "depth_0": {
        "pitch": 28,
        "resized": [
          64,
          64
        ],
        "far_clip": 2,
        "position": [
          0.27,
          0,
          0.086
        ],
        "near_clip": 0,
        "delay_prop": [
          5,
          1
        ],
        "resolution": [
          64,
          64
        ],
        "data_format": "depth",
        "pitch_range": [
          -1,
          1
        ],
        "horizontal_fov": 58,
        "position_range": [
          [
            -0.01,
            0.01
          ],
          [
            -0.01,
            0.01
          ],
          [
            -0.01,
            0.01
          ]
        ],
        "update_interval": 5,
        "dis_noise_global": 0,
        "link_attached_to": "base",
        "dis_noise_gaussian": 0
      },
      "activated": true
    }
  },
  "terrain": {
    "value": {
      "y_range": [
        -0.4,
        0.4
      ],
      "num_cols": 20,
      "num_rows": 10,
      "max_error": 0.1,
      "base_pts_x": [
        -0.2,
        -0.14666666666666667,
        -0.09333333333333334,
        -0.04000000000000001,
        0.013333333333333336,
        0.06666666666666665,
        0.12,
        0.17333333333333334,
        0.22666666666666668,
        0.28,
        0.3333333333333333,
        0.38666666666666666,
        0.44,
        0.49333333333333335,
        0.5466666666666666,
        0.6
      ],
      "base_pts_y": [
        -0.2,
        -0.14285714285714285,
        -0.08571428571428572,
        -0.02857142857142858,
        0.02857142857142858,
        0.08571428571428574,
        0.14285714285714285,
        0.2
      ],
      "body_pts_x": [
        -0.2,
        -0.06666666666666668,
        0.06666666666666665,
        0.2
      ],
      "body_pts_y": [
        -0.2,
        -0.06666666666666668,
        0.06666666666666665,
        0.2
      ],
      "curriculum": true,
      "feet_pts_x": [
        -0.1,
        0.1
      ],
      "feet_pts_y": [
        -0.1,
        0.1
      ],
      "scan_pts_x": [
        -0.5,
        -0.4483870967741935,
        -0.3967741935483871,
        -0.34516129032258064,
        -0.2935483870967742,
        -0.24193548387096775,
        -0.19032258064516128,
        -0.13870967741935486,
        -0.08709677419354839,
        -0.035483870967741915,
        0.016129032258064502,
        0.06774193548387097,
        0.11935483870967745,
        0.17096774193548392,
        0.22258064516129028,
        0.27419354838709675,
        0.3258064516129032,
        0.3774193548387097,
        0.42903225806451617,
        0.4806451612903226,
        0.532258064516129,
        0.5838709677419356,
        0.635483870967742,
        0.6870967741935483,
        0.7387096774193549,
        0.7903225806451613,
        0.8419354838709678,
        0.8935483870967742,
        0.9451612903225806,
        0.9967741935483873,
        1.0483870967741935,
        1.1
      ],
      "scan_pts_y": [
        -0.4,
        -0.3466666666666667,
        -0.29333333333333333,
        -0.24,
        -0.18666666666666668,
        -0.13333333333333336,
        -0.08000000000000002,
        -0.026666666666666672,
        0.026666666666666672,
        0.08000000000000002,
        0.1333333333333333,
        0.18666666666666665,
        0.24,
        0.29333333333333333,
        0.3466666666666667,
        0.4
      ],
      "border_size": 5,
      "restitution": 0,
      "terrain_dict": {
        "gap": 0,
        "pit": 0,
        "parkour": 0,
        "discrete": 0,
        "stairs_up": 1,
        "parkour_box": 1,
        "parkour_gap": 1,
        "rough_slope": 1,
        "stairs_down": 1,
        "parkour_flat": 0,
        "parkour_step": 1,
        "smooth_slope": 3,
        "parkour_stair": 1,
        "stepping_stone": 0
      },
      "terrain_size": [
        8,
        8
      ],
      "origin_zero_z": false,
      "simplify_grid": false,
      "hf2mesh_method": "grid",
      "max_difficulty": false,
      "slope_treshold": 1.5,
      "vertical_scale": 0.005,
      "measure_heights": true,
      "static_friction": 1,
      "description_type": "trimesh",
      "dynamic_friction": 1,
      "horizontal_scale": 0.02,
      "roughness_height": [
        0.02,
        0.04
      ],
      "downsampled_scale": 0.075,
      "edge_width_thresh": 0.05,
      "terrain_parkour_size": [
        18,
        4
      ],
      "height_update_interval": 5,
      "max_init_terrain_level": 9,
      "horizontal_scale_downsample": 0.05
    }
  },
  "commands": {
    "value": {
      "ratio_EMA": 0.8,
      "flat_ranges": {
        "lin_vel_x": [
          -0.8,
          0.8
        ],
        "lin_vel_y": [
          -0.8,
          0.8
        ],
        "ang_vel_yaw": [
          -1,
          1
        ]
      },
      "ang_vel_clip": 0.4,
      "lin_vel_clip": 0.2,
      "num_commands": 4,
      "stair_ranges": {
        "heading": [
          -3.14,
          3.14
        ],
        "lin_vel_x": [
          -0.6,
          0.6
        ],
        "lin_vel_y": [
          -0.4,
          0.4
        ],
        "ang_vel_yaw": [
          -1,
          1
        ]
      },
      "parkour_ranges": {
        "lin_vel_x": [
          0.6,
          1.2
        ],
        "ang_vel_yaw": [
          -1,
          1
        ]
      },
      "resampling_time": 8
    }
  },
  "algorithm": {
    "value": {
      "lam": 0.95,
      "gamma": 0.99,
      "use_amp": true,
      "schedule": "adaptive",
      "clip_param": 0.2,
      "desired_kl": 0.01,
      "entropy_coef": 0.01,
      "learning_rate": 0.0002,
      "max_grad_norm": 1,
      "value_loss_coef": 1,
      "num_mini_batches": 4,
      "num_learning_epochs": 5,
      "continue_from_last_std": true,
      "use_clipped_value_loss": true
    }
  },
  "init_state": {
    "value": {
      "pos": [
        0,
        0,
        0.42
      ],
      "rot": [
        0,
        0,
        0,
        1
      ],
      "ang_vel": [
        0,
        0,
        0
      ],
      "lin_vel": [
        0,
        0,
        0
      ],
      "default_joint_angles": {
        "FL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RL_hip_joint": 0.1,
        "RR_hip_joint": -0.1,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1,
        "RR_thigh_joint": 1
      }
    }
  },
  "domain_rand": {
    "value": {
      "add_dof_lag": false,
      "add_imu_lag": false,
      "push_robots": false,
      "action_delay": true,
      "dof_lag_range": [
        0,
        40
      ],
      "imu_lag_range": [
        1,
        10
      ],
      "push_duration": [
        0,
        0.05,
        0.1,
        0.15
      ],
      "randomize_com": true,
      "friction_range": [
        0.1,
        1.3
      ],
      "push_force_max": [
        [
          -300,
          600
        ],
        [
          -400,
          400
        ],
        [
          -5,
          5
        ]
      ],
      "push_interval_s": 6,
      "push_torque_max": [
        0,
        0
      ],
      "randomize_gains": true,
      "added_mass_range": [
        -0.5,
        3
      ],
      "compliance_range": [
        0.5,
        1.5
      ],
      "randomize_torque": true,
      "randomize_dof_lag": true,
      "randomize_imu_lag": true,
      "randomize_start_z": true,
      "restitution_range": [
        0,
        0.4
      ],
      "action_delay_range": [
        [
          0,
          2
        ]
      ],
      "motor_offset_range": [
        -0.035,
        0.035
      ],
      "randomize_friction": true,
      "joint_coulomb_range": [
        0.1,
        1
      ],
      "joint_viscous_range": [
        0.1,
        0.9
      ],
      "kd_multiplier_range": [
        0.8,
        1.2
      ],
      "kp_multiplier_range": [
        0.8,
        1.2
      ],
      "randomize_base_mass": true,
      "randomize_link_mass": true,
      "randomize_start_pos": true,
      "randomize_start_vel": true,
      "randomize_start_yaw": true,
      "joint_armature_range": [
        0.0001,
        0.05
      ],
      "joint_stiffness_range": [
        0,
        0
      ],
      "randomize_start_pitch": true,
      "com_displacement_range": [
        -0.05,
        0.05
      ],
      "randomize_action_delay": true,
      "randomize_motor_offset": true,
      "randomize_joint_damping": false,
      "randomize_start_dof_pos": true,
      "randomize_start_dof_vel": true,
      "randomize_start_z_range": 0.1,
      "torque_multiplier_range": [
        0.8,
        1.2
      ],
      "randomize_joint_armature": true,
      "randomize_joint_friction": false,
      "action_delay_update_steps": 48000,
      "randomize_joint_stiffness": false,
      "randomize_start_pos_range": 0.3,
      "randomize_start_vel_range": 0.1,
      "randomize_start_yaw_range": 1.2,
      "link_mass_multiplier_range": [
        0.9,
        1.1
      ],
      "push_duration_update_steps": 24000,
      "randomize_coulomb_friction": false,
      "randomize_dof_lag_each_step": false,
      "randomize_imu_lag_each_step": false,
      "randomize_start_pitch_range": 0.1,
      "randomize_start_dof_pos_range": 0.15,
      "randomize_start_dof_vel_range": 0.15,
      "joint_damping_multiplier_range": [
        0.3,
        1.5
      ],
      "joint_friction_multiplier_range": [
        0.01,
        1.15
      ],
      "randomize_action_delay_each_step": false
    }
  },
  "world_model": {
    "value": {
      "n_deter": 512,
      "n_stoch": 32,
      "n_cnn_enc": 4096,
      "n_mlp_enc": 1024,
      "n_discrete": 32,
      "hidden_size": 512,
      "unimix_ratio": 0.01,
      "state_initial": "learned",
      "step_interval": 5
    }
  },
  "normalization": {
    "value": {
      "obs_scales": {
        "quat": 1,
        "ang_vel": 0.25,
        "dof_pos": 1,
        "dof_vel": 0.05,
        "lin_vel": 2
      },
      "clip_actions": 1.2,
      "scan_norm_bias": 0.3,
      "clip_observations": 100,
      "feet_height_correction": 0.02
    }
  }
}