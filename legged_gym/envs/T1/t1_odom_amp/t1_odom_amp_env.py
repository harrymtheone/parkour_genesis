from collections import deque

import joblib
import torch

from legged_gym.envs.T1.t1_base_env import T1BaseEnv
from legged_gym.envs.base.utils import ObsBase
from legged_gym.utils.math import torch_rand_float, transform_by_yaw, quat_rotate_inverse


class ActorObs(ObsBase):
    def __init__(self, proprio, depth, priv_actor, rough_scan, scan, env_class):
        super().__init__()
        self.proprio = proprio.clone()
        self.depth = depth
        self.priv_actor = priv_actor.clone()
        self.rough_scan = rough_scan.clone()
        self.scan = scan.clone()
        self.env_class = env_class.clone()

    def no_depth(self):
        return ObsNoDepth(self.proprio, self.priv_actor, self.scan)


class ObsNoDepth(ObsBase):
    def __init__(self, proprio, priv_actor, scan):
        super().__init__()
        self.proprio = proprio.clone()
        self.priv_actor = priv_actor.clone()
        self.scan = scan.clone()

    @torch.compiler.disable
    def mirror(self):
        return ObsNoDepth(
            mirror_proprio_by_x(self.proprio),
            mirror_priv_by_x(self.priv_actor),
            torch.flip(self.scan, dims=[2]),
        )

    @staticmethod
    def mirror_dof_prop_by_x(dof_prop: torch.Tensor):
        dof_prop_mirrored = dof_prop.clone()
        mirror_dof_prop_by_x(dof_prop_mirrored, 0)
        return dof_prop_mirrored


class CriticObs(ObsBase):
    def __init__(self, priv_actor, priv_his, scan, edge_mask):
        super().__init__()
        self.priv_actor = priv_actor.clone()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()


class T1OdomAmpEnv(T1BaseEnv):

    def _init_robot_props(self):
        super()._init_robot_props()

        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

        self.head_link_indices = self.sim.create_indices(
            self.sim.get_full_names(['H2'], True), True)

        self.waist_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist'], False), False)

        self.ankle_roll_idx = [16, 22]
        self.ankle_pitch_idx = [15, 21]
        self.hip_pitch_idx = [11, 17]
        self.hip_roll_idx = [12, 18]
        self.hip_yaw_idx = [13, 19]
        self.knee_idx = [14, 20]
        self.waist_idx = [10]

        # 这里需要修改
        self.left_leg_indices = [11, 12, 13, 14, 15, 16]
        self.right_leg_indices = [17, 18, 19, 20, 21, 22]
        self.motion_ref_dof_pos = self._zero_tensor(self.num_envs, 23)

    def _init_buffers(self):
        super()._init_buffers()
        self.goal_distance = self._zero_tensor(self.num_envs)
        self.last_goal_distance = self._zero_tensor(self.num_envs)

        self.scan_dev_xy = self._zero_tensor(self.num_envs, 1, 2)
        self.scan_dev_z = self._zero_tensor(self.num_envs, 1)

        self.gen_amp_history = deque(maxlen=self.cfg.amp.amp_obs_hist_steps)
        self.amp_obs_hist_steps = self.cfg.amp.amp_obs_hist_steps
        for _ in range(self.cfg.amp.amp_obs_hist_steps):
            self.gen_amp_history.append(torch.zeros(
                self.num_envs, self.cfg.amp.num_single_amp_obs, dtype=torch.float, device=self.device))
        self.reset_amp_obs = None
        self.reset_idx = None

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        self.scan_dev_xy[:] = torch_rand_float(-0.03, 0.03, (self.num_envs, 2), device=self.device).unsqueeze(1)
        self.scan_dev_z[:] = torch_rand_float(-0.03, 0.03, (self.num_envs, 1), device=self.device)

    def _post_physics_pre_step(self):
        super()._post_physics_pre_step()
        self.goal_distance[:] = torch.norm(self.cur_goals[:, :2] - self.sim.root_pos[:, :2], dim=1)

    def _post_physics_post_step(self):
        super()._post_physics_post_step()
        self.last_goal_distance[:] = self.goal_distance

    def get_scan(self, noisy=False):
        # convert height points coordinate to world frame
        points = transform_by_yaw(
            self.scan_points,
            self.base_euler[:, 2:3].repeat(1, self.num_scan)
        ).unflatten(0, (self.num_envs, -1))

        points[:] += self.sim.root_pos.unsqueeze(1) + self.cfg.terrain.border_size

        if not noisy:
            return self._get_heights(points)

        z_gauss = 0.03 * torch.randn(self.num_envs, self.scan_points.size(1), device=self.device)

        points[:, :, :2] += self.scan_dev_xy
        heights = self._get_heights(points)
        heights[:] += self.scan_dev_z + z_gauss
        return heights

    def _post_physics_step(self):
        self.global_counter += 1

        self._refresh_variables()
        self._post_physics_pre_step()

        self._compute_amp_obs()  # compute amp observations for discriminator before reset env

        # compute observations, rewards, resets, ...
        if self.init_done:
            self._compute_reward()
            self._check_termination()
            reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(reset_ids) > 0:
                self._reset_idx(reset_ids)
                self._refresh_variables()

        self._post_physics_mid_step()
        if self.cfg.sensors.activated:
            # in case of contact, the sensor will be attached to the link
            if self.sensors is not None:
                self.sensors.update(self.global_counter, self.sim.link_pos, self.sim.link_quat, self.reset_buf)

        if self.init_done:
            self._compute_observations(reset_ids)
        else:
            self._compute_observations()
        self._post_physics_post_step()

        if not self.sim.headless:
            self.render()
            self.joystick_handler.handle_device_input()

    def _compute_observations(self, reset_ids=[]):
        """
        Computes observations
        """
        # add lag to sensor observation
        if self.cfg.domain_rand.add_dof_lag:
            dof_data = self.dof_lag_buf.get()
            dof_pos, dof_vel = dof_data[:, 0], dof_data[:, 1]
        else:
            dof_pos, dof_vel = self.sim.dof_pos, self.sim.dof_vel

        if self.cfg.domain_rand.add_imu_lag:
            imu_data = self.imu_lag_buf.get()
            base_quat, base_lin_vel, base_ang_vel = imu_data[..., :4], imu_data[..., 4:7], imu_data[..., 7:]
            projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        else:
            base_lin_vel = self.base_lin_vel
            base_ang_vel = self.base_ang_vel
            projected_gravity = self.projected_gravity

        # add noise to sensor observation
        dof_pos = self._add_noise(dof_pos, self.cfg.noise.noise_scales.dof_pos)
        dof_vel = self._add_noise(dof_vel, self.cfg.noise.noise_scales.dof_vel)
        base_lin_vel = self._add_noise(base_lin_vel, self.cfg.noise.noise_scales.lin_vel)
        base_ang_vel = self._add_noise(base_ang_vel, self.cfg.noise.noise_scales.ang_vel)
        projected_gravity = self._add_noise(projected_gravity, self.cfg.noise.noise_scales.gravity)

        # proprio observation
        proprio = torch.cat((
            base_ang_vel * self.obs_scales.ang_vel,  # 3
            projected_gravity,  # 3
            self.commands[:, :3] * self.commands_scale,  # 5
            (dof_pos - self.init_state_dof_pos)[:, self.dof_activated] * self.obs_scales.dof_pos,  # 12D
            dof_vel[:, self.dof_activated] * self.obs_scales.dof_vel,  # 12D
            self.last_action_output,  # 12D
        ), dim=-1)

        # explicit privileged information
        priv_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.get_feet_hmap() - self.cfg.normalization.feet_height_correction,  # 8
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler * self.obs_scales.quat,  # 3
            self.commands[:, :3] * self.commands_scale,  # 5D
            self.last_action_output,  # 12D
            (self.sim.dof_pos - self.init_state_dof_pos)[:, self.dof_activated] * self.obs_scales.dof_pos,  # 12D
            self.sim.dof_vel[:, self.dof_activated] * self.obs_scales.dof_vel,  # 12D
            self.ext_force[:, :2],  # 2
            self.ext_torque,  # 3
            self.sim.payload_masses / 10.,  # 1
            self.sim.contact_forces[:, self.feet_indices, 2] > 5.,  # 2
            self.goal_distance.unsqueeze(1) / 10.,  # 1
        ), dim=-1)

        # compute height map
        rough_scan = torch.clip(self.get_body_hmap() - self.cfg.normalization.scan_norm_bias, -1, 1.)
        rough_scan = rough_scan.view(self.num_envs, *self.cfg.env.scan_shape)

        scan_noisy = torch.clip(self.sim.root_pos[:, 2:3] - self.get_scan(noisy=True) - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan_noisy = scan_noisy.view(self.num_envs, *self.cfg.env.scan_shape)

        base_edge_mask = self.get_edge_mask().float().view(self.num_envs, *self.cfg.env.scan_shape)
        scan_edge = torch.stack([scan_noisy, base_edge_mask], dim=1)

        if self.cfg.sensors.activated:
            depth = self.sensors.get('depth_0').squeeze(2).half()
        else:
            depth = None

        priv_actor = base_lin_vel * self.obs_scales.lin_vel  # 3

        self.actor_obs = ActorObs(proprio, depth, priv_actor, rough_scan, scan_edge, self.env_class)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # compose critic observation
        priv_actor_clean = self.base_lin_vel * self.obs_scales.lin_vel  # 3

        scan = torch.clip(self.sim.root_pos[:, 2:3] - self.get_scan(noisy=False) - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view(self.num_envs, *self.cfg.env.scan_shape)

        reset_flag = self.episode_length_buf <= 1
        self.critic_his_buf.append(priv_obs, reset_flag)

        self.critic_obs = CriticObs(priv_actor_clean, self.critic_his_buf.get(), scan, base_edge_mask)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)

        if len(reset_ids) > 0:
            self._compute_amp_obs(reset_ids)

    def _compute_amp_obs(self, env_ids=[]):
        # update gen_amp_obs_buf after apply action
        gen_amp_obs_buf = self._get_gen_amp_obs()
        if len(env_ids) == 0:
            self.gen_amp_history.append(gen_amp_obs_buf)
            self.reset_idx = None
            self.reset_amp_obs = None
            self.gen_amp_obs_buf = torch.cat([self.gen_amp_history[i] for i in range(self.amp_obs_hist_steps)], dim=1)

        # update gen_amp_obs_buf after reset env, record reset_amp_obs to cal disc reward
        if len(env_ids) > 0:
            self.reset_idx = env_ids
            self.reset_amp_obs = self.gen_amp_obs_buf[env_ids].clone()
            for i in range(self.gen_amp_history.maxlen):
                self.gen_amp_history[i][env_ids] = gen_amp_obs_buf[env_ids]
            self.gen_amp_obs_buf[env_ids] = torch.cat([self.gen_amp_history[i][env_ids] for i in range(self.amp_obs_hist_steps)], dim=1)

    def _get_gen_amp_obs(self):
        gen_amp_obs = []

        for obs_name, obs_info in self.cfg.amp.amp_obs_dict.items():
            if obs_info["using"]:
                func_name = f"_amp_obs_{obs_name}"
                if hasattr(self, func_name):
                    obs_func = getattr(self, func_name)
                    obs = obs_func()
                    obs_scale = torch.tensor(obs_info["obs_scale"], device=self.device)
                    obs = obs * obs_scale
                    gen_amp_obs.append(obs)
                else:
                    raise NotImplementedError(f"Function {func_name} is not implemented.")
        gen_amp_obs = torch.cat(gen_amp_obs, dim=-1)
        return gen_amp_obs

    def _amp_obs_base_height(self):
        return self.base_height.unsqueeze(1)

    def _amp_obs_dof_pos(self):
        return self.sim.dof_pos[:, self.dof_activated]

    def _amp_obs_dof_vel(self):
        return self.sim.dof_vel[:, self.dof_activated]

    def _amp_obs_base_ang_vel(self):
        return self.base_ang_vel

    def _amp_obs_projected_gravity(self):
        return self.projected_gravity

    def _amp_obs_torso_projected_gravity(self):
        torso_quat = self.sim.root_quat
        torso_projected_gravity = quat_rotate_inverse(torso_quat, self.gravity_vec)
        return torso_projected_gravity

    def _amp_obs_base_lin_vel(self):
        return self.base_lin_vel

    def _amp_obs_knee_pos_to_base(self):
        knee_pos_to_base_gl = self.sim.link_pos[:, self.knee_indices, :3] - self.sim.root_pos[:, None, :3]
        knee_pos_to_base_expand = quat_rotate_inverse(self.sim.root_quat.repeat_interleave(len(self.knee_indices), dim=0), knee_pos_to_base_gl.reshape(-1, 3))
        knee_pos_to_base = knee_pos_to_base_expand.reshape(self.num_envs, -1, 3)
        return knee_pos_to_base.reshape(self.num_envs, -1)

    def _amp_obs_feet_pos_to_base(self):
        feet_pos_to_base_gl = self.sim.link_pos[:, self.feet_indices, :3] - self.sim.root_pos[:, None, :3]
        feet_pos_to_base_expand = quat_rotate_inverse(self.sim.root_quat.repeat_interleave(len(self.feet_indices), dim=0), feet_pos_to_base_gl.reshape(-1, 3))
        feet_pos_to_base = feet_pos_to_base_expand.reshape(self.num_envs, -1, 3)
        return feet_pos_to_base.reshape(self.num_envs, -1)

    def _amp_obs_elbow_pos_to_base(self):
        elbow_pos_to_base_gl = self.sim.link_pos[:, self.elbow_indices, :3] - self.sim.root_pos[:, None, :3]
        elbow_pos_to_base_expand = quat_rotate_inverse(self.sim.root_quat.repeat_interleave(len(self.elbow_indices), dim=0),
                                                       elbow_pos_to_base_gl.reshape(-1, 3))
        elbow_pos_to_base = elbow_pos_to_base_expand.reshape(self.num_envs, -1, 3)
        return elbow_pos_to_base.reshape(self.num_envs, -1)

    def _amp_obs_wrist_pos_to_base(self):
        wrist_pos_to_base_gl = self.sim.link_pos[:, self.wrist_indices, :3] - self.sim.root_pos[:, None, :3]
        wrist_pos_to_base_expand = quat_rotate_inverse(self.sim.root_quat.repeat_interleave(len(self.wrist_indices), dim=0),
                                                       wrist_pos_to_base_gl.reshape(-1, 3))
        wrist_pos_to_base = wrist_pos_to_base_expand.reshape(self.num_envs, -1, 3)
        return wrist_pos_to_base.reshape(self.num_envs, -1)

    def get_amp_obs_for_expert_trans(self):
        return self.gen_amp_obs_buf, self.reset_amp_obs, self.reset_idx

    def load_motion_frame(self, motion_path):
        """
        Loads motion data from a .pkl file generated by log_motion.py.
        The file should contain 'data' and 'data_idx' keys.

        Args:
            motion_path (str): The path to the motion data .pkl file.
        """
        try:
            with open(motion_path, 'rb') as f:
                motion_dict = joblib.load(f)

            if "data" not in motion_dict or "data_idx" not in motion_dict:
                print(f"Error: .pkl file at {motion_path} is missing 'data' or 'data_idx' key.")
                self.motion_data = None
                self.motion_data_idx = None
                return

            self.motion_data = torch.tensor(motion_dict["data"], dtype=torch.float32, device=self.device)
            self.motion_data_idx = motion_dict["data_idx"]
            print(f"Successfully loaded {self.motion_data.shape[0]} frames from {motion_path}.")

        except FileNotFoundError:
            print(f"Error: Motion file not found at {motion_path}")
            self.motion_data = None
            self.motion_data_idx = None
        except Exception as e:
            print(f"An unexpected error occurred while loading motion data: {e}")
            self.motion_data = None
            self.motion_data_idx = None

    def get_frame_by_time(self, time):
        return self.motion_data[time, :]

    def amp_visualize_motion(self, time):
        if self.motion_data is None or self.motion_data_idx is None:
            return

        visual_motion_frame = self.get_frame_by_time(time)

        # Extract dof positions using the data_idx map
        dof_pos_indices = self.motion_data_idx["dof_pos"]
        dof_pos_from_frame = visual_motion_frame[dof_pos_indices]

        # The recorded dof_pos should already have the correct size (13), including the waist.
        # We repeat it for all environments for visualization.
        full_dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        full_dof_pos[:, :] = dof_pos_from_frame

        self.motion_ref_dof_pos[:] = full_dof_pos

    def _reward_default_dof_pos(self):
        return (self.sim.dof_pos - self.init_state_dof_pos).abs().sum(dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_feet_clearance(self):
        rew = (self.feet_height / self.cfg.rewards.feet_height_target).clip(min=-1, max=1)

        rew[self.contact_filt | self.is_zero_command.unsqueeze(1)] = 0.
        # rew *= 1.0 - self._get_soft_stance_mask()
        return rew.sum(dim=1)


@torch.jit.script
def mirror_dof_prop_by_x(prop: torch.Tensor, start_idx: int):
    left_idx = start_idx + torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long, device=prop.device)
    right_idx = left_idx + 6

    dof_left = prop[:, left_idx].clone()
    prop[:, left_idx] = prop[:, right_idx]
    prop[:, right_idx] = dof_left

    prop[:, start_idx + 0] *= -1.  # invert waist

    invert_idx = start_idx + torch.tensor([2, 3, 6], dtype=torch.long, device=prop.device)
    prop[:, invert_idx] *= -1.
    prop[:, invert_idx + 6] *= -1.


@torch.jit.script
def mirror_proprio_by_x(prop: torch.Tensor) -> torch.Tensor:
    prop = prop.clone()

    # base angular velocity, [0:3], [-roll, pitch, -yaw]
    prop[:, 0] *= -1.
    prop[:, 2] *= -1.

    # projected gravity, [3:6], [x, -y, z]
    prop[:, 4] *= -1.

    # commands [6:11], [clock_l <--> clock_r, x, -y, -yaw]
    clock_l = prop[:, 6].clone()
    prop[:, 6] = prop[:, 7]
    prop[:, 7] = clock_l
    prop[:, 9:11] *= -1.

    # dof pos
    mirror_dof_prop_by_x(prop, 11)

    # dof vel
    mirror_dof_prop_by_x(prop, 11 + 13)

    # last actions
    mirror_dof_prop_by_x(prop, 11 + 13 + 13)

    return prop


@torch.jit.script
def mirror_priv_by_x(priv: torch.Tensor) -> torch.Tensor:
    priv = priv.clone()

    # linear velocity, [0:3], [x, -y, z]
    priv[:, 1] *= -1.

    return priv
