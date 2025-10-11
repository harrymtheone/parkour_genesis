import cv2
import numpy as np
import torch

from legged_gym.envs.T1.t1_base_env import T1BaseEnv
from legged_gym.envs.base.utils import ObsBase
from legged_gym.utils.math import torch_rand_float


class ActorObs(ObsBase):
    def __init__(self, proprio, depth, scan, est):
        super().__init__()
        self.proprio = proprio.clone()
        self.depth = depth
        self.scan = scan.clone()
        self.est = est.clone()

    def no_depth(self):
        return ObsNoDepth(self.proprio, self.scan, self.est)


class ObsNoDepth(ObsBase):
    def __init__(self, proprio, scan, est):
        super().__init__()
        self.proprio = proprio.clone()
        self.scan = scan.clone()
        self.est = est.clone()

    @torch.compiler.disable
    def mirror(self):
        return ObsNoDepth(
            mirror_proprio_by_x(self.proprio),
            torch.flip(self.scan, dims=[2]),
            self.mirror_est_by_x(self.est)
        )

    @staticmethod
    def mirror_dof_prop_by_x(dof_prop: torch.Tensor):
        dof_prop_mirrored = dof_prop.clone()
        mirror_dof_prop_by_x(dof_prop_mirrored, 0)
        return dof_prop_mirrored

    @staticmethod
    def mirror_est_by_x(est: torch.Tensor):
        est = est.clone()
        est[:, 1:3] = -est[:, 1:3]
        return est


class CriticObs(ObsBase):
    def __init__(self, est_gt, priv_his, scan, edge_mask):
        super().__init__()
        self.est_gt = est_gt.clone()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()


class T1OdomNegEnvironment(T1BaseEnv):

    def _init_robot_props(self):
        super()._init_robot_props()

        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

        self.ankle_indices = self.sim.create_indices(
            self.sim.get_full_names("Ankle", False), False)

        self.cam_link_indices = self.sim.create_indices(
            self.sim.get_full_names('Trunk', True), True)

    def _init_buffers(self):
        super()._init_buffers()
        self.goal_distance = self._zero_tensor(self.num_envs)
        self.last_goal_distance = self._zero_tensor(self.num_envs)

        self.scan_dev_xy = self._zero_tensor(self.num_envs, 1, 2)
        self.scan_dev_z = self._zero_tensor(self.num_envs, 1)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        self.scan_dev_xy[:] = torch_rand_float(-0.03, 0.03, (self.num_envs, 2), device=self.device).unsqueeze(1)
        self.scan_dev_z[:] = torch_rand_float(-0.03, 0.03, (self.num_envs, 1), device=self.device)

    def _post_physics_pre_step(self):
        super()._post_physics_pre_step()

        if self.sim.terrain is not None:
            self.goal_distance[:] = torch.norm(self.cur_goals[:, :2] - self.sim.root_pos[:, :2], dim=1)

    def _post_physics_post_step(self):
        super()._post_physics_post_step()
        self.last_goal_distance[:] = self.goal_distance

    def _compute_observations(self):
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

        self._compute_ref_state()

        clock = torch.stack(self._get_clock_input(), dim=1)
        command_input = torch.cat((clock, self.commands[:, :3] * self.commands_scale), dim=1)

        # proprio observation
        proprio = torch.cat((
            base_ang_vel * self.obs_scales.ang_vel,  # 3
            projected_gravity,  # 3
            command_input,  # 5
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
            command_input,  # 5D
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
        scan_noisy = torch.clip(self.sim.root_pos[:, 2:3] - self.get_scan(noisy=True) - self.base_height.unsqueeze(1), -1, 1.)
        scan_noisy = scan_noisy.view(self.num_envs, *self.cfg.env.scan_shape)

        base_edge_mask = -0.5 + self.get_edge_mask().float().view(self.num_envs, *self.cfg.env.scan_shape)
        scan_edge = torch.stack([scan_noisy, base_edge_mask], dim=1)

        if self.cfg.sensors.activated:
            depth = torch.cat([self.sensors.get('depth_0'), self.sensors.get('depth_1')], dim=1).half()
        else:
            depth = None

        est = torch.cat([
            base_lin_vel * self.obs_scales.lin_vel,  # 3
        ], dim=-1)

        self.actor_obs = ActorObs(proprio, depth, scan_edge, est)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # compose critic observation
        est_gt = torch.cat([
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
        ], dim=-1)

        # hmap = root_height - scan
        # base_height = root_height - scan.mean()
        # new_hmap = scan - scan.mean()
        # hmap = base_height - new_hmap
        scan = torch.clip(self.sim.root_pos[:, 2:3] - self.get_scan(noisy=False) - self.base_height.unsqueeze(1), -1, 1.)
        scan = scan.view(self.num_envs, *self.cfg.env.scan_shape)

        reset_flag = self.episode_length_buf <= 1
        self.critic_his_buf.append(priv_obs, reset_flag)

        self.critic_obs = CriticObs(est_gt, self.critic_his_buf.get(), scan, base_edge_mask)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            # self._draw_body_hmap()
            # self._draw_hmap_from_depth()
            # self.draw_cloud_from_depth()
            self._draw_goals()
            # self._draw_camera()
            # self._draw_link_COM(whole_body=False)
            # self._draw_feet_at_edge()
            # self._draw_foothold()

            # self._draw_height_field(draw_guidance=True)
            # self._draw_edge()

        if self.cfg.sensors.activated:
            depth_img = self.sensors.get('depth_0', get_depth=True)[self.lookat_id].cpu().numpy()
            depth_img = (depth_img - self.cfg.sensors.depth_0.near_clip) / self.cfg.sensors.depth_0.far_clip
            img = np.clip(depth_img * 255, 0, 255).astype(np.uint8)

            cv2.imshow("depth_front", cv2.resize(img, (320, 320)))
            cv2.waitKey(1)

        if self.cfg.sensors.activated:
            depth_img = self.sensors.get('depth_1', get_depth=True)[self.lookat_id].cpu().numpy()
            depth_img = (depth_img - self.cfg.sensors.depth_0.near_clip) / self.cfg.sensors.depth_0.far_clip
            img = np.clip(depth_img * 255, 0, 255).astype(np.uint8)

            cv2.imshow("depth_back", cv2.resize(img, (320, 320)))
            cv2.waitKey(1)

        super().render()

    def _reward_joint_pos_flat(self):
        diff = self.sim.dof_pos - torch.where(
            self.is_zero_command.unsqueeze(1),
            self.init_state_dof_pos,
            self.ref_dof_pos
        )
        diff = torch.norm(diff[:, self.dof_activated], dim=1)

        rew = torch.exp(-diff * 2) - 0.2 * diff.clamp(0, 0.5)
        rew[self.env_class >= 2] = 0.
        return rew

    def _reward_joint_pos_parkour(self):
        diff = self.sim.dof_pos - torch.where(
            self.is_zero_command.unsqueeze(1),
            self.init_state_dof_pos,
            self.ref_dof_pos
        )
        diff = torch.norm(diff[:, self.dof_activated], dim=1)

        rew = torch.exp(-diff * 2) - 0.2 * diff.clamp(0, 0.5)
        rew[self.env_class < 2] = 0.
        return rew

    def _reward_feet_contact_number_flat(self):
        swing = self._get_swing_mask()
        stance = self._get_stance_mask()
        contact = self.contact_filt

        rew = self._zero_tensor(self.num_envs, len(self.feet_indices))
        rew[swing] = torch.where(contact[swing], -0.3, 1.0)
        rew[stance] = torch.where(contact[stance], 1.0, -0.3)

        rew[self.env_class >= 2] = 0.
        return torch.mean(rew, dim=1)

    def _reward_feet_contact_number_parkour(self):
        swing = self._get_swing_mask()
        stance = self._get_stance_mask()
        contact = self.contact_filt

        rew = self._zero_tensor(self.num_envs, len(self.feet_indices))
        rew[swing] = torch.where(contact[swing], -0.3, 1.0)
        rew[stance] = torch.where(contact[stance], 1.0, -0.3)

        rew[self.env_class < 2] = 0.
        return torch.mean(rew, dim=1)

    def _reward_default_dof_pos(self):
        return (self.sim.dof_pos - self.init_state_dof_pos).abs().sum(dim=1)

    def _reward_default_dof_pos_yr(self):
        assert self.yaw_roll_dof_indices is not None
        joint_diff = self.sim.dof_pos - self.init_state_dof_pos
        yaw_roll = joint_diff[:, self.yaw_roll_dof_indices].abs()
        return (yaw_roll - 0.1).clip(min=0, max=50).sum(dim=1)

    def _reward_feet_distance(self):
        foot_pos = self.sim.link_pos[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        penalty_min = torch.clamp(self.cfg.rewards.min_dist - foot_dist, 0., 0.5)
        penalty_max = torch.clamp(foot_dist - self.cfg.rewards.max_dist, 0., 0.5)
        return penalty_min + penalty_max

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        knee_pos = self.sim.link_pos[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        penalty_min = torch.clamp(self.cfg.rewards.min_dist - knee_dist, 0., 0.5)
        penalty_max = torch.clamp(knee_dist - self.cfg.rewards.max_dist / 2, 0., 0.5)
        return penalty_min + penalty_max

    def _reward_feet_rotation(self):
        return self.feet_euler_xyz[:, :, :2].square().sum(dim=[1, 2])

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        return torch.abs(self.base_height - self.cfg.rewards.base_height_target)

    def _reward_base_acc(self):
        root_acc = self.last_root_vel - self.sim.root_lin_vel
        return torch.norm(root_acc, dim=1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_penalize_vy(self):
        rew = torch.abs(self.base_lin_vel[:, 1])
        rew[self.env_class < 100] = 0.
        return rew

    def _reward_stall(self):
        lin_vel_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        lin_vel_cmd_xy = torch.norm(self.commands[:, :2], dim=1)

        lin_vel_stall = (lin_vel_xy < self.cfg.commands.lin_vel_clip) & (lin_vel_cmd_xy > 0)
        ang_vel_stall = (torch.abs(self.base_ang_vel[:, 2]) < self.cfg.commands.ang_vel_clip) & (torch.abs(self.commands[:, 2]) > 0)

        return (lin_vel_stall | ang_vel_stall).float()

    def _reward_torques_ankle(self):
        rew = torch.sum(torch.square(self.torques[:, self.ankle_indices]), dim=1)
        rew[self.is_zero_command] = 0.
        return rew


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
