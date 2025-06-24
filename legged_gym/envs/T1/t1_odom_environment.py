import cv2
import numpy as np
import torch

from .t1_base_env import T1BaseEnv
from ..base.utils import ObsBase


class ActorObs(ObsBase):
    def __init__(self, proprio, depth, priv_actor, scan):
        super().__init__()
        self.proprio = proprio.clone()
        self.depth = depth.clone()
        self.priv_actor = priv_actor.clone()
        self.scan = scan.clone()

    def no_depth(self):
        return ObsNoDepth(self.proprio, self.priv_actor, self.scan)


class ObsNoDepth(ObsBase):
    def __init__(self, proprio, priv_actor, scan):
        super().__init__()
        self.proprio = proprio.clone()
        self.priv_actor = priv_actor.clone()
        self.scan = scan.clone()


class CriticObs(ObsBase):
    def __init__(self, priv_his, scan, edge_mask):
        super().__init__()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()


class T1OdomEnvironment(T1BaseEnv):

    def _init_robot_props(self):
        super()._init_robot_props()

        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

        self.head_link_indices = self.sim.create_indices(
            self.sim.get_full_names(['H2'], True), True)

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
            base_ang_vel = self.base_ang_vel
            projected_gravity = self.projected_gravity

        # add noise to sensor observation
        dof_pos = self._add_noise(dof_pos, self.cfg.noise.noise_scales.dof_pos)
        dof_vel = self._add_noise(dof_vel, self.cfg.noise.noise_scales.dof_vel)
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
            self.get_body_hmap() - self.cfg.normalization.scan_norm_bias,  # 16
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler * self.obs_scales.quat,  # 3
            command_input,  # 5D
            self.last_action_output,  # 12D
            (self.sim.dof_pos - self.init_state_dof_pos)[:, self.dof_activated] * self.obs_scales.dof_pos,  # 12D
            self.sim.dof_vel[:, self.dof_activated] * self.obs_scales.dof_vel,  # 12D
            self.ext_force[:, :2],  # 2
            self.ext_torque,  # 3
            self.sim.friction_coeffs,  # 1
            self.sim.payload_masses / 10.,  # 1
            self.sim.contact_forces[:, self.feet_indices, 2] > 5.,  # 2
        ), dim=-1)

        priv_actor_obs = self.base_lin_vel * self.obs_scales.lin_vel  # 3

        # compute height map
        scan = torch.clip(self.sim.root_pos[:, 2:3] - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))
        base_edge_mask = self.get_edge_mask().float().view((self.num_envs, *self.cfg.env.scan_shape))
        scan_edge = torch.stack([scan, base_edge_mask], dim=1)

        depth = self.sensors.get('depth_0').squeeze(2).half()
        self.actor_obs = ActorObs(proprio, depth, priv_actor_obs, scan_edge)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # compose critic observation
        reset_flag = self.episode_length_buf <= 1
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs = CriticObs(self.critic_his_buf.get(), scan, base_edge_mask)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            # self.draw_hmap_from_depth()
            # self.draw_cloud_from_depth()
            self._draw_goals()
            # self._draw_camera()
            # self._draw_link_COM(whole_body=False)
            self._draw_feet_at_edge()
            # self._draw_foothold()

            # self._draw_height_field(draw_guidance=True)
            # self._draw_edge()

        if self.cfg.sensors.activated:
            depth_img = self.sensors.get('depth_0', get_depth=True)[self.lookat_id].cpu().numpy()
            depth_img = (depth_img - self.cfg.sensors.depth_0.near_clip) / self.cfg.sensors.depth_0.far_clip
            img = np.clip(depth_img * 255, 0, 255).astype(np.uint8)

            cv2.imshow("depth_processed", cv2.resize(img, (530, 300)))
            cv2.waitKey(1)

        super().render()

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

    def _reward_tracking_goal(self):
        if self.sim.terrain is None:
            return self._zero_tensor(self.num_envs)

        root_lin_vel_xy = self.sim.root_lin_vel[:, :2]
        root_lin_vel_xy_norm = torch.norm(root_lin_vel_xy, dim=1, keepdim=True)
        root_lin_vel_unit_vec = torch.where(
            root_lin_vel_xy_norm > self.cfg.commands.lin_vel_clip,
            root_lin_vel_xy / (root_lin_vel_xy_norm + 1e-5),
            0.
        )

        target_unit_vec = self.target_pos_rel / (torch.norm(self.target_pos_rel, dim=1, keepdim=True) + 1e-5)

        rew = torch.sum(root_lin_vel_unit_vec * target_unit_vec, dim=1)
        return rew

    def _reward_stall(self):
        lin_vel_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        lin_vel_cmd_xy = torch.norm(self.commands[:, :2], dim=1)

        lin_vel_stall = (lin_vel_xy < self.cfg.commands.lin_vel_clip) & (lin_vel_cmd_xy > 0)
        ang_vel_stall = (torch.abs(self.base_ang_vel[:, 2]) < self.cfg.commands.ang_vel_clip) & (torch.abs(self.commands[:, 2]) > 0)

        return (lin_vel_stall | ang_vel_stall).float()

    def _reward_timeout(self):
        time_out = torch.clamp(self.tracking_goal_timer - TIMEOUT_TIME, min=0)
        rew = time_out * self.target_pos_rel.norm(dim=1).clip(max=3.) / 3.

        rew[self.env_class < 2] = 0.
        return rew
