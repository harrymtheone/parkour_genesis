import cv2
import numpy as np
import torch

from legged_gym.envs.base.humanoid_env import HumanoidEnv
from ..base.utils import ObsBase, HistoryBuffer
from ...utils.math import transform_by_yaw


class ActorObs(ObsBase):
    def __init__(self, proprio, prop_his, depth, priv_actor, scan):
        super().__init__()
        self.proprio = proprio.clone()
        self.prop_his = prop_his.clone()
        self.depth = depth.clone()
        self.priv_actor = priv_actor.clone()
        self.scan = scan.clone()

    def as_obs_next(self):
        # remove unwanted attribute to save CUDA memory
        return ObsNext(self.proprio)


class ObsNext(ObsBase):
    def __init__(self, proprio):
        self.proprio = proprio.clone()


class CriticObs(ObsBase):
    def __init__(self, priv_his, scan):
        super().__init__()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()


class T1GaitEnvironment(HumanoidEnv):

    def _init_robot_props(self):
        super()._init_robot_props()
        self.hip_indices = self.sim.create_indices(
            self.sim.get_full_names("Hip", True), True)

    def _init_buffers(self):
        super()._init_buffers()
        env_cfg = self.cfg.env
        self.prop_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_prop_his, env_cfg.n_proprio, device=self.device)
        self.critic_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

        self.body_hmap_points = self._init_height_points(self.cfg.terrain.body_pts_x, self.cfg.terrain.body_pts_y)
        self.feet_hmap_points = self._init_height_points(self.cfg.terrain.feet_pts_x, self.cfg.terrain.feet_pts_y)

    def get_feet_hmap(self):
        feet_pos = self.sim.link_pos[:, self.feet_indices]

        # convert height points coordinate to world frame
        n_points = self.feet_hmap_points.size(1)
        points = transform_by_yaw(self.feet_hmap_points, self.base_euler[:, 2].repeat(1, n_points)).unflatten(0, (self.num_envs, -1))
        points = feet_pos[:, :, None, :] + points[:, None, :, :]

        hmap = self._get_heights(points.flatten(1, 2) + self.cfg.terrain.border_size)
        hmap = feet_pos[..., 2:3] - hmap.unflatten(1, (len(self.feet_indices), -1))  # to relative height
        return hmap.flatten(1, 2)

    def get_body_hmap(self):
        # convert height points coordinate to world frame
        n_points = self.body_hmap_points.size(1)
        points = transform_by_yaw(self.body_hmap_points, self.base_euler[:, 2].repeat(1, n_points)).unflatten(0, (self.num_envs, -1))
        points = self.sim.root_pos[:, None, :] + points

        hmap = self._get_heights(points + self.cfg.terrain.border_size)
        return self.sim.root_pos[:, 2:3] - hmap  # to relative height

    def _compute_observations(self):
        """
        Computes observations
        """
        # add lag to sensor observation
        if self.cfg.domain_rand.add_dof_lag:
            dof_data = self.dof_lag_buf.get()
            dof_pos, dof_vel = dof_data[..., 0], dof_data[..., 1]
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

        sin_pos = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)
        command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        # proprio observation
        proprio = torch.cat((
            base_ang_vel * self.obs_scales.ang_vel,  # 3
            projected_gravity,  # 3
            command_input,  # 5
            (dof_pos[:, self.dof_activated] - self.init_state_dof_pos[:, self.dof_activated]) * self.obs_scales.dof_pos,  # 12D
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
            (self.sim.dof_pos[:, self.dof_activated] - self.init_state_dof_pos[:, self.dof_activated]) * self.obs_scales.dof_pos,  # 12D
            self.sim.dof_vel[:, self.dof_activated] * self.obs_scales.dof_vel,  # 12D
            self.ext_force[:, :2],  # 2
            self.ext_torque,  # 3
            self.sim.friction_coeffs,  # 1
            self.sim.payload_masses / 10.,  # 1
            self.sim.contact_forces[:, self.feet_indices, 2] > 5.,  # 2
        ), dim=-1)

        priv_actor_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.get_feet_hmap() - self.cfg.normalization.feet_height_correction,  # 8
            self.get_body_hmap() - self.cfg.normalization.scan_norm_bias,  # 16
        ), dim=-1)

        # compute height map
        scan = torch.clip(self.sim.root_pos[:, 2].unsqueeze(1) - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))

        # compose actor observation
        self.actor_obs = ActorObs(proprio, self.prop_his_buf.get(), self.sensors.get('depth_0').squeeze(2), priv_actor_obs, scan)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # update history buffer
        reset_flag = self.episode_length_buf <= 1
        self.prop_his_buf.append(proprio, reset_flag)

        # compose critic observation
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs = CriticObs(self.critic_his_buf.get(), scan)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self._draw_goals()
            # self._draw_height_field()
            # self._draw_edge()
            self._draw_camera()
            self._draw_feet_at_edge()

        if self.cfg.sensors.activated:
            depth_img = self.sensors.get('depth_0')
            depth_img = depth_img[self.lookat_id, 0].cpu().numpy()

            img = np.clip((depth_img + 0.5) * 255, 0, 255).astype(np.uint8)
            # img = np.clip(depth_img / self.cfg.sensors.depth_0.far_clip * 255, 0, 255).astype(np.uint8)

            cv2.imshow("depth_processed", cv2.resize(img, (530, 300)))
            cv2.waitKey(1)

            # # draw points cloud
            # cloud, cloud_valid = self.sensors.get('depth_0', get_cloud=True)
            # cloud, cloud_valid = cloud[self.lookat_id], cloud_valid[self.lookat_id]
            # pts = cloud[cloud_valid]
            #
            # if len(pts) > 0:
            #     indices = torch.randperm(len(pts))[:400]
            #     self.sim.draw_points(pts[indices], color=(1, 0, 0))

        super().render()

    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])

        rew[torch.logical_and(self.env_class >= 4, self.env_class < 12)] *= 0.5
        return rew

    def _reward_ang_vel_xy(self):
        rew = torch.square(self.base_ang_vel[:, :2])

        rew *= torch.tensor([[1, 3]], device=self.device)
        rew = torch.sum(rew, dim=1)

        rew[torch.logical_and(self.env_class >= 4, self.env_class < 12)] *= 0.3
        return rew

    def _reward_feet_clearance(self):
        # from unitree
        pos_error = torch.square(self.sim.link_pos[:, self.feet_indices, 2] - self.cfg.rewards.feet_height_target) * ~self.contact_filt
        return torch.sum(pos_error, dim=1)

    def _reward_action_rate(self):
        rew = (self.actions - self.last_actions).square().sum(dim=1)
        rew[torch.logical_and(self.env_class >= 4, self.env_class < 13)] *= 0.5
        return rew

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.sim.dof_pos - self.sim.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.sim.dof_pos - self.sim.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    @staticmethod
    def _reward_alive():
        # Reward for staying alive
        return 1.0

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.sim.dof_pos[:, self.hip_indices]), dim=1)
