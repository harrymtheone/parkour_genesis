import cv2
import numpy as np
import torch

from legged_gym.envs.base.quadruped_env import QuadrupedEnv
from ..base.utils import ObsBase, HistoryBuffer
from ...utils.math import transform_by_yaw


class WorldModelObs(ObsBase):
    def __init__(self, proprio, depth):
        self.proprio = proprio[:, :-12].clone()  # no last_actions
        self.depth = depth.clone()


class ActorObs(ObsBase):
    def __init__(self, proprio, prop_his):
        self.proprio = proprio.clone()
        self.prop_his = prop_his.clone()

    def as_obs_next(self):
        # remove unwanted attribute to save CUDA memory
        return ObsNext(self.proprio)


class ObsNext(ObsBase):
    def __init__(self, proprio):
        self.proprio = proprio.clone()


class CriticObs(ObsBase):
    def __init__(self, priv_his, scan, base_edge_mask):
        super().__init__()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.base_edge_mask = base_edge_mask.clone()


class Go1WMPEnvironment(QuadrupedEnv):

    def _init_buffers(self):
        super()._init_buffers()
        env_cfg = self.cfg.env

        self.prop_his_buf = HistoryBuffer(env_cfg.num_envs, env_cfg.len_prop_his, env_cfg.n_proprio, device=self.device)
        self.critic_his_buf = HistoryBuffer(env_cfg.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

        self.body_hmap_points = self._init_height_points(self.cfg.terrain.body_pts_x, self.cfg.terrain.body_pts_y)
        self.feet_hmap_points = self._init_height_points(self.cfg.terrain.feet_pts_x, self.cfg.terrain.feet_pts_y)

    def get_feet_hmap(self):
        feet_pos = self.sim.link_pos[:, self.feet_indices]

        # convert height points coordinate to world frame
        n_points = self.feet_hmap_points.size(1)
        points = transform_by_yaw(self.feet_hmap_points, self.base_euler[:, 2:3].repeat(1, n_points)).unflatten(0, (self.num_envs, -1))
        points = feet_pos[:, :, None, :] + points[:, None, :, :]

        hmap = self._get_heights(points.flatten(1, 2) + self.cfg.terrain.border_size)
        hmap = feet_pos[..., 2:3] - hmap.unflatten(1, (len(self.feet_indices), -1))  # to relative height
        return hmap.flatten(1, 2)

    def get_body_hmap(self):
        # convert height points coordinate to world frame
        n_points = self.body_hmap_points.size(1)
        points = transform_by_yaw(self.body_hmap_points, self.base_euler[:, 2:3].repeat(1, n_points)).unflatten(0, (self.num_envs, -1))
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

        # proprio observation
        proprio = torch.cat((
            base_ang_vel * self.obs_scales.ang_vel,  # 3
            projected_gravity,  # 3
            self.commands[:, :3] * self.commands_scale,  # 5
            (dof_pos - self.init_state_dof_pos) * self.obs_scales.dof_pos,  # 10D
            dof_vel * self.obs_scales.dof_vel,  # 10D
            self.last_action_output,  # 10D
        ), dim=-1)

        # explicit privileged information
        base_edge_mask = self.get_edge_mask().float()

        priv_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.get_feet_hmap() - self.cfg.normalization.feet_height_correction,  # 16
            self.get_body_hmap() - self.cfg.normalization.scan_norm_bias,  # 16
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler * self.obs_scales.quat,  # 3
            self.commands[:, :3] * self.commands_scale,  # 3D
            self.last_action_output,  # 12D
            (self.sim.dof_pos - self.init_state_dof_pos) * self.obs_scales.dof_pos,  # 12D
            self.sim.dof_vel * self.obs_scales.dof_vel,  # 12D
            self.ext_force[:, :2],  # 2
            self.ext_torque,  # 3
            self.sim.friction_coeffs,  # 1
            self.sim.payload_masses / 10.,  # 1
            self.sim.contact_forces[:, self.feet_indices, 2] > 5.,  # 2
        ), dim=-1)
        priv_actor_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.get_feet_hmap() - self.cfg.normalization.feet_height_correction,  # 16
            self.get_body_hmap() - self.cfg.normalization.scan_norm_bias,  # 16
        ), dim=-1)

        # compute height map
        scan = torch.clip(self.sim.root_pos[:, 2:3] - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))

        # compose actor observation
        actor_obs = ActorObs(proprio, self.prop_his_buf.get())
        actor_obs.clip(self.cfg.normalization.clip_observations)

        # compose world model observation
        world_model_obs = WorldModelObs(proprio, self.sensors.get('depth_0'))
        self.actor_obs = [actor_obs, world_model_obs]

        # update history buffer
        reset_flag = self.episode_length_buf <= 1
        self.prop_his_buf.append(proprio, reset_flag)

        # compose critic observation
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs = CriticObs(self.critic_his_buf.get(), scan, base_edge_mask)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self._draw_goals()
            # self._draw_height_field(draw_guidance=True)
            # self._draw_edge()
            # self._draw_camera()
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
            #     indices = torch.randperm(len(pts))[:200]
            #     self.sim.draw_points(pts[indices])

        super().render()
