import cv2
import numpy as np
import torch

from .t1_base_env import T1BaseEnv, mirror_proprio_by_x, mirror_dof_prop_by_x
from ..base.utils import ObsBase


class ActorObs(ObsBase):
    def __init__(self, proprio, cmd, last_actions, depth):
        super().__init__()
        self.proprio = proprio.clone()
        self.cmd = cmd.clone()
        self.last_actions = last_actions.clone()
        self.depth = depth.clone()

    def get_full_prop(self):
        return torch.cat([self.proprio, self.cmd, self.last_actions], dim=-1)


class ObsNext(ObsBase):
    def __init__(self, proprio):
        self.proprio = proprio.clone()
        self.proprio[:, 6: 6 + 5] = 0.
        self.proprio[:, -12:] = 0.


class CriticObs(ObsBase):
    def __init__(self, priv_his, scan, edge_mask):
        super().__init__()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()


class T1_BBM_Environment(T1BaseEnv):
    def _init_buffers(self):
        super()._init_buffers()

        bounding_box = self.cfg.sensors.depth_0.bounding_box  # x1, x2, y1, y2
        hmap_shape = self.cfg.sensors.depth_0.hmap_shape  # x dim, y dim
        self.depth_scan_points = self._init_height_points(np.linspace(*bounding_box[:2], hmap_shape[0]),
                                                          np.linspace(*bounding_box[2:], hmap_shape[1]))

    def _init_robot_props(self):
        super()._init_robot_props()

        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

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
            (dof_pos - self.init_state_dof_pos)[:, self.dof_activated] * self.obs_scales.dof_pos,  # 13D
            dof_vel[:, self.dof_activated] * self.obs_scales.dof_vel,  # 13D
            # command_input,  # 5
            # self.last_action_output,  # 13D
        ), dim=-1)

        # explicit privileged information
        priv_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.get_feet_hmap() - self.cfg.normalization.feet_height_correction,  # 8
            self.get_body_hmap() - self.cfg.normalization.scan_norm_bias,  # 16
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler * self.obs_scales.quat,  # 3
            command_input,  # 5D
            self.last_action_output,  # 13D
            (self.sim.dof_pos - self.init_state_dof_pos)[:, self.dof_activated] * self.obs_scales.dof_pos,  # 13D
            self.sim.dof_vel[:, self.dof_activated] * self.obs_scales.dof_vel,  # 13D
            self.ext_force[:, :2],  # 2
            self.ext_torque,  # 3
            self.sim.friction_coeffs,  # 1
            self.sim.payload_masses / 10.,  # 1
            self.sim.contact_forces[:, self.feet_indices, 2] > 5.,  # 2
        ), dim=-1)

        # compute height map
        depth = self.sensors.get('depth_0').squeeze(2)
        self.actor_obs = ActorObs(proprio, command_input, self.last_action_output, depth.half())
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # compose critic observation
        scan = torch.clip(self.sim.root_pos[:, 2:3] - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))
        base_edge_mask = self.get_edge_mask().float().view((self.num_envs, *self.cfg.env.scan_shape))

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
            # self._draw_feet_at_edge()
            # self._draw_foothold()

            # self._draw_height_field(draw_guidance=True)
            # self._draw_edge()

        if self.cfg.sensors.activated:
            depth_img = self.sensors.get('depth_0', get_depth=True)[self.lookat_id].cpu().numpy()
            depth_img = (depth_img - self.cfg.sensors.depth_0.near_clip) / self.cfg.sensors.depth_0.far_clip
            img = np.clip(depth_img * 255, 0, 255).astype(np.uint8)

            cv2.imshow("depth_processed", cv2.resize(img, (img.shape[0] * 5, img.shape[1] * 5)))
            cv2.waitKey(1)

        super().render()

def density_weighted_sampling(points, num_samples, k=10):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)

    # Estimate density as the mean distance to k-nearest neighbors
    density = np.mean(distances, axis=1)

    # Higher density -> lower probability of being sampled
    probabilities = density / np.sum(density)
    probabilities = 1 - probabilities  # Invert probabilities for uniformity
    probabilities /= np.sum(probabilities)

    # Sample points based on computed probabilities
    num_samples = min(num_samples, len(points) - 1)
    sampled_indices = np.random.choice(len(points), size=num_samples, replace=False, p=probabilities)

    return points[sampled_indices]
