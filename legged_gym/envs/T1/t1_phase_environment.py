import cv2
import numpy as np
import torch

from .t1_base_env import T1BaseEnv, mirror_proprio_by_x, mirror_dof_prop_by_x
from ..base.utils import ObsBase
from ...utils.math import transform_by_yaw, torch_rand_float


class ActorObs(ObsBase):
    def __init__(self, proprio, prop_his, depth, scan):
        super().__init__()
        self.proprio = proprio.clone()
        self.prop_his = prop_his.clone()
        self.depth = depth.clone()
        self.scan = scan.clone()

    def as_obs_next(self):
        # remove unwanted attribute to save CUDA memory
        return ObsNext(self.proprio)

    @torch.compiler.disable
    def mirror(self):
        return ActorObs(
            mirror_proprio_by_x(self.proprio),
            mirror_proprio_by_x(self.prop_his.flatten(0, 1)).view(self.prop_his.shape),
            torch.flip(self.depth, dims=[3]),
            torch.flip(self.scan, dims=[2]),
        )

    @staticmethod
    def mirror_dof_prop_by_x(dof_prop: torch.Tensor):
        dof_prop_mirrored = dof_prop.clone()
        mirror_dof_prop_by_x(dof_prop_mirrored, 0)
        return dof_prop_mirrored


class ActorObsNoDepth(ObsBase):
    def __init__(self, proprio, prop_his, scan, priv_actor):
        super().__init__()
        self.proprio = proprio.clone()
        self.prop_his = prop_his.clone()
        self.scan = scan.clone()
        self.priv_actor = priv_actor.clone()

    def as_obs_next(self):
        return ObsNext(self.proprio)

    @torch.compiler.disable
    def mirror(self):
        priv_actor_mirrored = self.priv_actor.clone()
        priv_actor_mirrored[:, 1] *= -1.0

        return ActorObsNoDepth(
            mirror_proprio_by_x(self.proprio),
            mirror_proprio_by_x(self.prop_his.flatten(0, 1)).view(self.prop_his.shape),
            torch.flip(self.scan, dims=[2]),
            priv_actor_mirrored
        )

    @staticmethod
    def mirror_dof_prop_by_x(dof_prop: torch.Tensor):
        dof_prop_mirrored = dof_prop.clone()
        mirror_dof_prop_by_x(dof_prop_mirrored, 0)
        return dof_prop_mirrored


class ObsNext(ObsBase):
    def __init__(self, proprio):
        self.proprio = proprio.clone()


class CriticObs(ObsBase):
    def __init__(self, priv_his, scan, edge_mask):
        super().__init__()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()


class T1_Phase_Environment(T1BaseEnv):

    def step(self, action_clock):
        if self.cfg.policy.enable_modulator:
            self.last_phase_increment_ratio[:] = self.phase_increment_ratio
            self.last_phase_bias[:] = self.phase_bias

            self.phase_increment_ratio[:] = 1 + 0.5 * action_clock[:, -3]
            self.phase_bias[:] = 0.5 * action_clock[:, -2:]

            self.phase_increment_ratio[:] = self.phase_increment_ratio.clip(min=self.ratio_range[0], max=self.ratio_range[1])
            self.phase_bias[:] = self.phase_bias.clip(min=self.bias_range[0], max=self.bias_range[1])

            return super().step(action_clock[:, :-3])
        else:
            return super().step(action_clock)

    def _init_buffers(self):
        super()._init_buffers()

        self.phase_increment_ratio = 1 + self._zero_tensor(self.num_envs)
        self.phase_bias = self._zero_tensor(self.num_envs, 2)

        # self.ratio_range = (0.9, 1.1)
        # self.bias_range = (-0.1, 0.1)

        self.ratio_range = (0.7, 1.5)
        self.bias_range = (-0.3, 0.3)

        self.last_phase_increment_ratio = torch.zeros_like(self.phase_increment_ratio)
        self.last_phase_bias = torch.zeros_like(self.phase_bias)

    def _init_robot_props(self):
        super()._init_robot_props()

        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

    def _resample_commands(self, env_ids: torch.Tensor):
        super()._resample_commands(env_ids)

        if not self.cfg.policy.enable_modulator:
            self.phase_increment_ratio[env_ids] = torch_rand_float(self.ratio_range[0], self.ratio_range[1], (len(env_ids), 1), self.device).squeeze()

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        if not self.cfg.policy.enable_modulator:
            self.phase_bias[env_ids] = torch_rand_float(0, 1., (len(env_ids), 1), self.device)
            self.phase_bias[env_ids, 1:2] += torch_rand_float(self.bias_range[0], self.bias_range[1], (len(env_ids), 1), self.device)

    def update_reward_curriculum(self, epoch):
        super().update_reward_curriculum(epoch)

        if epoch < 3000:
            self.ratio_range = (0.9, 1.1)
            self.bias_range = (-0.1, 0.1)
        elif epoch < 6000:
            self.ratio_range = (0.8, 1.3)
            self.bias_range = (-0.2, 0.2)
        elif epoch < 1000000:
            self.ratio_range = (0.7, 1.5)
            self.bias_range = (-0.3, 0.3)

    def _post_physics_pre_step(self):
        super()._post_physics_pre_step()

        self.phase_length_buf[:] += self.dt * (self.phase_increment_ratio - 1)
        self._update_phase()

    def _get_clock_input(self):
        clock_l = torch.sin(2 * torch.pi * (self.phase + self.phase_bias[:, 0]))
        clock_r = torch.sin(2 * torch.pi * (self.phase - 0.5 + self.phase_bias[:, 1]))
        return clock_l, clock_r

    def _reward_default_clock(self):
        rew = torch.square(self.phase_increment_ratio - 1)
        rew += torch.square(self.phase_bias).sum(dim=1)
        rew += torch.square(self.phase_bias[:, 0] - self.phase_bias[:, 1])
        return rew

    def _reward_clock_smoothness(self):
        return torch.square(self.phase_increment_ratio - self.last_phase_increment_ratio) + torch.square(self.phase_bias - self.last_phase_bias).sum(dim=1)

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

        priv_actor_obs = self.base_lin_vel * self.obs_scales.lin_vel

        # compute height map
        scan = torch.clip(self.sim.root_pos[:, 2:3] - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))
        base_edge_mask = self.get_edge_mask().float().view((self.num_envs, *self.cfg.env.scan_shape))
        scan_edge = torch.stack([scan, base_edge_mask], dim=1)

        if self.cfg.sensors.activated:
            self.actor_obs = ActorObs(proprio, self.prop_his_buf.get(), self.sensors.get('depth_0').squeeze(2), scan_edge)
        else:
            self.actor_obs = ActorObsNoDepth(proprio, self.prop_his_buf.get(), scan_edge, priv_actor_obs)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # update history buffer
        reset_flag = self.episode_length_buf <= 1
        prop_no_cmd = proprio.clone()
        prop_no_cmd[:, 6: 6 + 5] = 0.
        self.prop_his_buf.append(prop_no_cmd, reset_flag)

        # compose critic observation
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
            self._draw_foothold()

            # self._draw_height_field(draw_guidance=False)
            # self._draw_edge()

        if self.cfg.sensors.activated:
            depth_img = self.sensors.get('depth_0', get_depth=True)[self.lookat_id].cpu().numpy()
            depth_img = (depth_img - self.cfg.sensors.depth_0.near_clip) / self.cfg.sensors.depth_0.far_clip
            img = np.clip(depth_img * 255, 0, 255).astype(np.uint8)

            cv2.imshow("depth_processed", cv2.resize(img, (530, 300)))
            cv2.waitKey(1)

        super().render()

    def draw_cloud_from_depth(self):
        # draw points cloud
        cloud, cloud_valid = self.sensors.get('depth_0', get_cloud=True)
        cloud, cloud_valid = cloud[self.lookat_id], cloud_valid[self.lookat_id]
        pts = cloud[cloud_valid].cpu().numpy()

        if len(pts) > 0:
            pts = density_weighted_sampling(pts, 500)
            self.sim.draw_points(pts)

    def draw_est_hmap(self, est):
        feet_hmap = est[self.lookat_id, -16 - 8:-16]
        self._draw_feet_hmap(feet_hmap)

        body_hmap = est[self.lookat_id, -16:]
        self._draw_body_hmap(body_hmap)

    def _draw_feet_hmap(self, estimation=None):
        num_feet = len(self.feet_indices)
        num_feet_pts = self.feet_hmap_points.size(1)

        # compute points position
        height_points = transform_by_yaw(
            self.feet_hmap_points[None, self.lookat_id, :, :].repeat(num_feet, 1, 1),
            self.feet_euler_xyz[self.lookat_id, :, None, 2].repeat(1, num_feet_pts)
        ).view(num_feet, num_feet_pts, 3)

        height_points = height_points + self.sim.link_pos[self.lookat_id, self.feet_indices, None, :]

        # compute points height
        if estimation is None:
            hmap = self.get_feet_hmap()[self.lookat_id]
        else:
            hmap = estimation + self.cfg.normalization.feet_height_correction

        height_points[:, :, 2] -= hmap.view(num_feet, -1)

        height_points = height_points.flatten(0, 1).cpu().numpy()
        self.pending_vis_task.append(dict(points=height_points, color=(1, 1, 0)))

    def _draw_body_hmap(self, estimation=None):
        # compute points position
        height_points = transform_by_yaw(
            self.body_hmap_points[self.lookat_id],
            self.base_euler[self.lookat_id, 2].repeat(self.body_hmap_points.size(1))
        )

        height_points[:] += self.sim.root_pos[self.lookat_id, None, :]

        # compute points height
        if estimation is None:
            hmap = self.get_body_hmap()[self.lookat_id]
        else:
            hmap = estimation + self.cfg.normalization.scan_norm_bias

        height_points[:, 2] -= hmap

        height_points = height_points.cpu().numpy()
        self.pending_vis_task.append(dict(points=height_points))


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
    sampled_indices = np.random.choice(len(points), size=num_samples, replace=False, p=probabilities)

    return points[sampled_indices]
