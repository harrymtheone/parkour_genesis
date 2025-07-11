import cv2
import numpy as np
import torch

from .t1_base_env import T1BaseEnv, density_weighted_sampling
from ..base.utils import ObsBase
from ...utils.math import torch_rand_float, transform_by_yaw


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


class T1OdomEnvironment(T1BaseEnv):

    def _init_robot_props(self):
        super()._init_robot_props()

        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

        self.head_link_indices = self.sim.create_indices(
            self.sim.get_full_names(['H2'], True), True)

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

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            # self._draw_body_hmap()
            # self.draw_hmap_from_depth()
            # self.draw_cloud_from_depth()
            self._draw_goals()
            # self._draw_camera()
            # self._draw_link_COM(whole_body=False)
            self._draw_feet_at_edge()
            self._draw_foothold()

            # self._draw_height_field(draw_guidance=True)
            # self._draw_edge()

        if self.cfg.sensors.activated:
            depth_img = self.sensors.get('depth_0', get_depth=True)[self.lookat_id].cpu().numpy()
            depth_img = (depth_img - self.cfg.sensors.depth_0.near_clip) / self.cfg.sensors.depth_0.far_clip
            img = np.clip(depth_img * 255, 0, 255).astype(np.uint8)

            cv2.imshow("depth_processed", cv2.resize(img, (320, 320)))
            cv2.waitKey(1)

        # self.draw_cloud_from_depth()
        super().render()

    def draw_cloud_from_depth(self):
        # draw points cloud
        cloud, cloud_valid = self.sensors.get('depth_0', get_cloud=True)
        cloud, cloud_valid = cloud[self.lookat_id], cloud_valid[self.lookat_id]
        pts = cloud[cloud_valid].cpu().numpy()

        if len(pts) > 0:
            pts = density_weighted_sampling(pts, 500)
            self.pending_vis_task.append(dict(points=pts))

    def _reward_goal_dist_change(self, vel_thresh=0.6):
        if self.sim.terrain is None:
            return self._zero_tensor(self.num_envs)

        dist_change = self.last_goal_distance - self.goal_distance
        rew = dist_change.clip(max=vel_thresh * self.dt)
        rew *= torch.clip(1 - 2 * torch.abs(self.delta_yaw) / torch.pi, min=0.)
        rew *= (self.env_class >= 100) & ~self.is_zero_command & (dist_change > -1.)  # dist increase a lot in a sudden, meaning goal updated
        return rew

    def _reward_penalize_vy(self):
        rew = torch.abs(self.base_lin_vel[:, 1])
        rew[self.env_class < 100] = 0.
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
