from collections import deque

import cv2
import numpy as np
import torch

from legged_gym.envs.T1.t1_base_env import T1BaseEnv
from legged_gym.envs.base.utils import ObsBase
from legged_gym.utils.math import quat_rotate_inverse


class ActorObs(ObsBase):
    def __init__(self, proprio, depth):
        super().__init__()
        self.proprio = proprio.clone()
        self.depth = depth.clone()

    def as_obs_next(self):
        # remove unwanted attribute to save CUDA memory
        return ObsNext(self.proprio)

    def mirror(self):
        return type(self)(
            mirror_proprio_by_x(self.proprio),
            mirror_proprio_by_x(self.prop_his.flatten(0, 1)).view(self.prop_his.shape),
            torch.flip(self.depth, dims=[3]),
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
    def __init__(self, priv_his, scan, edge_mask, est_gt):
        super().__init__()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()
        self.est_gt = est_gt.clone()


class T1_PIE_Env(T1BaseEnv):

    def _init_robot_props(self):
        super()._init_robot_props()

        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

    def _init_buffers(self):
        super()._init_buffers()

        self.gen_amp_history = deque(maxlen=self.cfg.amp.amp_obs_hist_steps)
        self.amp_obs_hist_steps = self.cfg.amp.amp_obs_hist_steps
        for _ in range(self.cfg.amp.amp_obs_hist_steps):
            self.gen_amp_history.append(torch.zeros(
                self.num_envs, self.cfg.amp.num_single_amp_obs, dtype=torch.float, device=self.device))
        self.reset_amp_obs = None
        self.reset_idx = None

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

        # proprio observation
        proprio = torch.cat((
            base_ang_vel * self.obs_scales.ang_vel,  # 3
            projected_gravity,  # 3
            self.commands[:, :3] * self.commands_scale,
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
            self.commands[:, :3] * self.commands_scale,
            self.last_action_output,  # 12D
            (self.sim.dof_pos - self.init_state_dof_pos)[:, self.dof_activated] * self.obs_scales.dof_pos,  # 12D
            self.sim.dof_vel[:, self.dof_activated] * self.obs_scales.dof_vel,  # 12D
            self.ext_force[:, :2],  # 2
            self.ext_torque,  # 3
            self.sim.payload_masses / 10.,  # 1
            self.sim.contact_forces[:, self.feet_indices, 2] > 5.,  # 2
        ), dim=-1)

        est_gt = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
        ), dim=-1)

        # compute height map
        edge_mask = -0.5 + self.get_edge_mask().float().view(self.num_envs, *self.cfg.env.scan_shape)

        # depth = torch.cat([self.sensors.get('depth_0'), self.sensors.get('depth_1')], dim=1)
        depth = self.sensors.get('depth_0')
        if self.cfg.algorithm.use_amp:
            depth = depth.half()

        # compose actor observation
        self.actor_obs = ActorObs(proprio, depth)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # update history buffer
        reset_flag = self.episode_length_buf <= 1
        self.prop_his_buf.append(proprio, reset_flag)

        scan = torch.clip(self.sim.root_pos[:, 2:3] - self.get_scan(noisy=False) - self.base_height.unsqueeze(1), -1, 1.)
        scan = scan.view(self.num_envs, *self.cfg.env.scan_shape)

        # compose critic observation
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs = CriticObs(self.critic_his_buf.get(), scan, edge_mask, est_gt)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self._draw_goals()
            self._draw_feet_hmap()
            # self._draw_height_field()
            # self._draw_edge()
            # self._draw_camera()
            # self._draw_feet_at_edge()

        if self.cfg.sensors.activated:
            depth_img_f = self.sensors.get('depth_0', get_depth=True)[self.lookat_id].cpu().numpy()
            depth_img_f = (depth_img_f - self.cfg.sensors.depth_0.near_clip) / self.cfg.sensors.depth_0.far_clip
            img_f = np.clip(depth_img_f * 255, 0, 255).astype(np.uint8)
            cv2.imshow("depth_front", cv2.resize(img_f, (320, 320)))
            cv2.waitKey(1)

            # depth_img_r = self.sensors.get('depth_1', get_depth=True)[self.lookat_id].cpu().numpy()
            # depth_img_r = (depth_img_r - self.cfg.sensors.depth_1.near_clip) / self.cfg.sensors.depth_1.far_clip
            # img_r = np.clip(depth_img_r * 255, 0, 255).astype(np.uint8)
            # cv2.imshow("depth_back", cv2.resize(img_r, (320, 320)))
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

    def _reward_default_dof_pos(self):
        return (self.sim.dof_pos - self.init_state_dof_pos).abs().sum(dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)


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

    # commands [6:9], [x, -y, -yaw]
    prop[:, 7:9] *= -1.

    # dof pos
    mirror_dof_prop_by_x(prop, 9)

    # dof vel
    mirror_dof_prop_by_x(prop, 9 + 13)

    # last actions
    mirror_dof_prop_by_x(prop, 9 + 13 + 13)

    return prop


@torch.jit.script
def mirror_priv_by_x(priv: torch.Tensor) -> torch.Tensor:
    priv = priv.clone()

    # linear velocity, [0:3], [x, -y, z]
    priv[:, 1] *= -1.

    return priv
