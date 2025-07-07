import cv2
import numpy as np
import torch

from .t1_base_env import T1BaseEnv, mirror_proprio_by_x, mirror_dof_prop_by_x
from ..base.utils import ObsBase


def linear_change(start, end, span, start_it, cur_it):
    cur_value = start + (end - start) * (cur_it - start_it) / span
    cur_value = max(cur_value, min(start, end))
    cur_value = min(cur_value, max(start, end))
    return cur_value


class ActorObs(ObsBase):
    def __init__(self, proprio, prop_his, depth, priv_actor):
        super().__init__()
        self.proprio = proprio.clone()
        self.prop_his = prop_his.clone()
        self.depth = depth.clone()
        self.priv_actor = priv_actor.clone()

    def as_obs_next(self):
        # remove unwanted attribute to save CUDA memory
        return ObsNext(self.proprio)

    def mirror(self):
        return type(self)(
            mirror_proprio_by_x(self.proprio),
            mirror_proprio_by_x(self.prop_his.flatten(0, 1)).view(self.prop_his.shape),
            torch.flip(self.depth, dims=[3]),
            torch.flip(self.scan, dims=[2]),
            torch.flip(self.edge_mask, dims=[2]),
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
    def __init__(self, priv, priv_his, scan, edge_mask):
        super().__init__()
        self.priv = priv.clone()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()


class T1PIEEnvironment(T1BaseEnv):
    def _init_robot_props(self):
        super()._init_robot_props()
        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

    def _init_buffers(self):
        super()._init_buffers()
        self.goal_task_timer = self._zero_tensor(self.num_envs)

    def _post_physics_mid_step(self):
        super()._post_physics_mid_step()

        self.goal_task_timer[self.reached_goal_env] = 0
        timer_increase = ~self.reached_goal_env & ~self.is_zero_command
        self.goal_task_timer[timer_increase] += 1 * self.commands[timer_increase, 0] / self.cfg.commands.parkour_ranges.lin_vel_x[1]

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
            (dof_pos - self.init_state_dof_pos)[:, self.dof_activated] * self.obs_scales.dof_pos,  # 13
            dof_vel[:, self.dof_activated] * self.obs_scales.dof_vel,  # 13
            self.last_action_output,  # 13
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

        priv_actor = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
        ), dim=-1)

        # compute height map
        scan = torch.clip(self.sim.root_pos[:, 2].unsqueeze(1) - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))
        edge_mask = self.get_edge_mask().float()

        # compose actor observation
        self.actor_obs = ActorObs(proprio, self.prop_his_buf.get(), self.sensors.get('depth_0').squeeze(2), priv_actor)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # update history buffer
        reset_flag = self.episode_length_buf <= 1
        self.prop_his_buf.append(proprio, reset_flag)

        # compose critic observation
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs = CriticObs(priv_obs, self.critic_his_buf.get(), scan, edge_mask)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self._draw_goals()
            # self._draw_height_field()
            # self._draw_edge()
            # self._draw_camera()
            # self._draw_feet_at_edge()

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

    def _reward_timeout(self):
        time_out = self.goal_task_timer - 100
        effective_out = torch.clamp(time_out, min=0)
        time_out_rew = effective_out * 0.001
        time_out_rew[self.env_class < 100] = 0.
        return time_out_rew

    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        diff = self.sim.dof_pos - torch.where(
            self.is_zero_command.unsqueeze(1),
            self.init_state_dof_pos,
            self.ref_dof_pos
        )
        diff = torch.norm(diff[:, self.dof_activated], dim=1)

        # diff = self.sim.dof_pos - self.ref_dof_pos
        # diff = torch.norm(diff[:, self.dof_activated], dim=1)
        # diff[self.is_zero_command] = 0.

        rew = torch.exp(-diff * 2) - 0.2 * diff.clamp(0, 0.5)
        rew[self.env_class >= 2] *= 0.1
        return rew

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        rew = torch.where(self.contact_filt == self._get_stance_mask(), 1, -0.3)
        rew[self.env_class >= 2] *= 0.1
        return torch.mean(rew, dim=1)

    # def _reward_lin_vel_z(self):
    #     # Penalize z axis base linear velocity
    #     return torch.square(self.base_lin_vel[:, 2])
    #
    # def _reward_ang_vel_xy(self):
    #     # Penalize xy axes base angular velocity
    #     return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    #
    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    #
    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     return torch.square(self.base_height - self.cfg.rewards.base_height_target)
    #
    # def _reward_feet_clearance(self):
    #     # encourage the robot to lift its legs when it moves
    #     x = self.feet_height.clip(min=self.cfg.rewards.feet_height_target, max=self.cfg.rewards.feet_height_target_max)
    #     pos_err = torch.square(self.feet_height - x)
    #     return torch.sum(pos_err, dim=1) / self.cfg.rewards.feet_height_target
    #
    # def _reward_feet_distance(self):
    #     feet_pos = self.sim.link_pos[:, self.feet_indices, :2]
    #     feet_dist = torch.norm(feet_pos[:, 0, :] - feet_pos[:, 1, :], dim=1)
    #
    #     x = feet_dist.clip(min=self.cfg.rewards.min_dist, max=self.cfg.rewards.max_dist)
    #     pos_err = torch.square(feet_dist - x) / self.cfg.rewards.min_dist
    #     return torch.sum(pos_err)
    #
    # def _reward_knee_distance(self):
    #     knee_pos = self.sim.link_pos[:, self.knee_indices, :2]
    #     knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
    #
    #     x = knee_dist.clip(min=self.cfg.rewards.min_dist, max=self.cfg.rewards.max_dist / 2)
    #     pos_err = torch.square(knee_dist - x) / self.cfg.rewards.min_dist
    #     return torch.sum(pos_err)
    #
    # def _reward_feet_rotation(self):
    #     return torch.sum(self.feet_euler_xyz[..., :2].square(), dim=[1, 2])
    #
    # @staticmethod
    # def _reward_alive():
    #     return 1.
