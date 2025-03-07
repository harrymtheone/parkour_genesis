import torch

from legged_gym.envs.base.humanoid_env import HumanoidEnv
from ..base.utils import ObsBase, HistoryBuffer


class ActorObs(ObsBase):
    def __init__(self, proprio, prop_his, priv_actor):
        super().__init__()
        self.proprio = proprio.clone()
        self.prop_his = prop_his.clone()
        self.priv_actor = priv_actor.clone()

    def as_obs_next(self):
        # remove unwanted attribute to save CUDA memory
        return ObsNext(self.proprio)


class ObsNext(ObsBase):
    def __init__(self, proprio):
        self.proprio = proprio.clone()


class CriticObs(ObsBase):
    def __init__(self, priv, priv_his, scan):
        super().__init__()
        self.priv = priv.clone()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()

    def to_tensor(self):
        return torch.cat((self.priv_his.flatten(1), self.scan.flatten(1)), dim=1)


class T1GaitEnvironment(HumanoidEnv):

    def _init_buffers(self):
        super()._init_buffers()

        env_cfg = self.cfg.env
        self.prop_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_prop_his, env_cfg.n_proprio, device=self.device)
        self.critic_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

    def _init_robot_props(self):
        super()._init_robot_props()
        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Roll', 'Yaw'], False), False)

        self.waist_dof_indices = self.sim.create_indices(
            self.sim.get_full_names('Waist', False), False)

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

        clock = 0 * torch.stack(self._get_clock_input(), dim=1)
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
            command_input,  # 5D
            (self.sim.dof_pos - self.init_state_dof_pos)[:, self.dof_activated] * self.obs_scales.dof_pos,  # 12D
            self.sim.dof_vel[:, self.dof_activated] * self.obs_scales.dof_vel,  # 12D
            self.last_action_output,  # 12D
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler * self.obs_scales.quat,  # 3
            self.ext_force[:, :2],  # 2
            self.ext_torque,  # 3
            self.sim.friction_coeffs,  # 1
            self.sim.payload_masses / 10.,  # 1
            self.sim.contact_forces[:, self.feet_indices, 2] > 5.,  # 2
        ), dim=-1)

        priv_actor_obs = self.base_lin_vel * self.obs_scales.lin_vel

        # compute height map
        scan = torch.clip(self.sim.root_pos[:, 2].unsqueeze(1) - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))

        # compose actor observation
        reset_flag = self.episode_length_buf <= 1
        self.prop_his_buf.append(proprio, reset_flag)
        self.actor_obs = ActorObs(proprio, self.prop_his_buf.get(), priv_actor_obs)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # compose critic observation
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs = CriticObs(priv_obs, self.critic_his_buf.get(), scan)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self._draw_goals()
            # self._draw_height_field(draw_guidance=True)
            # self._draw_edge()
            # self._draw_camera()
            self._draw_feet_at_edge()

        super().render()

    # ----------------------------------------- Rewards -------------------------------------------

    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
    #     rew = torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)
    #     return rew
    #
    # def _reward_tracking_ang_vel(self):
    #     diff = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-diff * self.cfg.rewards.tracking_sigma)
    #
    # def _reward_orientation(self):
    #     rew = (self.projected_gravity[:, :2]).square().sum(dim=1)
    #     return rew
    #
    # def _reward_energy(self):
    #     power = (self.torques * self.sim.dof_vel)[:, self.dof_activated]
    #     rew = power.square().sum(dim=1)
    #     return rew
    #
    # def _reward_dof_vel(self):
    #     rew = self.sim.dof_vel[:, self.dof_activated].square().sum(dim=1)
    #     return rew
    #
    # def _reward_dof_acc(self):
    #     dof_acc = (self.last_dof_vel - self.sim.dof_vel) / self.dt
    #     rew = dof_acc[:, self.dof_activated].square().sum(dim=1)
    #     return rew
    #
    # def _reward_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(self.sim.dof_pos - self.sim.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
    #     out_of_limits += (self.sim.dof_pos - self.sim.dof_pos_limits[:, 1]).clip(min=0.)
    #     return torch.sum(out_of_limits, dim=1)
    #
    # def _reward_torques(self):
    #     weighted_torques = self.torques / self.p_gains
    #     rew = weighted_torques[:, self.dof_activated].square().sum(dim=1)
    #     return rew
    #
    # def _reward_contact_forces(self):
    #     contact_forces = torch.norm(self.sim.contact_forces[:, self.feet_indices], dim=-1)
    #     return (contact_forces - self.cfg.rewards.max_contact_force).clip(min=0).sum(dim=1)
    #
    # def _reward_action_rate(self):
    #     rew = (self.actions - self.last_actions).square().sum(dim=1)
    #     return rew
    #
    # def _reward_arm_dof_err(self):
    #     raise NotImplementedError
    #
    # def _reward_waist_dof_err(self):
    #     rew = (self.sim.dof_pos - self.init_state_dof_pos)[:, self.waist_dof_indices]
    #     return rew.square().sum(dim=1)
    #
    # def _reward_leg_yaw_roll(self):
    #     assert self.yaw_roll_dof_indices is not None
    #     yaw_roll = (self.sim.dof_pos - self.init_state_dof_pos)[:, self.yaw_roll_dof_indices]
    #     return yaw_roll.square().sum(dim=1)
    #
    # def _reward_feet_away(self):
    #     foot_pos = self.sim.link_pos[:, self.feet_indices, :2]
    #     foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
    #     rew = foot_dist.clip(max=self.cfg.rewards.min_dist)
    #     return rew

    # ----------------------------------------- Rewards -------------------------------------------

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        rew = torch.where(
            self.commands[:, 0] > self.cfg.commands.lin_vel_clip,
            torch.sum(target_vec_norm * self.sim.root_lin_vel[:, :2], dim=1) / (self.commands[:, 0]),
            0  # no reward for zero_command
        )
        rew = torch.clip(rew, max=1)

        rew[self.env_class < 4] = 0.
        return rew

    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.base_euler[:, 2]))
        return rew

    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        return rew

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.sim.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1. * (torch.norm(self.sim.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        delta_torques = (self.torques - self.last_torques)[:, self.dof_activated]
        return delta_torques.square().sum(dim=1)

    def _reward_torques(self):
        return self.torques[:, self.dof_activated].square().sum(dim=1)

    def _reward_dof_error(self):
        dof_error = (self.sim.dof_pos - self.init_state_dof_pos)[:, self.dof_activated]
        return dof_error.square().sum(dim=1)

    def _reward_feet_stumble(self):
        raise NotImplementedError
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                        4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        raise NotImplementedError
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew

    def _reward_contact_forces(self):
        contact_forces = torch.norm(self.sim.contact_forces[:, self.feet_indices], dim=-1)
        return (contact_forces - self.cfg.rewards.max_contact_force).clip(min=0).sum(dim=1)
