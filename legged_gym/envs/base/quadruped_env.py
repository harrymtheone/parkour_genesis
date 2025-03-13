import torch

from legged_gym.envs.base.parkour_task import ParkourTask


class QuadrupedEnv(ParkourTask):
    def get_observations(self):
        return self.actor_obs

    def get_critic_observations(self):
        return self.critic_obs

    def _init_robot_props(self):
        super()._init_robot_props()

        self.base_index = self.sim.create_indices(
            self.sim.get_full_names(self.cfg.asset.base_link_name, True), True)
        self.feet_indices = self.sim.create_indices(
            self.sim.get_full_names(self.cfg.asset.foot_name, True), True)

        self.hip_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(self.cfg.asset.hip_dof_name, False), False)
        self.thigh_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(self.cfg.asset.thigh_dof_name, False), False)
        self.calf_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(self.cfg.asset.calf_dof_name, False), False)

    def _init_buffers(self):
        super()._init_buffers()

        self.feet_air_time = self._zero_tensor(self.num_envs, self.feet_indices.size(0))
        self.feet_air_time_avg = self._zero_tensor(self.num_envs, self.feet_indices.shape[0]) + 0.1
        self.contact_filt = self._zero_tensor(self.num_envs, len(self.feet_indices), dtype=torch.bool)
        self.contact_forces_avg = self._zero_tensor(self.num_envs, len(self.feet_indices))

        self.last_contacts = self._zero_tensor(self.num_envs, len(self.feet_indices), dtype=torch.bool)
        self.last_feet_vel_xy = self._zero_tensor(self.num_envs, len(self.feet_indices), 2)

        self.feet_height = self._zero_tensor(self.num_envs, len(self.feet_indices))

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        self.last_feet_vel_xy[env_ids] = 0.

    def _refresh_variables(self):
        super()._refresh_variables()

        contact = torch.norm(self.sim.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt[:] = contact | self.last_contacts

        # update feet height
        feet_pos = self.sim.link_pos[:, self.feet_indices]
        proj_ground_height = self._get_heights(feet_pos + self.cfg.terrain.border_size, use_guidance=self.cfg.rewards.use_guidance_terrain)
        self.feet_height[:] = feet_pos[:, :, 2] + self.cfg.normalization.feet_height_correction - proj_ground_height

    def _check_termination(self):
        super()._check_termination()
        roll_cutoff = torch.abs(self.base_euler[:, 0]) > 1.57
        pitch_cutoff = torch.abs(self.base_euler[:, 1]) > 1.57
        self.reset_buf[:] |= roll_cutoff
        self.reset_buf[:] |= pitch_cutoff

    def _post_physics_pre_step(self):
        super()._post_physics_pre_step()

        alpha = self.cfg.rewards.EMA_update_alpha
        self.contact_forces_avg[self.contact_filt] = alpha * self.contact_forces_avg[self.contact_filt]
        self.contact_forces_avg[self.contact_filt] += (1 - alpha) * self.sim.contact_forces[:, self.feet_indices][self.contact_filt][:, 2]

        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time[self.contact_filt | self.is_zero_command.unsqueeze(1)] += self.dt
        self.feet_air_time_avg[first_contact] = alpha * self.feet_air_time_avg[first_contact] + (1 - alpha) * self.feet_air_time[first_contact]

    def _post_physics_post_step(self):
        super()._post_physics_post_step()
        self.feet_air_time[:] *= ~(self.contact_filt | self.is_zero_command.unsqueeze(1))
        self.last_feet_vel_xy[:] = self.sim.link_vel[:, self.feet_indices, :2]
        self.last_contacts[:] = torch.norm(self.sim.contact_forces[:, self.feet_indices], dim=-1) > 2.

    # ----------------------------------------- Rewards -------------------------------------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

        rew[torch.logical_and(self.env_class >= 4, self.env_class < 12)] *= 0.3
        return rew

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-6)
        rew = torch.sum(target_vec_norm * self.sim.root_lin_vel[:, :2], dim=1) / (self.commands[:, 0] + 1e-6)
        # no reward for zero_command and envs of non-parkour terrain
        rew[(self.commands[:, 0] < self.cfg.commands.lin_vel_clip) | (self.env_class < 4)] = 0.
        return rew.clip(max=1)

    def _reward_tracking_yaw(self):
        diff = self.commands[:, 2] - self.base_ang_vel[:, 2]
        return torch.exp(-torch.square(diff) / self.cfg.rewards.tracking_sigma)

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

    def _reward_orientation(self):
        rew = torch.square(self.projected_gravity[:, :2])
        rew *= torch.tensor([[1, 3]], device=self.device)
        rew = torch.sum(rew, dim=1)

        rew[torch.logical_and(self.env_class >= 2, self.env_class < 13)] *= 0.3
        return rew

    def _reward_torques(self):
        rew = torch.sum(torch.square(self.torques), dim=1)
        rew[torch.logical_and(self.env_class >= 4, self.env_class < 13)] *= 0.3
        return rew

    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.torques - self.last_torques), dim=1)
        rew[torch.logical_and(self.env_class >= 4, self.env_class < 13)] *= 0.3
        return rew

    def _reward_action_rate(self):
        rew = (self.actions - self.last_actions).square().sum(dim=1)
        rew[torch.logical_and(self.env_class >= 4, self.env_class < 13)] *= 0.5
        return rew

    def _reward_action_smoothness(self):
        return (self.actions + self.last_last_actions - 2 * self.last_actions).square().sum(dim=1)

    def _reward_dof_acc(self):
        rew = torch.sum(torch.square((self.last_dof_vel - self.sim.dof_vel) / self.dt), dim=1)
        rew[torch.logical_and(self.env_class >= 4, self.env_class < 13)] *= 0.3
        return rew

    def _reward_dof_error(self):
        return torch.sum(torch.square(self.sim.dof_pos - self.init_state_dof_pos), dim=1)

    def _reward_hip_pos(self):
        diff = torch.square(self.sim.dof_pos[:, self.hip_dof_indices] - self.init_state_dof_pos[:, self.hip_dof_indices])
        rew = torch.sum(diff, dim=1)
        return rew

    def _reward_thigh_pos(self):
        diff = torch.square(self.sim.dof_pos[:, self.thigh_dof_indices] - self.init_state_dof_pos[:, self.thigh_dof_indices])
        rew = torch.sum(diff, dim=1)
        return rew

    def _reward_feet_accel(self):
        feet_vel_xy = self.sim.link_vel[:, self.feet_indices]
        feet_accel_xy = (feet_vel_xy - self.last_feet_vel_xy) / self.dt
        rew = torch.exp(-0.02 * feet_accel_xy.norm(dim=2).sum(dim=1))
        return rew

    def _reward_feet_air_time(self):
        # Reward long steps
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        rew = self.feet_air_time * first_contact  # reward only on first contact with the ground

        # no reward for zero command
        rew *= ~self.is_zero_command.unsqueeze(1)

        # no reward for legs with smaller feet_air_time_avg
        mask_equalization = torch.empty_like(self.feet_air_time_avg)
        mask_equalization[:, 0] = torch.where(self.feet_air_time_avg[:, 0] < self.feet_air_time_avg[:, 1], 1, 0)
        mask_equalization[:, 1] = torch.where(self.feet_air_time_avg[:, 1] < self.feet_air_time_avg[:, 0], 1, 0)
        mask_equalization[:, 2] = torch.where(self.feet_air_time_avg[:, 2] < self.feet_air_time_avg[:, 3], 1, 0)
        mask_equalization[:, 3] = torch.where(self.feet_air_time_avg[:, 3] < self.feet_air_time_avg[:, 2], 1, 0)
        rew *= mask_equalization

        return torch.sum(rew, dim=1, dtype=torch.float)

    # def _reward_feet_clearance(self):
    #     """
    #     Calculates reward based on the clearance of the swing leg from the ground during movement.
    #     Encourages appropriate lift of the feet during the swing phase of the gait.
    #     """
    #     # feet height should larger than target feet height at the peak
    #     rew = torch.clip(self.feet_height / self.cfg.rewards.feet_height_target, 0, 1)
    #
    #     # revert reward for zero command
    #     rew[self.is_zero_command] = torch.where(torch.all(self.contact_filt[self.is_zero_command]),
    #                                             0,
    #                                             -rew[self.is_zero_command] - 1.0)
    #
    #     # no reward for legs with smaller feet_air_time_avg
    #     mask_equalization = torch.empty_like(self.feet_air_time_avg)
    #     mask_equalization[:, 0] = torch.where(self.feet_air_time_avg[:, 0] < self.feet_air_time_avg[:, 1], 1, 0)
    #     mask_equalization[:, 1] = torch.where(self.feet_air_time_avg[:, 1] < self.feet_air_time_avg[:, 0], 1, 0)
    #     mask_equalization[:, 2] = torch.where(self.feet_air_time_avg[:, 2] < self.feet_air_time_avg[:, 3], 1, 0)
    #     mask_equalization[:, 3] = torch.where(self.feet_air_time_avg[:, 3] < self.feet_air_time_avg[:, 2], 1, 0)
    #     rew *= mask_equalization
    #
    #     return rew.sum(dim=1, dtype=torch.float)

    def _reward_feet_clearance(self):
        env_only_yaw = torch.logical_and(
            torch.norm(self.commands[:, :2], dim=1) < self.cfg.commands.lin_vel_clip,
            torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip
        )

        feet_vel_xy = torch.norm(self.sim.link_vel[:, self.feet_indices], dim=2)

        # encourage the robot to lift its legs when it moves, especially rotates in place
        rew = (self.feet_height - self.cfg.rewards.feet_height_target).square()

        # the robot do not need to lift too much when you walk slowly
        rew[~env_only_yaw] *= feet_vel_xy[~env_only_yaw]

        return rew.sum(dim=1, dtype=torch.float)

    def _reward_base_stumble(self):
        base_stumble = torch.norm(self.sim.contact_forces[:, self.base_index].squeeze(1), dim=1) > 1
        rew = base_stumble.float()
        return rew

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        force = self.sim.contact_forces[:, self.feet_indices]
        rew = torch.any(torch.norm(force[:, :, :2], dim=2) > 4 * torch.abs(force[:, :, 2]), dim=1).float()
        return rew

    def _reward_feet_edge(self):
        feet_pos_xy = self.sim.link_pos[:, self.feet_indices, :2] + self.cfg.terrain.border_size  # (num_envs, 4, 2)
        feet_pos_xy = (feet_pos_xy / self.cfg.terrain.horizontal_scale).round().long()
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.sim.edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.sim.edge_mask.shape[1] - 1)
        feet_at_edge = self.sim.edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = torch.sum(self.feet_at_edge.float(), dim=-1)
        return rew

    def _reward_collision(self):
        return torch.sum(1. * (torch.norm(self.sim.contact_forces[:, self.penalised_contact_indices], dim=-1) > 0.1), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.sim.dof_pos - self.sim.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.sim.dof_pos - self.sim.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        rew = torch.square(self.base_height - self.cfg.rewards.base_height_target)
        return rew.clip(-1, 1)

    def _reward_termination(self):
        raise NotImplementedError
