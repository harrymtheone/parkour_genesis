import torch

from .parkour_task import ParkourTask


class SimpleTask(ParkourTask):

    def _resample_commands(self, env_ids: torch.Tensor):
        self.commands[env_ids] = 0
        self.commands[env_ids, 0] = 0.5

    # ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        diff = self.dof_pos - torch.where(
            self.is_zero_command.unsqueeze(1),
            self.default_dof_pos,
            self.ref_dof_pos
        )

        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r[self.env_class == 12] *= 0.1
        # r[torch.norm(self.commands[:, :1], dim=1) > 1.0] = 1.0
        # r[stand_command] = r.clone()[stand_command] * 0.6
        # r[self.is_zero_command] = 1.0
        return r

    def _reward_feet_clearance(self):
        swing_mask = ~self._get_stance_mask()

        # encourage the robot to lift its legs when it moves
        # rew = (self.feet_height > self.cfg.rewards.feet_height_target) * (self.feet_height < self.cfg.rewards.feet_height_target_max)
        rew = torch.clip(self.feet_height / self.cfg.rewards.feet_height_target, 0, 1)
        rew = torch.sum(rew * swing_mask, dim=1, dtype=torch.float)
        return rew

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        return air_time.sum(dim=1)

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact conditions.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_body_states[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_body_states[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        # print(self.contact_forces[:, self.feet_indices, :])
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error_abs = torch.sum(torch.abs(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_square = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)

        rew = torch.where(
            self.is_zero_command,
            torch.exp(-lin_vel_error_abs * self.cfg.rewards.tracking_sigma * 2),
            torch.exp(-lin_vel_error_square * self.cfg.rewards.tracking_sigma)
        )

        return rew

    # def _reward_tracking_lin_vel(self):
    #     """
    #     Tracks linear velocity commands along the xy axes.
    #     Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
    #     """
    #     # stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
    #     lin_vel_error = torch.sum(torch.square(
    #         self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     # lin_vel_error[stand_command] = torch.sum(torch.abs(
    #     #     self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)[stand_command]
    #     r = torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)
    #     # r[stand_command] = r.clone()[stand_command] * 0.6
    #     # r[stand_command] = 1.0
    #     return r

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        ang_vel_error_abs = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_square = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])

        rew = torch.where(
            self.is_zero_command,
            torch.exp(-ang_vel_error_abs * self.cfg.rewards.tracking_sigma * 2),
            torch.exp(-ang_vel_error_square * self.cfg.rewards.tracking_sigma)
        )
        return rew

    # def _reward_tracking_ang_vel(self):
    #     """
    #     Tracks angular velocity commands for yaw rotation.
    #     Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
    #     """
    #     # stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
    #     ang_vel_error = torch.square(
    #         self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     # ang_vel_error[stand_command] = torch.abs(
    #     #     self.commands[:, 2] - self.base_ang_vel[:, 2])[stand_command]
    #     r = torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    #     # r[stand_command] = r.clone()[stand_command] * 0.6
    #     # r[stand_command] = 1.0
    #     return r

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.05)

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        # stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        r = (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
        # r[stand_command] = r.clone()[stand_command] * 0.7
        # r[stand_command] = 1.0
        return r

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 5:7]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_stance_mask()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        # stance_mask = self._get_gait_phase()
        # measured_terrain_heights_around_base = torch.sum(
        #     self.rigid_body_states[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        # base_height = self.root_states[:, 2] - (measured_terrain_heights_around_base - 0.05)
        # return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

        rew = torch.square(self.base_height - self.cfg.rewards.base_height_target)
        return rew.clip(-1, 1)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_stand_still(self):
        # penalize motion at zero commands
        # stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        # not_being_pushed = torch.norm(self.ext_forces, dim=1) < 100
        # stand_command = torch.logical_and(stand_command, not_being_pushed)
        dof_idx = [0, 1, 2, 3, 5, 6, 7, 8]
        dof_default_error = self.dof_pos[:, dof_idx] - self.default_dof_pos[:, dof_idx]
        ankle_dof_error = self.feet_euler_xyz[:, :, 1]
        rew = torch.exp(-dof_default_error.square().sum(dim=1) - ankle_dof_error.square().sum(dim=1))
        rew[self.is_zero_command] = 0
        return rew

    # def _reward_stand_still(self):
    #     # penalize motion at zero commands
    #     stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
    #     r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
    #     r = torch.where(stand_command, r.clone(),
    #                     torch.zeros_like(r))
    #     return r

    def _reward_termination(self):
        # Terminal reward / penalty
        rew = self.reset_buf & ~(self.episode_length_buf > self.max_episode_length)
        return rew.float()

    def _reward_feet_rotation(self):
        # rotation = torch.sum(torch.square(self.feet_euler_xyz[:,:,:2]),dim=[1,2])
        pitch = torch.sum(torch.square(self.feet_euler_xyz[:, :, 1]), dim=1)
        return torch.exp(-(pitch / 1.).square())

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        self.dof_vel_limits[[4, 9]] = 10
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
