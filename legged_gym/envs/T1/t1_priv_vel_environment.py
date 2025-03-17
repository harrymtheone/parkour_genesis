import torch

from ..base.humanoid_env import HumanoidEnv
from ..base.utils import ObsBase, HistoryBuffer


class ActorObs(ObsBase):
    def __init__(self, priv, priv_his, scan, edge_mask):
        super().__init__()
        self.priv = priv.clone()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()


class PredictorObs(ObsBase):
    def __init__(self, actor_input):
        super().__init__()
        self.actor_input = actor_input.clone()


class T1PrivVelEnvironment(HumanoidEnv):

    def _init_buffers(self):
        super()._init_buffers()

        env_cfg = self.cfg.env
        self.priv_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

        self.commands_corrected = torch.zeros_like(self.commands)

    def _init_robot_props(self):
        super()._init_robot_props()
        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

    def _compute_ref_state(self):
        clock_l, clock_r = self._get_clock_input()

        ref_dof_pos = self._zero_tensor(self.num_envs, self.num_actions)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1

        # left swing
        clock_l[clock_l > 0] = 0
        ref_dof_pos[:, 1] = clock_l * scale_1
        ref_dof_pos[:, 4] = -clock_l * scale_2
        ref_dof_pos[:, 5] = clock_l * scale_1

        # right swing
        clock_r[clock_r > 0] = 0
        ref_dof_pos[:, 7] = clock_r * scale_1
        ref_dof_pos[:, 10] = -clock_r * scale_2
        ref_dof_pos[:, 11] = clock_r * scale_1

        # # Add double support phase
        # ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.

        self.ref_dof_pos[:] = self.init_state_dof_pos
        self.ref_dof_pos[:, self.dof_activated] += ref_dof_pos

    def _compute_observations(self):
        """
        Computes observations
        """
        self._compute_ref_state()

        clock = torch.stack(self._get_clock_input(), dim=1)
        command_input = torch.cat((clock, self.commands_corrected[:, :3] * self.commands_scale), dim=1)

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

        # compute height map
        scan = torch.clip(self.sim.root_pos[:, 2].unsqueeze(1) - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))

        reset_flag = self.episode_length_buf <= 1
        self.priv_his_buf.append(priv_obs, reset_flag)
        self.actor_obs = ActorObs(priv_obs, self.priv_his_buf.get(), scan, self.get_edge_mask().float())
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        self.critic_obs = self.actor_obs

    def step(self, actions):
        # shape (n_envs, 1)
        actions, self.vel_correction = actions
        self.commands_corrected[:] = self.commands
        self.commands_corrected[:, 0:1] += self.vel_correction

        return super().step(actions)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()

        # reward episode sums
        self.episode_sums['rew_vel_predictor'] = self._zero_tensor(self.num_envs)

    def _compute_reward(self):
        super()._compute_reward()

        rew_predictor = torch.zeros_like(self.rew_buf)
        rew_predictor[:] += self._reward_vel_correction() * 1. * self.dt
        rew_predictor[:] += self._reward_feet_edge() * self.reward_scales['feet_edge'] * self.dt
        # rew_predictor[:] += self._reward_feet_stumble() * self.reward_scales['feet_stumble'] * self.dt
        self.extras['rew_vel_predictor'] = rew_predictor

        self.episode_sums['rew_vel_predictor'][:] += self._reward_vel_correction() * 0.1 * self.dt

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self._draw_goals()
            # self._draw_height_field(draw_guidance=True)
            # self._draw_edge()
            # self._draw_camera()
            self._draw_feet_at_edge()

        super().render()

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error_abs = torch.sum(torch.abs(self.commands_corrected[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_square = torch.sum(torch.square(self.commands_corrected[:, :2] - self.base_lin_vel[:, :2]), dim=1)

        return torch.where(
            self.is_zero_command,
            torch.exp(-lin_vel_error_abs * self.cfg.rewards.tracking_sigma * 2),
            torch.exp(-lin_vel_error_square * self.cfg.rewards.tracking_sigma)
        )

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands_corrected[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands_corrected[:, 0])

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

        reward[self.is_zero_command] = 0.
        return reward

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(self.commands_corrected[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands_corrected[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        r = (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
        return r

    def _reward_vel_correction(self):
        return torch.exp(-torch.abs(self.vel_correction.squeeze(1)) * 5.0)
