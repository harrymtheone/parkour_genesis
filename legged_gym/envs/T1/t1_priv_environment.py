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


class T1PrivEnvironment(HumanoidEnv):

    def _init_buffers(self):
        super()._init_buffers()

        env_cfg = self.cfg.env
        self.priv_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

        self.phase_enabled = self._zero_tensor(self.num_envs, dtype=torch.bool)
        self.phase_enabled[:] = self.cfg.env.enable_clock_input

    def _init_robot_props(self):
        super()._init_robot_props()
        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

    def update_reward_curriculum(self, epoch):
        super().update_reward_curriculum(epoch)

        enabled_ratio = self.linear_change(1., 0., 3000, 3000, epoch)
        num_enabled = int(self.num_envs * enabled_ratio)
        self.phase_enabled[:num_enabled] = True
        self.phase_enabled[num_enabled:] = False

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
        clock[~self.phase_enabled] = 0.
        command_input = torch.cat((clock, self.commands[:, :3] * self.commands_scale), dim=1)

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

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self._draw_goals()
            # self._draw_height_field(draw_guidance=True)
            # self._draw_edge()
            # self._draw_camera()
            self._draw_feet_at_edge()

        super().render()

    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        diff = self.sim.dof_pos - torch.where(
            self.is_zero_command.unsqueeze(1),
            self.init_state_dof_pos,
            self.ref_dof_pos
        )
        diff = diff[:, self.dof_activated]

        rew = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)

        rew[self.env_class >= 2] *= 0.1  # stair
        rew[~self.phase_enabled] = 0.
        return rew

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        rew = torch.where(self.contact_filt == self._get_stance_mask(), 1, -0.3)

        rew[self.env_class >= 2] *= 0.1  # stair
        rew[~self.phase_enabled] = 0.
        return torch.mean(rew, dim=1)

    def _reward_feet_clearance(self):
        # encourage the robot to lift its legs when it moves
        rew_no_phase = (self.feet_height / self.cfg.rewards.feet_height_target).clip(min=-1, max=1)
        rew_phase = torch.where(self._get_stance_mask(), -torch.abs(rew_no_phase), rew_no_phase)

        rew = torch.where(self.phase_enabled.unsqueeze(1), rew_phase, rew_no_phase)
        return rew.sum(dim=1)
