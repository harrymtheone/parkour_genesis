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
