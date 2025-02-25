import torch

from legged_gym.envs.base.humanoid_env import HumanoidEnv
from ..base.utils import ObsBase, HistoryBuffer


class ActorObs(ObsBase):
    def __init__(self, prop_his, scan):
        super().__init__()
        self.prop_his = prop_his.clone()
        self.scan = scan.clone()


class CriticObs(ObsBase):
    def __init__(self, priv_his, scan):
        super().__init__()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()

    def to_tensor(self):
        return torch.cat((self.priv_his.flatten(1), self.scan.flatten(1)), dim=1)


class PddScanEnvironment(HumanoidEnv):

    def _init_buffers(self):
        super()._init_buffers()

        env_cfg = self.cfg.env
        self.prop_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_prop_his, env_cfg.n_proprio, device=self.device)
        self.critic_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

    def _compute_ref_state(self):
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()

        self.ref_dof_pos[:] = 0.
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1

        # left swing
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1

        # right swing
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 7] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 9] = -sin_pos_r * scale_1

        # Add double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.

        self.ref_dof_pos[:] += self.init_state_dof_pos

    def _compute_observations(self):
        """
        Computes observations
        """

        # add lag to sensor observation
        if self.cfg.domain_rand.add_dof_lag:
            dof_data = self.dof_lag_buf.get()
            dof_pos, dof_vel = dof_data[..., 0], dof_data[..., 1]
        else:
            dof_pos, dof_vel = self.dof_pos, self.dof_vel

        if self.cfg.domain_rand.add_imu_lag:
            imu_data = self.imu_lag_buf.get()
            base_quat, base_lin_vel, base_ang_vel = imu_data[..., :4], imu_data[..., 4:7], imu_data[..., 7:]
            projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        else:
            base_ang_vel = self.base_ang_vel
            projected_gravity = self.projected_gravity

        # add noise to sensor observation
        base_ang_vel = self._add_noise(base_ang_vel, self.cfg.noise.noise_scales.ang_vel)
        projected_gravity = self._add_noise(projected_gravity, self.cfg.noise.noise_scales.gravity)
        dof_pos = self._add_noise(dof_pos, self.cfg.noise.noise_scales.dof_pos)
        dof_vel = self._add_noise(dof_vel, self.cfg.noise.noise_scales.dof_vel)

        self._compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)
        command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

        # proprio observation
        proprio = torch.cat((
            base_ang_vel * self.obs_scales.ang_vel,  # 3
            projected_gravity,  # 3
            command_input,  # 5
            (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 10D
            dof_vel * self.obs_scales.dof_vel,  # 10D
            self.last_action_output,  # 10D
        ), dim=-1)

        # explicit privileged information
        priv_obs = torch.cat((
            command_input,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 10D
            self.dof_vel * self.obs_scales.dof_vel,  # 10D
            self.last_action_output,  # 10D
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler * self.obs_scales.quat,  # 3
            # self.rand_push_force[:, :2],  # 2
            # self.rand_push_torque,  # 3
            # self.env_frictions,  # 1
            # self.body_mass / 10.,  # 1
            self._zero_tensor(self.num_envs, 7),
            self.contact_forces[:, self.feet_indices, 2] > 5.,  # 2
        ), dim=-1)

        # compute height map
        scan = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))

        # compose actor observation
        reset_flag = self.episode_length_buf <= 1
        self.prop_his_buf.append(proprio, reset_flag)
        self.actor_obs = ActorObs(self.prop_his_buf.get(), scan)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # compose critic observation
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs = CriticObs(self.critic_his_buf.get(), scan)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)
