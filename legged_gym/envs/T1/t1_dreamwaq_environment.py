import torch

from ..base.humanoid_env import HumanoidEnv
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


class T1DreamWaqEnvironment(HumanoidEnv):

    def _init_buffers(self):
        super()._init_buffers()

        env_cfg = self.cfg.env
        self.prop_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_prop_his, env_cfg.n_proprio, device=self.device)
        self.critic_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

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
