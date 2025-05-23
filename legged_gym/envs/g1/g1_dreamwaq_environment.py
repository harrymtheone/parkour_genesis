import torch

from .g1_base_env import T1BaseEnv, mirror_proprio_by_x, mirror_dof_prop_by_x
from ..base.utils import ObsBase, HistoryBuffer, CircularBuffer
from ...utils.math import torch_rand_float


class ActorObs(ObsBase):
    def __init__(self, proprio, prop_his, priv_actor):
        super().__init__()
        self.proprio = proprio.clone()
        self.prop_his = prop_his.clone()
        self.priv_actor = priv_actor.clone()

    def as_obs_next(self):
        # remove unwanted attribute to save CUDA memory
        return ObsNext(self.proprio)

    @torch.compiler.disable
    def mirror(self):
        priv_actor = self.priv_actor.clone()
        priv_actor[:, 1] *= -1.

        return ActorObs(
            mirror_proprio_by_x(self.proprio),
            mirror_proprio_by_x(self.prop_his.flatten(0, 1)).unflatten(0, self.prop_his.shape[:2]),
            priv_actor,
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
    def __init__(self, priv, priv_his, scan):
        super().__init__()
        self.priv = priv.clone()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()

    def to_tensor(self):
        return torch.cat((self.priv_his.flatten(1), self.scan.flatten(1)), dim=1)


class G1DreamWaqEnvironment(T1BaseEnv):
    def step(self, actions):
        self.last_action_output[:] = actions
        self.actions_his_buf.append(actions)

        action_fft = torch.fft.fft(self.actions_his_buf.get_all(), dim=0)
        freqs = torch.fft.fftfreq(action_fft.size(0), d=self.dt).to(self.device)
        mask = torch.abs(freqs) <= 10
        fft_filtered = action_fft * mask[:, None, None]
        act_his_filtered = torch.real(torch.fft.ifft(fft_filtered, dim=0))
        self.actions_filtered_his_buf.append(act_his_filtered[0])

        # actions = actions.clone()
        # actions[:, 5] = act_his_filtered[0, :, 5]
        # actions[:, 6] = act_his_filtered[0, :, 6]
        # actions[:, 11] = act_his_filtered[0, :, 11]
        # actions[:, 12] = act_his_filtered[0, :, 12]

        # clip action range
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions[:, self.dof_activated] = torch.clip(actions, -clip_actions, clip_actions)

        # step the simulator
        self._step_environment()
        self._post_physics_step()

        return self.actor_obs, self.critic_obs, self.rew_buf.clone(), self.reset_buf.clone(), self.extras

    def _init_buffers(self):
        super()._init_buffers()

        env_cfg = self.cfg.env
        self.prop_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_prop_his, env_cfg.n_proprio, device=self.device)
        self.critic_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

        self.phase_increment_ratio = torch_rand_float(0.8, 1.2, (self.num_envs, 1), self.device).squeeze(1)

        self.actions_his_buf = CircularBuffer(50, self.num_envs, (self.num_actions,), self.device)
        self.actions_filtered_his_buf = CircularBuffer(50, self.num_envs, (self.num_actions,), self.device)

    def _init_robot_props(self):
        super()._init_robot_props()
        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

    def _post_physics_pre_step(self):
        super()._post_physics_pre_step()

        self.phase_length_buf[:] += self.dt * (self.phase_increment_ratio - 1)
        self._update_phase()

    def _reward_low_pass_filter(self):
        action_his = self.actions_his_buf.get_all()

        magnitude = torch.abs(torch.fft.fft(action_his, dim=0))
        freqs = torch.fft.fftfreq(action_his.size(0), d=self.dt).to(self.device)

        high_freq_mask = torch.abs(freqs) > 20.
        high_freq_sum = (magnitude * high_freq_mask[:, None, None]).sum(dim=0)

        return high_freq_sum.mean(dim=1)

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
        self.actor_obs = ActorObs(proprio, self.prop_his_buf.get(), priv_actor_obs)
        self.prop_his_buf.append(proprio, reset_flag)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # compose critic observation
        self.critic_obs = CriticObs(priv_obs, self.critic_his_buf.get(), scan)
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)
