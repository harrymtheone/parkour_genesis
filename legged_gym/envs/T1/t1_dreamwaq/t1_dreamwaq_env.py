import torch

from legged_gym.envs.T1.t1_base_env import T1BaseEnv
from legged_gym.envs.base.utils import ObsBase


class ActorObs(ObsBase):
    def __init__(self, proprio):
        super().__init__()
        self.proprio = proprio.clone()

    def as_obs_next(self):
        # remove unwanted attribute to save CUDA memory
        return ActorObs(self.proprio)

    # @torch.compiler.disable
    # def mirror(self):
    #     priv_actor = self.priv_actor.clone()
    #     priv_actor[:, 1] *= -1.
    #
    #     return ActorObs(
    #         mirror_proprio_by_x(self.proprio),
    #         mirror_proprio_by_x(self.prop_his.flatten(0, 1)).unflatten(0, self.prop_his.shape[:2]),
    #         priv_actor,
    #     )
    #
    # @staticmethod
    # def mirror_dof_prop_by_x(dof_prop: torch.Tensor):
    #     dof_prop_mirrored = dof_prop.clone()
    #     mirror_dof_prop_by_x(dof_prop_mirrored, 0)
    #     return dof_prop_mirrored


class CriticObs(ObsBase):
    def __init__(self, priv_his, scan, edge_mask, est_gt):
        super().__init__()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()
        self.est_gt = est_gt.clone()


class T1DreamWaqEnv(T1BaseEnv):
    def _init_robot_props(self):
        super()._init_robot_props()
        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

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

        # compose actor observation
        reset_flag = self.episode_length_buf <= 1
        self.actor_obs = ActorObs(proprio)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)

        # compute height map
        scan = torch.clip(self.sim.root_pos[:, 2].unsqueeze(1) - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape))
        edge_mask = -0.5 + self.get_edge_mask().float().view(self.num_envs, *self.cfg.env.scan_shape)

        est_gt = self.base_lin_vel * self.obs_scales.lin_vel

        # compose critic observation
        self.critic_obs = CriticObs(self.critic_his_buf.get(), scan, edge_mask, est_gt)
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)
