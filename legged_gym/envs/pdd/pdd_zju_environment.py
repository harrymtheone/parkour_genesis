import torch

from .pdd_base_env import PddBaseEnvironment
from ..base.utils import ObsBase, HistoryBuffer


class ActorObs(ObsBase):
    def __init__(self, proprio, prop_his, depth, priv, scan, recon_prev=None):
        super().__init__()
        self.proprio = proprio.clone()
        self.prop_his = prop_his.clone()
        self.depth = depth.clone()
        self.priv = priv.clone()
        self.scan = scan.clone()
        self.recon_prev = recon_prev

    def as_obs_next(self):
        # remove unwanted attribute to save CUDA memory
        return ObsNext(self.proprio)


class ObsNext(ObsBase):
    def __init__(self, proprio):
        self.proprio = proprio.clone()


class CriticObs(ObsBase):
    def __init__(self, priv_his, scan):
        super().__init__()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()


class PddZJUEnvironment(PddBaseEnvironment):

    def _init_buffers(self):
        super()._init_buffers()
        env_cfg = self.cfg.env

        self.actor_obs = None
        self.critic_obs = None

        self.prop_his_buf = HistoryBuffer(env_cfg.num_envs, env_cfg.len_prop_his, env_cfg.n_proprio, dtype=torch.float16, device=self.device)
        self.critic_his_buf = HistoryBuffer(env_cfg.num_envs, env_cfg.len_critic_his, env_cfg.n_priv, dtype=torch.float16, device=self.device)

        self.body_hmap_points = self._init_height_points(self.cfg.terrain.body_pts_x, self.cfg.terrain.body_pts_y)
        self.feet_hmap_points = self._init_height_points(self.cfg.terrain.feet_pts_x, self.cfg.terrain.feet_pts_y)

    def get_feet_hmap(self):
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]

        # convert height points coordinate to world frame
        points = quat_apply_yaw(self.base_quat.repeat(1, self.feet_hmap_points.shape[1]), self.feet_hmap_points)
        points = feet_pos[:, :, None, :] + points[:, None, :, :]

        hmap = self._get_heights(points.flatten(1, 2) + self.terrain.cfg.border_size)
        hmap = feet_pos[..., 2:3] - hmap.unflatten(1, (len(self.feet_indices), -1))  # to relative height
        return hmap.flatten(1, 2)

    def get_body_hmap(self):
        # convert height points coordinate to world frame
        points = quat_apply_yaw(self.base_quat.repeat(1, self.body_hmap_points.shape[1]), self.body_hmap_points)
        points += (self.root_states[:, :3]).unsqueeze(1)

        hmap = self._get_heights(points + self.terrain.cfg.border_size)
        return self.root_states[:, 2:3] - hmap  # to relative height

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
        ), dim=-1).to(torch.float16)

        # explicit privileged information
        priv_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.get_feet_hmap(),  # 8
            self.get_body_hmap() - self.cfg.normalization.scan_norm_bias,  # 16

            command_input,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 10D
            self.dof_vel * self.obs_scales.dof_vel,  # 10D
            self.last_action_output,  # 10D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            # self.rand_push_force[:, :2],  # 2
            # self.rand_push_torque,  # 3
            # self.env_frictions,  # 1
            # self.body_mass / 10.,  # 1
            self._zero_tensor(self.num_envs, 7),
            self.contact_forces[:, self.feet_indices, 2] > 5.,  # 2
        ), dim=-1).to(torch.float16)

        # update depth buffer
        reset_flag = self.episode_length_buf <= 1
        self.depth_buf.step(reset_flag)

        # compute height map
        scan = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.scan_hmap - self.cfg.normalization.scan_norm_bias, -1, 1.)
        scan = scan.view((self.num_envs, *self.cfg.env.scan_shape)).to(torch.float16)

        # compose actor observation
        self.actor_obs = ActorObs(proprio, self.prop_his_buf.get(), self.depth_buf.get().squeeze(2), priv_obs, scan)
        self.actor_obs.clip(self.cfg.normalization.clip_observations)
        # update history buffer
        self.prop_his_buf.append(proprio, reset_flag)

        # compose critic observation
        self.critic_his_buf.append(priv_obs, reset_flag)
        self.critic_obs = CriticObs(self.critic_his_buf.get(), scan)
        self.critic_obs.clip(self.cfg.normalization.clip_observations)

