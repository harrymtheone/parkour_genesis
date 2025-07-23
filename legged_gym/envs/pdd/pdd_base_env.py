import torch

from legged_gym.envs.base.humanoid_env import HumanoidEnv
from legged_gym.envs.base.utils import HistoryBuffer
from legged_gym.utils.math import transform_by_yaw


class PddBaseEnvironment(HumanoidEnv):
    def _init_robot_props(self):
        super()._init_robot_props()
        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['1_joint', '2_joint'], False), False)

    def _init_buffers(self):
        super()._init_buffers()

        env_cfg = self.cfg.env
        self.prop_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_prop_his, env_cfg.n_proprio, device=self.device)
        self.critic_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

        self.body_hmap_points = self._init_height_points(self.cfg.terrain.body_pts_x, self.cfg.terrain.body_pts_y)
        self.feet_hmap_points = self._init_height_points(self.cfg.terrain.feet_pts_x, self.cfg.terrain.feet_pts_y)

    def _compute_ref_state(self):
        ref_dof_pos = self._zero_tensor(self.num_envs, self.num_actions)
        scale_1 = self.cfg.commands.target_joint_pos_scale
        scale_2 = 2 * scale_1

        phase = torch.stack(self._get_clock_input(wrap_sin=False), dim=1)
        swing_ratio = self.cfg.commands.air_ratio
        delta_t = self.cfg.commands.delta_t

        phase_swing = torch.clip((phase - delta_t / 2) / (swing_ratio - delta_t), min=0., max=1.)
        clock = torch.sin(torch.pi * phase_swing)

        # left motion
        ref_dof_pos[:, 2] = -clock[:, 0] * scale_1
        ref_dof_pos[:, 3] = clock[:, 0] * scale_2
        ref_dof_pos[:, 4] = -clock[:, 0] * scale_1

        # right motion
        ref_dof_pos[:, 7] = -clock[:, 1] * scale_1
        ref_dof_pos[:, 8] = clock[:, 1] * scale_2
        ref_dof_pos[:, 9] = -clock[:, 1] * scale_1

        self.ref_dof_pos[:] = self.init_state_dof_pos
        self.ref_dof_pos[:, self.dof_activated] += ref_dof_pos

    def get_feet_hmap(self):
        feet_pos = self.sim.link_pos[:, self.feet_indices]

        # convert height points coordinate to world frame
        n_points = self.feet_hmap_points.size(1)
        points = transform_by_yaw(self.feet_hmap_points, self.base_euler[:, 2:3].repeat(1, n_points)).unflatten(0, (self.num_envs, -1))
        points = feet_pos[:, :, None, :] + points[:, None, :, :]

        hmap = self._get_heights(points.flatten(1, 2) + self.cfg.terrain.border_size)
        hmap = feet_pos[..., 2:3] - hmap.unflatten(1, (len(self.feet_indices), -1))  # to relative height
        return hmap.flatten(1, 2)

    def get_body_hmap(self):
        # convert height points coordinate to world frame
        n_points = self.body_hmap_points.size(1)
        points = transform_by_yaw(self.body_hmap_points, self.base_euler[:, 2:3].repeat(1, n_points)).unflatten(0, (self.num_envs, -1))
        points = self.sim.root_pos[:, None, :] + points

        hmap = self._get_heights(points + self.cfg.terrain.border_size)
        return self.sim.root_pos[:, 2:3] - hmap  # to relative height
