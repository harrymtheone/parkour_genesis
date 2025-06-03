import torch

from legged_gym.envs.base.humanoid_env import HumanoidEnv
from legged_gym.envs.base.utils import HistoryBuffer
from legged_gym.utils.math import transform_by_yaw


class T1BaseEnv(HumanoidEnv):
    def _init_buffers(self):
        super()._init_buffers()
        env_cfg = self.cfg.env
        self.critic_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

        self.body_hmap_points = self._init_height_points(self.cfg.terrain.body_pts_x, self.cfg.terrain.body_pts_y)
        self.feet_hmap_points = self._init_height_points(self.cfg.terrain.feet_pts_x, self.cfg.terrain.feet_pts_y)

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


@torch.jit.script
def mirror_dof_prop_by_x(prop: torch.Tensor, start_idx: int):
    left_idx = start_idx + torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long, device=prop.device)
    right_idx = left_idx + 6

    dof_left = prop[:, left_idx].clone()
    prop[:, left_idx] = prop[:, right_idx]
    prop[:, right_idx] = dof_left

    prop[:, start_idx + 0] *= -1.  # invert waist

    invert_idx = start_idx + torch.tensor([2, 3, 6], dtype=torch.long, device=prop.device)
    prop[:, invert_idx] *= -1.
    prop[:, invert_idx + 6] *= -1.


@torch.jit.script
def mirror_proprio_by_x(prop: torch.Tensor) -> torch.Tensor:
    prop = prop.clone()

    # base angular velocity, [0:3], [-roll, pitch, -yaw]
    prop[:, 0] *= -1.
    prop[:, 2] *= -1.

    # projected gravity, [3:6], [x, -y, z]
    prop[:, 4] *= -1.

    # commands [6:11], [clock_l <--> clock_r, x, -y, -yaw]
    clock_l = prop[:, 6].clone()
    prop[:, 6] = prop[:, 7]
    prop[:, 7] = clock_l
    prop[:, 9:11] *= -1.

    # dof pos
    mirror_dof_prop_by_x(prop, 11)

    # dof vel
    mirror_dof_prop_by_x(prop, 11 + 13)

    # last actions
    mirror_dof_prop_by_x(prop, 11 + 13 + 13)

    return prop


@torch.jit.script
def mirror_priv_by_x(priv: torch.Tensor) -> torch.Tensor:
    priv = priv.clone()

    # base velocity, [0:3], [x, -y, z]
    priv[:, 1] *= -1.

    # feet height map, [3:19], flip
    feet_hmap = priv[:, 3:19].unflatten(1, (4, 2, 2))
    feet_hmap = torch.flip(feet_hmap, dims=[1, 3])
    priv[:, 3:19] = feet_hmap.flatten(1)

    # body height map, [19:35], flip
    body_hmap = priv[:, 19:35].unflatten(1, (4, 4))
    body_hmap = torch.flip(body_hmap, dims=[2])
    priv[:, 19:35] = body_hmap.flatten(1)

    return priv
