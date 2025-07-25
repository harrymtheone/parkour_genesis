import torch

from legged_gym.envs.base.humanoid_env import HumanoidEnv
from legged_gym.envs.base.utils import HistoryBuffer
from legged_gym.utils.math import transform_by_yaw, density_weighted_sampling


class T1BaseEnv(HumanoidEnv):
    def _init_buffers(self):
        super()._init_buffers()
        env_cfg = self.cfg.env
        if hasattr(env_cfg, 'len_prop_his'):
            self.prop_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_prop_his, env_cfg.n_proprio, device=self.device)
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

    def _draw_feet_hmap(self, estimation=None):
        num_feet = len(self.feet_indices)
        num_feet_pts = self.feet_hmap_points.size(1)

        # compute points position
        height_points = transform_by_yaw(
            self.feet_hmap_points[None, self.lookat_id, :, :].repeat(num_feet, 1, 1),
            self.feet_euler_xyz[self.lookat_id, :, None, 2].repeat(1, num_feet_pts)
        ).view(num_feet, num_feet_pts, 3)

        height_points = height_points + self.sim.link_pos[self.lookat_id, self.feet_indices, None, :]

        # compute points height
        if estimation is None:
            hmap = self.get_feet_hmap()[self.lookat_id]
        else:
            hmap = estimation + self.cfg.normalization.feet_height_correction

        height_points[:, :, 2] -= hmap.view(num_feet, -1)

        height_points = height_points.flatten(0, 1).cpu().numpy()
        self.pending_vis_task.append(dict(points=height_points, color=(1, 1, 0)))

    def _draw_body_hmap(self, estimation=None):
        # compute points position
        height_points = transform_by_yaw(
            self.body_hmap_points[self.lookat_id],
            self.base_euler[self.lookat_id, 2].repeat(self.body_hmap_points.size(1))
        )

        height_points[:] += self.sim.root_pos[self.lookat_id, None, :]

        # compute points height
        if estimation is None:
            hmap = self.get_body_hmap()[self.lookat_id]
        else:
            hmap = estimation.flatten() + self.cfg.normalization.scan_norm_bias

        height_points[:, 2] -= hmap

        height_points = height_points.cpu().numpy()
        self.pending_vis_task.append(dict(points=height_points, color=(1, 1, 0)))

    def _compute_ref_state(self):
        # clock_l, clock_r = self._get_clock_input()
        #
        # ref_dof_pos = self._zero_tensor(self.num_envs, self.num_actions)
        # scale_1 = self.cfg.rewards.target_joint_pos_scale
        # scale_2 = 2 * scale_1
        #
        # # left swing (with double support phase)
        # clock_l[clock_l > self.cfg.commands.double_support_phase] = 0
        # ref_dof_pos[:, 1] = clock_l * scale_1
        # ref_dof_pos[:, 4] = -clock_l * scale_2
        # ref_dof_pos[:, 5] = clock_l * scale_1
        #
        # # right swing (with double support phase)
        # clock_r[clock_r > self.cfg.commands.double_support_phase] = 0
        # ref_dof_pos[:, 7] = clock_r * scale_1
        # ref_dof_pos[:, 10] = -clock_r * scale_2
        # ref_dof_pos[:, 11] = clock_r * scale_1
        #
        # self.ref_dof_pos[:] = self.init_state_dof_pos
        # self.ref_dof_pos[:, self.dof_activated] += ref_dof_pos

        ref_dof_pos = self._zero_tensor(self.num_envs, self.num_actions)
        scale_1 = self.cfg.commands.target_joint_pos_scale
        scale_2 = 2 * scale_1

        phase = torch.stack(self._get_clock_input(wrap_sin=False), dim=1)
        swing_ratio = self.cfg.commands.air_ratio
        delta_t = self.cfg.commands.delta_t

        phase_swing = torch.clip((phase - delta_t / 2) / (swing_ratio - delta_t), min=0., max=1.)
        clock = torch.sin(torch.pi * phase_swing)

        # left motion
        ref_dof_pos[:, 1] = -clock[:, 0] * scale_1
        ref_dof_pos[:, 4] = clock[:, 0] * scale_2
        ref_dof_pos[:, 5] = -clock[:, 0] * scale_1

        # right motion
        ref_dof_pos[:, 7] = -clock[:, 1] * scale_1
        ref_dof_pos[:, 10] = clock[:, 1] * scale_2
        ref_dof_pos[:, 11] = -clock[:, 1] * scale_1

        self.ref_dof_pos[:] = self.init_state_dof_pos
        self.ref_dof_pos[:, self.dof_activated] += ref_dof_pos

    def draw_cloud_from_depth(self):
        # draw points cloud
        cloud, cloud_valid = self.sensors.get('depth_0', get_cloud=True)
        cloud, cloud_valid = cloud[self.lookat_id], cloud_valid[self.lookat_id]
        pts = cloud[cloud_valid].cpu().numpy()

        if len(pts) > 0:
            pts = density_weighted_sampling(pts, 500)
            self.pending_vis_task.append(dict(points=pts))


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
