from typing import List

import numpy as np
import torch

from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import torch_rand_float, transform_by_yaw, wrap_to_pi
from legged_gym.utils.terrain import Terrain
from .base_task import BaseTask


class ParkourTask(BaseTask):

    def _parse_cfg(self, args):
        super()._parse_cfg(args)
        self.cmd_ranges_flat = class_to_dict(self.cfg.commands.flat_ranges)
        self.cmd_ranges_stair = class_to_dict(self.cfg.commands.stair_ranges)
        self.cmd_ranges_parkour = class_to_dict(self.cfg.commands.parkour_ranges)

        self.curriculum = self.cfg.terrain.curriculum
        if self.cfg.terrain.description_type not in ['heightfield', 'trimesh']:
            self.curriculum = False

    # ----------------------------------------- Initialization -------------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device)

        # initialize some data used later on
        self.torques = self._zero_tensor(self.num_envs, self.num_actions)
        self.base_height = self._zero_tensor(self.num_envs)

        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.sim.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.sim.root_lin_vel)

        self.target_yaw = self._zero_tensor(self.num_envs)  # used by info panel in play.py
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self.reach_goal_timer = self._zero_tensor(self.num_envs)
            self.reached_goal_ids = self._zero_tensor(self.num_envs, dtype=torch.bool)
            self.reach_goal_cutoff = self._zero_tensor(self.num_envs, dtype=torch.bool)
            self.cur_goal_idx = self._zero_tensor(self.num_envs, dtype=torch.long)
            self.cur_goals = self._zero_tensor(self.num_envs, 3)
            self.delta_yaw = self._zero_tensor(self.num_envs)
            self.target_pos_rel = self._zero_tensor(self.num_envs, 2)
            self.num_trials = self._zero_tensor(self.num_envs, dtype=torch.long)

        self.base_hmap_points = self._init_height_points(self.cfg.terrain.base_pts_x, self.cfg.terrain.base_pts_y)
        if self.cfg.terrain.measure_heights:
            self.scan_points = self._init_height_points(self.cfg.terrain.scan_pts_x, self.cfg.terrain.scan_pts_y)
            self.num_scan = self.cfg.env.n_scan
            self.scan_hmap = self._zero_tensor(self.num_envs, self.num_scan)  # in world frame

    # ---------------------------------------------- Robots Creation ----------------------------------------------

    def _get_env_origins(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if max_init_level >= self.cfg.terrain.num_rows:
                raise ValueError("max_init_level should be less than num_rows")

            if not self.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1

            self.env_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), dtype=torch.long, device=self.device)
            self.env_cols = torch.div(torch.arange(self.num_envs, device=self.device),
                                      (self.num_envs / self.cfg.terrain.num_cols),
                                      rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.sim.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins = self.terrain_origins[self.env_levels, self.env_cols]  # (num_envs, 3)
            self.terrain_class = torch.from_numpy(self.sim.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class = self.terrain_class[self.env_levels, self.env_cols]  # (num_envs, )
            print('normal terrain env number:', torch.sum(self.env_class < 4).cpu().numpy())

            self.terrain_goals = torch.from_numpy(self.sim.terrain.goals).to(self.device).to(torch.float)
            self.terrain_goal_num = torch.from_numpy(self.sim.terrain.num_goals).to(self.device).to(torch.long)
            self.env_goals = self.terrain_goals[self.env_levels, self.env_cols]  # (num_envs, num_goals, 3))
            self.env_goal_num = self.terrain_goal_num[self.env_levels, self.env_cols]  # (num_envs, num_goals))

            self.target_pos_rel = self._zero_tensor(self.num_envs, 2)
            self.target_yaw = self._zero_tensor(self.num_envs)
            self.num_trials = self._zero_tensor(self.num_envs, dtype=torch.long)

        else:
            super()._get_env_origins()

    def _reset_idx(self, env_ids: torch.Tensor):
        # update curriculum
        if self.curriculum:
            self._update_terrain_curriculum(env_ids)

        super()._reset_idx(env_ids)

        # reset buffers
        self.last_action_output[:] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.

        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self.cur_goal_idx[env_ids] = 0
            self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras['episode_rew'] = {}
        for key in self.episode_sums.keys():
            self.extras['episode_rew']['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / (self.max_episode_length * self.dt)
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.curriculum:
            self.extras['episode_terrain_level'] = {}

            env_cls = np.unique(self.env_class.cpu().numpy())
            env_cls_name = [Terrain.terrain_type(c).name for c in env_cls]

            for c, n in zip(env_cls, env_cls_name):
                self.extras['episode_terrain_level'][n] = torch.mean(self.env_levels[self.env_class == c].float())

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras['time_outs'] = self.time_out_cutoff.clone()

            if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
                self.extras['reach_goals'] = self.reach_goal_cutoff.clone()

    # ----------------------------------------- Height Measurement -------------------------------------------

    def _init_height_points(self, pts_x, pts_y):
        """ Returns points at which the height measurements are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        x, y = torch.tensor(pts_x), torch.tensor(pts_y)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        points = self._zero_tensor(self.num_envs, grid_x.numel(), 3)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_scan(self):
        # convert height points coordinate to world frame
        return self._get_heights(
            transform_by_yaw(self.scan_points, self.base_euler[:, 2].repeat(1, self.num_scan)).unflatten(0, (self.num_envs, -1))
            + self.sim.root_pos.unsqueeze(1)
            + self.cfg.terrain.border_size
        )

    def get_base_height_map(self):
        # convert height points coordinate to world frame
        n_points = self.base_hmap_points.size(1)
        return self._get_heights(
            transform_by_yaw(self.base_hmap_points, self.base_euler[:, 2].repeat(1, n_points)).unflatten(0, (self.num_envs, -1))
            + self.sim.root_pos.unsqueeze(1)
            + self.cfg.terrain.border_size
        )

    def _get_heights(self, points, use_guidance=False):
        if self.cfg.terrain.description_type not in ['heightfield', 'trimesh']:
            return self._zero_tensor(self.num_envs, points.size(1))

        points = (points / self.sim.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.sim.height_samples.size(0) - 2)
        py = torch.clip(py, 0, self.sim.height_samples.size(1) - 2)

        if use_guidance:
            heights = self.sim.height_guidance[px, py]
            heights += self.sim.height_guidance[px + 1, py]
            heights += self.sim.height_guidance[px, py + 1]
            heights += self.sim.height_guidance[px + 1, py + 1]
            heights = (heights / 4).to(torch.short)
        else:
            heights1 = self.sim.height_samples[px, py]
            heights2 = self.sim.height_samples[px + 1, py]
            heights3 = self.sim.height_samples[px, py + 1]
            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    # ----------------------------------------- Post Physics -------------------------------------------

    def _refresh_variables(self):
        super()._refresh_variables()

        # update height measurements
        if self.cfg.terrain.measure_heights and (self.global_counter % self.cfg.terrain.height_update_interval == 0):
            self.scan_hmap[:] = self.get_scan()
        self.base_height[:] = self.sim.root_pos[:, 2] - self.get_base_height_map().mean(dim=1)

    def _check_termination(self):
        super()._check_termination()
        self.reach_goal_cutoff[:] = (self.cur_goal_idx >= self.env_goal_num) & (self.env_class >= 4)
        self.reset_buf[:] |= self.reach_goal_cutoff

    def _post_physics_mid_step(self):
        super()._post_physics_mid_step()

        self.cur_goals[:] = self.env_goals[torch.arange(self.num_envs), self.cur_goal_idx]

        # update goals
        dist = torch.norm(self.sim.root_pos[:, :2] - self.cur_goals[:, :2], dim=1)
        self.reached_goal_ids[:] = (dist < self.cfg.env.next_goal_threshold) & (self.env_class >= 4)

        # update goals
        self.reach_goal_timer[self.reached_goal_ids] += 1
        self.reach_goal_timer[~self.reached_goal_ids] = 0
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1

    def _post_physics_post_step(self):
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.sim.dof_vel
        self.last_torques[:] = self.torques
        self.last_root_vel[:] = self.sim.root_lin_vel

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return

        dis_to_origin = torch.norm(self.sim.root_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        threshold = torch.norm(self.commands[env_ids, :2], dim=1) * self.episode_length_buf[env_ids] * self.dt

        move_up = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        move_down = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        # curriculum logic for flat terrain
        env_is_flat = self.env_class[env_ids] < 2
        if len(env_is_flat) > 0:
            # move_up[env_is_flat] = torch.logical_or(self.time_out_buf[env_ids][env_is_flat],
            #                                         dis_to_origin[env_is_flat] > self.cfg.terrain.terrain_size[0] / 2)
            # move_down[env_is_flat] = ~move_up[env_is_flat]
            move_up[env_is_flat] = dis_to_origin[env_is_flat] > self.cfg.terrain.terrain_size[0] / 4  # half of one-side of the terrain
            move_down[env_is_flat] = dis_to_origin[env_is_flat] < self.cfg.terrain.terrain_size[0] / 8

        # curriculum logic for stair terrain
        env_is_stair = torch.logical_and(self.env_class[env_ids] >= 2, self.env_class[env_ids] < 4)
        if len(env_is_stair) > 0:
            move_up[env_is_stair] = dis_to_origin[env_is_stair] > self.cfg.terrain.terrain_size[0] / 2
            move_down[env_is_stair] = dis_to_origin[env_is_stair] < 0.4 * threshold[env_is_stair]

        # curriculum logic for parkour terrain
        env_is_parkour = self.env_class[env_ids] >= 4
        if len(env_is_parkour) > 0:
            move_up[env_is_parkour] = (self.cur_goal_idx >= self.env_goal_num)[env_ids][env_is_parkour]
            move_down[env_is_parkour] = (self.cur_goal_idx < self.env_goal_num // 2)[env_ids][env_is_parkour]

        self.env_levels[env_ids] += 1 * move_up - 1 * move_down

        # downgrade after multiple failures
        level_changed = torch.logical_xor(move_up, move_down)
        self.num_trials[env_ids[level_changed]] = 0
        self.num_trials[env_ids[~level_changed]] += 1
        self.env_levels[env_ids] = torch.where(
            self.num_trials[env_ids] >= 5,
            # torch.randint_like(self.env_levels[env_ids], low=0, high=self.env_levels[env_ids]),
            (torch.rand(self.env_levels[env_ids].shape, device=self.device) * self.env_levels[env_ids]).long(),
            self.env_levels[env_ids]
        )
        self.num_trials[self.num_trials >= 5] = 0

        # Robots that solve the last level are sent to a random one
        self.env_levels[env_ids] = torch.where(
            self.env_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.env_levels[env_ids], self.max_terrain_level),
            torch.clip(self.env_levels[env_ids], 0)
        )  # (the minimum level is zero)

        # randomize the terrain level for flat terrain
        # self.env_levels[env_ids[env_is_flat]] = torch.randint_like(self.env_levels[env_ids[env_is_flat]], self.max_terrain_level)
        self.env_origins[env_ids] = self.terrain_origins[self.env_levels[env_ids], self.env_cols[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.env_levels[env_ids], self.env_cols[env_ids]]
        self.env_goals[:] = self.terrain_goals[self.env_levels, self.env_cols]
        self.env_goal_num[:] = self.terrain_goal_num[self.env_levels, self.env_cols]

    def _resample_commands(self, env_ids: torch.Tensor):
        self.commands[env_ids] = 0
        motion_type = self._zero_tensor(self.num_envs, dtype=torch.long)

        def sample_cmd(rng, clip, num_samp):
            if rng[0] * rng[1] > 0:  # same sign
                assert abs(rng[0]) >= clip, "Abs of range_min should be greater than clip"
                return torch_rand_float(rng[0], rng[1], (num_samp, 1), device=self.device).squeeze(1)
            else:  # different sign
                cmd = torch_rand_float(rng[0], rng[1], (num_samp, 1), device=self.device).squeeze(1)
                ratio = torch.where(cmd < 0, (rng[0] + clip) / rng[0], (rng[1] - clip) / rng[1])
                return cmd * ratio + torch.sign(cmd) * clip

        def sample_motion_type(weights, num_samp):  # stationary, xy, yaw, xy & yaw
            weights = torch.tensor(weights, dtype=torch.float, device=self.device)
            return torch.multinomial(weights / weights.sum(), num_samp, replacement=True)

        # sample command for slope terrain (omniverse mode, no heading)
        env_ids_flat = env_ids[self.env_class[env_ids] < 2]
        if len(env_ids_flat) > 0:
            self.commands[env_ids_flat, 0] = sample_cmd(self.cmd_ranges_flat["lin_vel_x"],
                                                        self.cfg.commands.lin_vel_clip,
                                                        len(env_ids_flat))
            self.commands[env_ids_flat, 1] = sample_cmd(self.cmd_ranges_flat["lin_vel_y"],
                                                        self.cfg.commands.lin_vel_clip,
                                                        len(env_ids_flat))

            self.commands[env_ids_flat, 2] = sample_cmd(self.cmd_ranges_flat["ang_vel_yaw"],
                                                        self.cfg.commands.ang_vel_clip,
                                                        len(env_ids_flat))
            motion_type[env_ids_flat] = sample_motion_type([2, 2, 2, 4], len(env_ids_flat))

        # sample command for stair terrain (heading mode, yaw command is updated by heading)
        env_ids_stair = env_ids[torch.logical_and(self.env_class[env_ids] >= 2, self.env_class[env_ids] < 4)]
        if len(env_ids_stair) > 0:
            self.commands[env_ids_stair, 0] = sample_cmd(self.cmd_ranges_stair["lin_vel_x"],
                                                         self.cfg.commands.lin_vel_clip,
                                                         len(env_ids_stair))
            # self.commands[env_ids_stair, 1] = sample_cmd(self.cmd_ranges_stair["lin_vel_y"],
            #                                                     self.cfg.commands.lin_vel_clip,
            #                                                     len(env_ids_stair))

            self.commands[env_ids_stair, 3] = sample_cmd(self.cmd_ranges_stair["heading"],
                                                         self.cfg.commands.ang_vel_clip,
                                                         len(env_ids_stair))
            motion_type[env_ids_stair] = sample_motion_type([2, 0, 0, 8], len(env_ids_stair))

        # sample command for parkour terrain (goal guided, no yaw command)
        env_ids_parkour = env_ids[self.env_class[env_ids] >= 4]
        if len(env_ids_parkour) > 0:
            self.commands[env_ids_parkour, 0] = sample_cmd(self.cmd_ranges_parkour["lin_vel_x"],
                                                           self.cfg.commands.lin_vel_clip,
                                                           len(env_ids_parkour))
            motion_type[env_ids_parkour] = sample_motion_type([1, 0, 0, 9], len(env_ids_parkour))

        # # for parkour stair, we don't want too large speed
        # env_ids_parkour_stair = env_ids[self.env_class[env_ids] == Terrain.terrain_type.parkour_stair]
        # if len(env_ids_parkour_stair) > 0:
        #     # self.commands[env_ids_parkour_stair, 0] = torch.abs(sample_cmd(self.cmd_ranges_stair["lin_vel_x"],
        #     #                                                                self.cfg.commands.lin_vel_clip,
        #     #                                                                len(env_ids_parkour_stair)))
        #     self.commands_parkour[env_ids_parkour_stair, 0] = torch.abs(sample_cmd(self.cmd_ranges_stair["lin_vel_x"],
        #                                                                            self.cfg.commands.lin_vel_clip,
        #                                                                            len(env_ids_parkour_stair)))

        # re-scale command (to prevent speed norm greater than x_vel_max)
        commands_normal = self.commands[self.env_class < 4]
        command_norm = torch.norm(commands_normal[:, :2], dim=1, keepdim=True)
        command_norm = torch.clip(command_norm, min=self.cmd_ranges_flat["lin_vel_x"][1])
        commands_normal[:, :2] *= self.cmd_ranges_flat["lin_vel_x"][1] / command_norm

        # randomly clip some commands
        motion_type = motion_type[env_ids]
        self.commands[env_ids[motion_type == 0]] = 0  # stationary
        self.commands[env_ids[motion_type == 1], 2:4] = 0  # xy
        self.commands[env_ids[motion_type == 2], :2] = 0  # yaw

        self.command_x_parkour[:] = self.commands[:, 0]

    def _update_command(self):
        super()._update_command()

        if self.cfg.terrain.description_type not in ["heightfield", "trimesh"]:
            return

        # update target_pos_rel and target_yaw
        self.target_pos_rel[:] = self.cur_goals[:, :2] - self.sim.root_pos[:, :2]
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw[:] = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        if self.global_counter % 5 == 0:
            self.delta_yaw[:] = wrap_to_pi(self.target_yaw - self.base_euler[:, 2])

        if not self.cfg.play.control:
            # stair terrains use heading commands
            env_is_stair = torch.logical_and(self.env_class >= 2, self.env_class < 4)
            forward = transform_by_yaw(self.forward_vec, self.base_euler[:, 2])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[env_is_stair, 2] = wrap_to_pi(self.commands[env_is_stair, 3] - heading[env_is_stair])
            self.commands[env_is_stair, 2] = torch.clip(self.commands[env_is_stair, 2], *self.cmd_ranges_stair['ang_vel_yaw'])

            # envs' yaw value of parkour terrain is computed using yaw difference
            env_is_parkour = self.env_class >= 4
            self.commands[env_is_parkour, 2] = self.delta_yaw[env_is_parkour]
            self.commands[env_is_parkour, 2] = torch.clip(self.commands[env_is_parkour, 2], *self.cmd_ranges_parkour['ang_vel_yaw'])

            cmd_ratio = torch.clip(1 - 2 * torch.abs(self.delta_yaw / torch.pi), min=0)
            self.commands[env_is_parkour, 0] = (cmd_ratio * self.command_x_parkour)[env_is_parkour]

    def _add_noise(self, x, scale):
        if self.cfg.noise.add_noise:
            scale = scale * self.cfg.noise.noise_level
            return x + torch_rand_float(-scale, scale, x.shape, self.device)
        return x

    # ----------------------------------------- Graphics -------------------------------------------
    def draw_hmap(self, hmap, color=(0, 1, 0)):
        def func(hmap, color):
            hmap = hmap[self.lookat_id].flatten().cpu().numpy()
            base_pos = self.sim.root_pos[self.lookat_id].cpu().numpy()
            yaw = self.base_euler[self.lookat_id, 2].repeat(self.scan_points.size(1))
            height_points = transform_by_yaw(self.scan_points[self.lookat_id], yaw).cpu().numpy()
            height_points[:, :2] += base_pos[None, :2]

            height_points[:, 2] = base_pos[2] - hmap - self.cfg.normalization.scan_norm_bias
            self.sim.draw_points(height_points, color=color)

        self.pending_vis_task.append((func, hmap, color))

    def _draw_goals(self):
        if self.env_goal_num[self.lookat_id] == 0:
            return

        goals = self.env_goals[self.lookat_id, :self.env_goal_num[self.lookat_id]].cpu().numpy()
        self.sim.draw_points(goals, self.cfg.env.next_goal_threshold, (1, 0, 0), sphere_lines=16)

        cur_goal_idx = min(self.cur_goal_idx[self.lookat_id].item(), len(goals) - 1)
        cur_goal = goals[cur_goal_idx]

        if self.reached_goal_ids[self.lookat_id]:
            self.sim.draw_points([cur_goal], self.cfg.env.next_goal_threshold, (0, 1, 0), sphere_lines=16)
        else:
            self.sim.draw_points([cur_goal], self.cfg.env.next_goal_threshold, (0, 0, 1), sphere_lines=16)

    def _draw_camera(self):
        cam_pos = self.sensors.get('depth_0', get_pos=True)
        self.sim.draw_points(cam_pos, 0.05, (1, 0, 0), sphere_lines=16)

        # for i, goal in enumerate(self.env_goals[self.lookat_id].cpu().numpy()):
        #     pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
        #
        #     if i == self.cur_goal_idx[self.lookat_id].cpu().item():
        #         gymutil.draw_lines(self.sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        #
        #         if self.reached_goal_ids[self.lookat_id]:
        #             gymutil.draw_lines(self.sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        #     else:
        #         gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        #
        # if not self.cfg.depth.use_camera:
        #     sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
        #     pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
        #     for i in range(5):
        #         norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        #         target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        #         pose_arrow = pose_robot[:2] + 0.1 * (i + 3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
        #         pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
        #         gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_height_field(self, draw_guidance=True):
        # for debug use!!!
        # draw height lines
        if self.init_done or (not self.cfg.terrain.measure_heights):
            return

        pts = []
        for i in range(self.sim.height_samples.shape[0]):
            for j in range(self.sim.height_samples.shape[1]):
                if j % 10 != 0:
                    continue

                x = i * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
                y = j * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size

                if draw_guidance:
                    z = self.sim.height_guidance[i, j] * self.cfg.terrain.vertical_scale + 0.02
                else:
                    z = self.sim.height_samples[i, j] * self.cfg.terrain.vertical_scale + 0.02

                pts.append((x, y, z))
        self.sim.draw_points(pts)

    def _draw_edge(self):
        # for debug use!!!
        if self.init_done or (not self.cfg.terrain.measure_heights):
            return

        pts_edge = []
        pts_non_edge = []
        for i in range(self.sim.edge_mask.shape[0]):
            for j in range(self.sim.edge_mask.shape[1]):
                if j % 10 != 0:
                    continue
                x = i * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
                y = j * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
                z = self.sim.height_samples[i, j] * self.cfg.terrain.vertical_scale + 0.02

                if self.sim.edge_mask[i, j]:
                    pts_edge.append((x, y, z))
                else:
                    pts_non_edge.append((x, y, z))

        self.sim.draw_points(pts_edge, color=(0, 1, 0))
        self.sim.draw_points(pts_non_edge, color=(1, 0, 0))

    # def _draw_cloud_depth(self):
    #     cloud = self.cloud_depth[self.lookat_id]
    #     valid = self.cloud_depth_valid[self.lookat_id]
    #     cloud = cloud[valid].cpu().numpy()
    #     self._draw_point_cloud(cloud, color=(0, 1, 1))

    # def _draw_voxel_depth(self):
    #     if self.global_counter % self.cfg.depth.update_interval == 0 or self.global_counter < 10:
    #         self._pts_depth = self._draw_voxel_grid(*self._extract_voxel_grid(self.voxel_grid_depth),
    #                                                 ROI=self.cfg.reconstruction.ROI_depth_bound, color=(1, 0.5, 0))
    #     else:
    #         self._draw_point_cloud(self._pts_depth, sample_rate=1, color=(1, 0.5, 0))
    #
    # def _draw_voxel_terrain(self):
    #     if self.global_counter % self.cfg.depth.update_interval == 0 or self.global_counter < 10:
    #         self._pts_terrain = self._draw_voxel_grid(*self._extract_voxel_grid(self.voxel_grid_terrain),
    #                                                   ROI=self.cfg.reconstruction.ROI_body_bound, color=(1, 0, 1))
    #     else:
    #         self._draw_point_cloud(self._pts_terrain, sample_rate=1, color=(1, 0, 1))
    #
    # def _draw_voxel_recon(self):
    #     if self.voxel_recon is None:
    #         return
    #
    #     if self.global_counter % self.cfg.depth.update_interval == 0 or self.global_counter < 10:
    #         self._pts_recon = self._draw_voxel_grid(*self._extract_voxel_grid(self.voxel_recon),
    #                                                 ROI=self.cfg.reconstruction.ROI_body_bound)
    #     else:
    #         self._draw_point_cloud(self._pts_recon, sample_rate=1, color=(1, 1, 0))
    #
    # def _extract_voxel_grid(self, voxel_grid):
    #     return voxel_grid[self.lookat_id, ..., 0] > 0, voxel_grid[self.lookat_id, ..., 1:] + 0.5
    #
    # def _draw_voxel_grid(self, grid_occupied: torch.Tensor, grid_com: torch.Tensor, in_world_frame=False, ROI=None, color=(1, 1, 0)):
    #     if not in_world_frame:
    #         # transform to voxel grid root frame
    #         x = torch.arange(self.cfg.reconstruction.grid_shape[0], dtype=torch.long)
    #         y = torch.arange(self.cfg.reconstruction.grid_shape[1], dtype=torch.long)
    #         z = torch.arange(self.cfg.reconstruction.grid_shape[2], dtype=torch.long)
    #         x, y, z = torch.meshgrid(x, y, z, indexing='ij')
    #         indices = torch.stack((x, y, z), dim=-1).to(self.device)
    #         grid_com = grid_com.clone()
    #         grid_com[:] = (grid_com + indices) * self.cfg.reconstruction.grid_size
    #
    #         # convert to base frame
    #         grid_com += torch.tensor([ROI[0][0], ROI[1][0], ROI[2][0]], dtype=torch.float, device=self.device)[None, None, None, :]
    #
    #         # convert to world frame
    #         grid_com = quat_apply(self.base_quat[self.lookat_id].unsqueeze(0), grid_com)
    #         grid_com += self.root_states[self.lookat_id, None, None, None, :3]
    #
    #     if type(grid_occupied) is torch.Tensor:
    #         grid_occupied = grid_occupied.cpu().numpy()
    #         grid_com = grid_com.cpu().numpy()
    #
    #     pts = grid_com[grid_occupied]
    #     self._draw_point_cloud(pts, sample_rate=1, color=color)
    #     return pts

    # def _draw_point_cloud(self, cloud: np.ndarray, sample_rate=1.0, color=(0, 1, 0), env_id=None):
    #     if sample_rate < 1:
    #         cloud_idx = np.random.choice(np.arange(len(cloud)), int(len(cloud) * 0.1), replace=False)
    #     else:
    #         cloud_idx = np.arange(len(cloud))
    #
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color)
    #
    #     env_id = self.lookat_id is env_id is None
    #
    #     for idx in cloud_idx:
    #         sphere_pose = gymapi.Transform(gymapi.Vec3(cloud[idx, 0], cloud[idx, 1], cloud[idx, 2] + 0.02), r=None)
    #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)
