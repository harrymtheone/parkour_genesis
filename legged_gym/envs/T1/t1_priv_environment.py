import torch

from .t1_base_env import T1BaseEnv
from ..base.utils import ObsBase, HistoryBuffer


class ActorObs(ObsBase):
    def __init__(self, priv, priv_his, scan, edge_mask):
        super().__init__()
        self.priv = priv.clone()
        self.priv_his = priv_his.clone()
        self.scan = scan.clone()
        self.edge_mask = edge_mask.clone()


class T1PrivEnvironment(T1BaseEnv):
    def _init_robot_props(self):
        super()._init_robot_props()
        self.yaw_roll_dof_indices = self.sim.create_indices(
            self.sim.get_full_names(['Waist', 'Roll', 'Yaw'], False), False)

    def _init_buffers(self):
        super()._init_buffers()

        env_cfg = self.cfg.env
        self.priv_his_buf = HistoryBuffer(self.num_envs, env_cfg.len_critic_his, env_cfg.num_critic_obs, device=self.device)

        self.goal_task_timer = self._zero_tensor(self.num_envs)

    def _post_physics_mid_step(self):
        super()._post_physics_mid_step()

        self.goal_task_timer[self.reached_goal_ids] = 0
        timer_increase = ~self.reached_goal_ids & ~self.is_zero_command
        self.goal_task_timer[timer_increase] += 1 * self.commands[timer_increase, 0] / self.cfg.commands.parkour_ranges.lin_vel_x[1]

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

    def render(self):
        if self.cfg.terrain.description_type in ["heightfield", "trimesh"]:
            self._draw_goals()
            # self._draw_height_field(draw_guidance=True)
            # self._draw_edge()
            # self._draw_camera()
            # self._draw_feet_at_edge()
            self._draw_foothold()

        super().render()

    def _reward_timeout(self):
        time_out = self.goal_task_timer - 100
        effective_out = torch.clamp(time_out, min=0)
        time_out_rew = effective_out * 0.001
        time_out_rew[self.env_class < 4] = 0.
        return time_out_rew
