import math

import torch
import warp as wp

from legged_gym.envs.base.utils import DelayBuffer
from legged_gym.simulator import get_simulator, SimulatorType, DriveMode, SensorManager
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.joystick import JoystickHandler
from legged_gym.utils.math import torch_rand_float, inv_quat, axis_angle_to_quat, quat_to_xyz, transform_quat_by_quat, transform_by_quat, xyz_to_quat


class BaseTask:
    def __init__(self, cfg, args):
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.cfg = cfg
        self.debug = True
        self.init_done = False

        if hasattr(args, 'drive_mode') and args.drive_mode is not None:
            self.cfg.asset.default_dof_drive_mode = args.drive_mode
        if args.simulator is SimulatorType.IsaacGym:
            self.cfg.asset.default_dof_drive_mode = 3

        # prepare simulator
        simulator_class = get_simulator(args.simulator)
        # from legged_gym.simulator.genesis_wrapper import GenesisWrapper
        # self.sim: GenesisWrapper = simulator_class(cfg, args)
        self.sim = simulator_class(cfg, args)

        self._parse_cfg(args)
        self._init_robot_props()
        self._init_buffers()
        self._prepare_reward_function()

        if cfg.sensors.activated:
            self.sensors = SensorManager(cfg, self.sim, self.device)

        # reset agents to initialize them
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._post_physics_step()

        self.init_done = True

    # ------------------------------------------------- Interfaces -------------------------------------------------

    def get_observations(self):
        return self.actor_obs

    def get_critic_observations(self):
        return self.critic_obs

    @property
    def lookat_id(self):
        return self.sim.lookat_id

    # ------------------------------------------------- Utils -------------------------------------------------
    def _zero_tensor(self, *shape, dtype=torch.float, requires_grad=False):
        return torch.zeros(*shape, dtype=dtype, device=self.device, requires_grad=requires_grad)

    def _zero_wp_array(self, *shape, dtype=None, device=None):
        if dtype is None:
            raise ValueError('dtype cannot be None!')

        if device is None:
            device = wp.device_from_torch(self.device)

        return wp.zeros(shape, dtype=dtype, device=device)

    # ------------------------------------------------- Initialization -------------------------------------------------

    def _parse_cfg(self, args):
        self.device = torch.device(args.device)
        self.num_envs = self.cfg.env.num_envs
        self.dt = self.cfg.control.decimation * self.cfg.sim.dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)
        self.push_interval = math.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        self.num_dof = self.sim.num_dof
        self.num_actions = self.cfg.env.num_actions

        self.obs_scales = self.cfg.normalization.obs_scales
        self.only_positive_rewards = self.cfg.rewards.only_positive_rewards
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)

        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] * self.cfg.rewards.rew_norm_factor

    def _init_buffers(self):
        self.base_euler = self._zero_tensor(self.num_envs, 3)  # in base frame
        self.base_lin_vel = self._zero_tensor(self.num_envs, 3)  # in base frame
        self.base_ang_vel = self._zero_tensor(self.num_envs, 3)  # in base frame
        self.projected_gravity = self._zero_tensor(self.num_envs, 3)  # # in base frame
        self.gravity_vec = torch.tensor([[0, 0, -1]], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)  # in world frame
        self.forward_vec = torch.tensor([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # allocate buffers
        self.actions = self._zero_tensor(self.num_envs, self.num_dof)  # action clipped
        self.last_action_output = self._zero_tensor(self.num_envs, self.num_actions)  # network output
        self.actor_obs, self.critic_obs = None, None
        self.rew_buf = self._zero_tensor(self.num_envs)
        self.reset_buf = self._zero_tensor(self.num_envs, dtype=torch.bool)
        self.episode_length_buf = self._zero_tensor(self.num_envs, dtype=torch.long)
        self.time_out_cutoff = self._zero_tensor(self.num_envs, dtype=torch.bool)
        self.commands = self._zero_tensor(self.num_envs, self.cfg.commands.num_commands)  # x vel, y vel, yaw vel, heading
        self.is_zero_command = self._zero_tensor(self.num_envs, dtype=torch.bool)

        # Lag buffer
        if self.cfg.domain_rand.action_delay:
            self.action_delay_buf = DelayBuffer(self.num_envs, (self.num_dof,),
                                                self.cfg.domain_rand.action_delay_range.pop(0),
                                                self.cfg.domain_rand.randomize_action_delay_each_step,
                                                device=self.device)

        if self.cfg.domain_rand.add_dof_lag:
            self.dof_lag_buf = DelayBuffer(self.num_envs, (self.num_dof, 2),
                                           self.cfg.domain_rand.dof_lag_range,
                                           self.cfg.domain_rand.randomize_dof_lag_each_step,
                                           device=self.device)

        if self.cfg.domain_rand.add_imu_lag:
            self.imu_lag_buf = DelayBuffer(self.num_envs, (4 + 3,),
                                           self.cfg.domain_rand.imu_lag_range,
                                           self.cfg.domain_rand.randomize_imu_lag_each_step,
                                           device=self.device)

        # Push robot
        self.ext_force = self._zero_tensor(self.num_envs, 3)
        self.ext_torque = self._zero_tensor(self.num_envs, 3)

        self.global_counter = 0
        self.extras = {}

        if not self.sim.headless:
            self.joystick_handler = JoystickHandler(self.sim)

    # ---------------------------------------------- Robots Creation ----------------------------------------------

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise, create a grid.
        """
        self.env_origins = self._zero_tensor(self.num_envs, 3)
        self.env_class = self._zero_tensor(self.num_envs)

        # create a grid of robots
        num_cols = math.floor(math.sqrt(self.num_envs))
        num_rows = math.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='xy')
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _init_robot_props(self):
        # process base initial state
        self.init_state_pos = torch.tensor(self.cfg.init_state.pos, dtype=torch.float, device=self.device).unsqueeze(0)
        self.init_state_quat = torch.tensor(self.cfg.init_state.rot, dtype=torch.float, device=self.device).unsqueeze(0)
        self.init_state_lin_vel = torch.tensor(self.cfg.init_state.lin_vel, dtype=torch.float, device=self.device).unsqueeze(0)
        self.init_state_ang_vel = torch.tensor(self.cfg.init_state.ang_vel, dtype=torch.float, device=self.device).unsqueeze(0)

        # joint positions offsets and PD gains
        self.init_state_dof_pos = self._zero_tensor(1, self.num_dof)
        self.p_gains = self._zero_tensor(1, self.num_dof)
        self.d_gains = self._zero_tensor(1, self.num_dof)
        self.dof_activated = self._zero_tensor(self.num_dof, dtype=torch.bool)

        for i, dof_name in enumerate(self.sim.dof_names):
            self.init_state_dof_pos[0, i] = self.cfg.init_state.default_joint_angles[dof_name]
            found = False

            for key in self.cfg.control.stiffness.keys():
                if key in dof_name:
                    self.p_gains[0, i] = self.cfg.control.stiffness[key]
                    self.d_gains[0, i] = self.cfg.control.damping[key]
                    found = True

            if not found:
                raise ValueError(f"PD gain of joint {dof_name} were not defined")

            for key in self.cfg.control.activated:
                if key in dof_name:
                    self.dof_activated[i] = True

        # set the kp and kd value for simulator built-in PD controller
        if self.sim.drive_mode is not DriveMode.torque:
            self.sim.set_dof_kp(self.p_gains.repeat(self.num_envs, 1))
            self.sim.set_dof_kv(self.d_gains.repeat(self.num_envs, 1))

        # compute soft dof position limits
        self.soft_dof_pos_limits = self.sim.dof_pos_limits.clone()
        m = (self.soft_dof_pos_limits[:, 0] + self.soft_dof_pos_limits[:, 1]) / 2
        r = self.soft_dof_pos_limits[:, 1] - self.soft_dof_pos_limits[:, 0]
        self.soft_dof_pos_limits[:, 0] = (m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit)
        self.soft_dof_pos_limits[:, 1] = (m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit)

        # initialize env reset origin and goals
        self._get_env_origins()

        # create indices
        self.penalised_contact_indices = self.sim.create_indices(
            self.sim.get_full_names(self.cfg.asset.penalize_contacts_on, True), True)
        self.termination_contact_indices = self.sim.create_indices(
            self.sim.get_full_names(self.cfg.asset.terminate_after_contacts_on, True), True)

    # ------------------------------------------------- Simulation Step -------------------------------------------------

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.last_action_output[:] = actions

        # clip action range
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions[:, self.dof_activated] = torch.clip(self.last_action_output, -clip_actions, clip_actions)

        if self.cfg.domain_rand.action_delay:
            self.action_delay_buf.append(self.actions)

        # step the simulator
        self._step_environment()
        self._post_physics_step()

        return self.actor_obs, self.critic_obs, self.rew_buf.clone(), self.reset_buf.clone(), self.extras

    def _step_environment(self):
        if self.sim.drive_mode is DriveMode.torque:
            for step_i in range(self.cfg.control.decimation):
                if self.cfg.domain_rand.action_delay:
                    self.action_delay_buf.step()
                    self.actions[:] = self.action_delay_buf.get()

                torques = self._compute_torques()
                self.sim.control_dof_torque(torques)
                self.sim.step_environment()

                if self.cfg.domain_rand.add_dof_lag:
                    self.dof_lag_buf.append(torch.stack([self.sim.dof_pos[:, self.dof_activated],
                                                         self.sim.dof_vel[:, self.dof_activated]], dim=2))
                    self.dof_lag_buf.step()

                if self.cfg.domain_rand.add_imu_lag:
                    self.imu_lag_buf.append(torch.stack([self.sim.root_quat,
                                                         self.sim.root_ang_vel], dim=2))
                    self.imu_lag_buf.step()
        else:
            target_dof_pos = self.actions * self.cfg.control.action_scale + self.init_state_dof_pos
            self.sim.control_dof_position(target_dof_pos)
            self.sim.step_environment()

    def _compute_torques(self):
        # pd controller
        target_dof_pos = self.actions * self.cfg.control.action_scale + self.init_state_dof_pos

        if self.cfg.domain_rand.randomize_motor_offset:
            target_dof_pos[:] += self.motor_offsets

        if self.cfg.domain_rand.randomize_gains:
            torques = self.p_gain_multiplier * self.p_gains * (target_dof_pos - self.sim.dof_pos)
            torques[:] -= self.d_gain_multiplier * self.d_gains * self.sim.dof_vel
        else:
            torques = self.p_gains * (target_dof_pos - self.sim.dof_pos)
            torques[:] -= self.d_gains * self.sim.dof_vel

        if self.cfg.domain_rand.randomize_coulomb_friction:
            torques[:] -= self.sim.dof_vel * self.randomized_joint_viscous
            torques[:] -= torch.sign(self.sim.dof_vel) * self.randomized_joint_coulomb

        if self.cfg.domain_rand.randomize_torque:
            torques[:] *= self.torque_mul

        return torch.clip(torques, -self.sim.torque_limits, self.sim.torque_limits)

    def _post_physics_step(self):
        self.global_counter += 1

        self._refresh_variables()
        self._post_physics_pre_step()

        # compute observations, rewards, resets, ...
        if self.init_done:
            self._compute_reward()
            self._check_termination()
            self._reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
            self._refresh_variables()

        self._post_physics_mid_step()
        if self.cfg.sensors.activated:
            self.sensors.update(self.global_counter, self.sim.root_pos, self.sim.root_quat, self.episode_length_buf <= 1)
        self._compute_observations()
        self._post_physics_post_step()

        if not self.sim.headless:
            self.render()
            self.joystick_handler.handle_device_input()

    def _refresh_variables(self):
        self.sim.refresh_variable()

        self.is_zero_command[:] = torch.logical_and(
            torch.norm(self.commands[:, :2], dim=1) <= self.cfg.commands.lin_vel_clip,
            torch.abs(self.commands[:, 2]) <= self.cfg.commands.ang_vel_clip
        )

        # prepare quantities
        # self.base_euler[:] = quat_to_xyz(transform_quat_by_quat(
        #     self.sim.root_quat,
        #     inv_quat(self.init_state_quat.repeat(self.num_envs, 1))
        # ))
        self.base_euler[:] = quat_to_xyz(self.sim.root_quat)
        inv_quat_yaw = axis_angle_to_quat(
            -self.base_euler[:, 2],
            torch.tensor([0, 0, 1], device=self.device, dtype=torch.float)
        )
        inv_base_quat = inv_quat(self.sim.root_quat)
        self.base_lin_vel[:] = transform_by_quat(self.sim.root_lin_vel, inv_quat_yaw)
        self.base_ang_vel[:] = transform_by_quat(self.sim.root_ang_vel, inv_base_quat)
        self.projected_gravity[:] = transform_by_quat(self.gravity_vec, inv_base_quat)

    def _post_physics_pre_step(self):
        self.episode_length_buf[:] += 1

        if self.cfg.domain_rand.action_delay and (self.global_counter % self.cfg.domain_rand.action_delay_update_steps == 0):
            if len(self.cfg.domain_rand.action_delay_range) > 0:
                self.action_delay_buf.update_delay_range(self.cfg.domain_rand.action_delay_range.pop(0))

    def _post_physics_mid_step(self):
        if self.cfg.play.control:
            # overwrite commands
            self.commands.zero_()
            self.commands[self.lookat_id, :3] = torch.Tensor(self.joystick_handler.get_control_input())
        else:
            self._update_command()

        # constraint command ranges
        self.commands[:, :2] *= torch.norm(self.commands[:, :2], dim=1, keepdim=True) > self.cfg.commands.lin_vel_clip
        self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_clip

        # push robot
        if self.cfg.domain_rand.push_robots:
            duration_i = int(self.global_counter / self.cfg.domain_rand.push_duration_update_steps)
            duration_i = min(duration_i, len(self.cfg.domain_rand.push_duration) - 1)
            duration = self.cfg.domain_rand.push_duration[duration_i] / self.dt
            if self.global_counter % self.push_interval <= duration:
                self._push_robots()
            else:
                self.ext_force.zero_()
                self.ext_torque.zero_()

    def _post_physics_post_step(self):
        raise NotImplementedError

    def _update_command(self):
        # resample command target
        env_ids = self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        self._resample_commands(torch.arange(self.num_envs, device=self.device)[env_ids])

    def _check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf[:] = torch.any(torch.norm(self.sim.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        roll_cutoff = torch.abs(self.base_euler[:, 0]) > 0.6
        pitch_cutoff = torch.abs(self.base_euler[:, 1]) > 0.6
        height_cutoff = self.sim.root_pos[:, 2] < -10
        self.time_out_cutoff[:] = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs

        self.reset_buf[:] |= self.time_out_cutoff
        self.reset_buf[:] |= roll_cutoff
        self.reset_buf[:] |= pitch_cutoff
        self.reset_buf[:] |= height_cutoff

    def _resample_commands(self, env_ids: torch.Tensor):
        raise NotImplementedError

    def _compute_observations(self):
        # raise NotImplementedError
        pass

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. """
        apply_force = self._zero_tensor(self.num_envs, self.sim.num_bodies, 3)
        apply_torque = self._zero_tensor(self.num_envs, self.sim.num_bodies, 3)

        if self.global_counter % self.push_interval == 0:
            self.ext_force[:, 0] = torch_rand_float(self.cfg.domain_rand.push_force_max[0][0],
                                                    self.cfg.domain_rand.push_force_max[0][1],
                                                    (self.num_envs, 1),
                                                    device=self.device).squeeze(1)
            self.ext_force[:, 1] = torch_rand_float(self.cfg.domain_rand.push_force_max[1][0],
                                                    self.cfg.domain_rand.push_force_max[1][1],
                                                    (self.num_envs, 1),
                                                    device=self.device).squeeze(1)
            self.ext_force[:, 2] = torch_rand_float(self.cfg.domain_rand.push_force_max[2][0],
                                                    self.cfg.domain_rand.push_force_max[2][1],
                                                    (self.num_envs, 1),
                                                    device=self.device).squeeze(1)
            self.ext_torque[:] = torch_rand_float(self.cfg.domain_rand.push_torque_max[0],
                                                  self.cfg.domain_rand.push_torque_max[1],
                                                  (self.num_envs, 3),
                                                  device=self.device)

        apply_force[:, 0] = self.ext_force * self.is_zero_command.unsqueeze(-1)
        apply_torque[:, 0] = self.ext_torque * self.is_zero_command.unsqueeze(-1)
        self.sim.apply_perturbation(apply_force, apply_torque)

    def render(self):
        self.sim.render()

    # ---------------------------------------------- Robots Reset ----------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        # reset robot states
        self.sim.control_dof_torque(self._zero_tensor(self.num_envs, self.num_dof))
        self._reset_dof_state(env_ids)
        self._reset_root_state(env_ids)
        self._resample_commands(env_ids)
        self.sim.step_environment()

        if self.cfg.domain_rand.action_delay:
            self.action_delay_buf.reset(env_ids)
        if self.cfg.domain_rand.add_dof_lag:
            self.dof_lag_buf.reset(env_ids)
        if self.cfg.domain_rand.add_imu_lag:
            self.imu_lag_buf.reset(env_ids)

        # Randomize joint parameters:
        self._randomize_dof_props(env_ids)

    def _reset_dof_state(self, env_ids):
        dof_pos = self.init_state_dof_pos.repeat(len(env_ids), 1)
        if self.cfg.domain_rand.randomize_start_dof_pos:
            dof_pos[:] += torch_rand_float(-self.cfg.domain_rand.randomize_start_dof_pos_range,
                                           self.cfg.domain_rand.randomize_start_dof_pos_range,
                                           (len(env_ids), self.num_dof),
                                           device=self.device)
        dof_vel = self._zero_tensor(len(env_ids), self.num_dof)
        if self.cfg.domain_rand.randomize_start_dof_vel:
            dof_vel[:] += torch_rand_float(-self.cfg.domain_rand.randomize_start_dof_vel_range,
                                           self.cfg.domain_rand.randomize_start_dof_vel_range,
                                           (len(env_ids), self.num_dof),
                                           device=self.device)

        self.sim.set_dof_state(env_ids, dof_pos, dof_vel)

    def _reset_root_state(self, env_ids):
        # base position
        root_pos = self.env_origins[env_ids] + self.init_state_pos

        # randomize base position
        if self.cfg.domain_rand.randomize_start_pos:
            root_pos[:, :2] += torch_rand_float(-self.cfg.domain_rand.randomize_start_pos_range,
                                                self.cfg.domain_rand.randomize_start_pos_range,
                                                (len(env_ids), 2),
                                                device=self.device)

        if self.cfg.domain_rand.randomize_start_y:
            root_pos[:, 2] += torch.abs(torch_rand_float(-self.cfg.domain_rand.randomize_start_y_range,
                                                         self.cfg.domain_rand.randomize_start_y_range,
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1))

        # base velocity
        root_lin_vel = self._zero_tensor(len(env_ids), 3)
        root_ang_vel = self._zero_tensor(len(env_ids), 3)
        if self.cfg.domain_rand.randomize_start_vel:
            root_lin_vel[:, :2] += torch_rand_float(-self.cfg.domain_rand.randomize_start_vel_range,
                                                    self.cfg.domain_rand.randomize_start_vel_range,
                                                    (len(env_ids), 2),
                                                    device=self.device)

        # randomize base orientation
        rand_root_euler = self._zero_tensor(len(env_ids), 3)
        if self.cfg.domain_rand.randomize_start_pitch:
            rand_root_euler[:, 1:2] = torch_rand_float(-self.cfg.domain_rand.randomize_start_pitch_range,
                                                       self.cfg.domain_rand.randomize_start_pitch_range,
                                                       (len(env_ids), 1),
                                                       device=self.device)

        if self.cfg.domain_rand.randomize_start_yaw:
            rand_root_euler[:, 2:3] = torch_rand_float(-self.cfg.domain_rand.randomize_start_yaw_range,
                                                       self.cfg.domain_rand.randomize_start_yaw_range,
                                                       (len(env_ids), 1),
                                                       device=self.device)
        root_quat = transform_quat_by_quat(
            xyz_to_quat(rand_root_euler),
            self.init_state_quat.repeat(len(env_ids), 1)
        )

        self.sim.set_root_state(env_ids,
                                root_pos,
                                root_quat,
                                root_lin_vel,
                                root_ang_vel)

    def _randomize_dof_props(self, env_ids):
        if not self.init_done:
            if self.cfg.domain_rand.randomize_torque:
                self.torque_mul = self._zero_tensor(self.num_envs, self.num_dof)

            if self.cfg.domain_rand.randomize_gains:
                self.p_gain_multiplier = self._zero_tensor(self.num_envs, self.num_dof)
                self.d_gain_multiplier = self._zero_tensor(self.num_envs, self.num_dof)

            if self.cfg.domain_rand.randomize_motor_offset:
                self.motor_offsets = self._zero_tensor(self.num_envs, self.num_dof)

            if self.cfg.domain_rand.randomize_joint_stiffness:
                self.joint_stiffness = self._zero_tensor(self.num_envs)

            if self.cfg.domain_rand.randomize_joint_damping:
                self.joint_damping_multiplier = self._zero_tensor(self.num_envs)

            if self.cfg.domain_rand.randomize_joint_friction:
                self.joint_friction = self._zero_tensor(self.num_envs)

            if self.cfg.domain_rand.randomize_joint_armature:
                self.joint_armatures = self._zero_tensor(self.num_envs)

            if self.cfg.domain_rand.randomize_coulomb_friction:
                self.randomized_joint_coulomb = self._zero_tensor(self.num_envs, self.num_dof)
                self.randomized_joint_viscous = self._zero_tensor(self.num_envs, self.num_dof)

        if self.cfg.domain_rand.randomize_torque:
            self.torque_mul[env_ids] = torch_rand_float(self.cfg.domain_rand.torque_multiplier_range[0],
                                                        self.cfg.domain_rand.torque_multiplier_range[1],
                                                        (len(env_ids), self.num_dof),
                                                        device=self.device)

        if self.cfg.domain_rand.randomize_gains:
            self.p_gain_multiplier[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_multiplier_range[0],
                                                               self.cfg.domain_rand.kp_multiplier_range[1],
                                                               (len(env_ids), self.num_dof),
                                                               device=self.device)
            self.d_gain_multiplier[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_multiplier_range[0],
                                                               self.cfg.domain_rand.kd_multiplier_range[1],
                                                               (len(env_ids), self.num_dof),
                                                               device=self.device)
            if self.sim.drive_mode is not DriveMode.torque:
                self.sim.set_dof_kp(self.p_gain_multiplier[env_ids] * self.p_gains, env_ids)
                self.sim.set_dof_kv(self.d_gain_multiplier[env_ids] * self.d_gains, env_ids)

        if self.cfg.domain_rand.randomize_motor_offset:
            self.motor_offsets[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_offset_range[0],
                                                           self.cfg.domain_rand.motor_offset_range[1],
                                                           (len(env_ids), self.num_dof),
                                                           device=self.device)

        if self.cfg.domain_rand.randomize_joint_stiffness:
            self.joint_stiffness[env_ids] = torch_rand_float(self.cfg.domain_rand.joint_stiffness_range[0],
                                                             self.cfg.domain_rand.joint_stiffness_range[1],
                                                             (len(env_ids), 1),
                                                             device=self.device).squeeze(1)
            stiffness_all = self.joint_stiffness[env_ids].unsqueeze(1).repeat(1, self.num_dof)
            self.sim.set_dof_stiffness(stiffness_all, env_ids)
            raise NotImplementedError

        if self.cfg.domain_rand.randomize_joint_damping:
            self.joint_damping_multiplier[env_ids] = torch_rand_float(self.cfg.domain_rand.joint_damping_multiplier_range[0],
                                                                      self.cfg.domain_rand.joint_damping_multiplier_range[1],
                                                                      (len(env_ids), 1),
                                                                      device=self.device).squeeze(1)
            damping_multiplier_all = self.joint_damping_multiplier[env_ids].unsqueeze(1).repeat(1, self.num_dof)
            self.sim.set_dof_damping_coef(damping_multiplier_all, env_ids)

        if self.cfg.domain_rand.randomize_joint_friction:
            self.joint_friction[env_ids] = torch_rand_float(self.cfg.domain_rand.joint_friction_range[0],
                                                            self.cfg.domain_rand.joint_friction_range[1],
                                                            (len(env_ids), 1),
                                                            device=self.device).squeeze(1)
            friction_all = self.joint_friction[env_ids].unsqueeze(1).repeat(1, self.num_dof)
            self.sim.set_dof_friction(friction_all, env_ids)

        if self.cfg.domain_rand.randomize_joint_armature:
            self.joint_armatures[env_ids] = torch_rand_float(self.cfg.domain_rand.joint_armature_range[0],
                                                             self.cfg.domain_rand.joint_armature_range[1],
                                                             (len(env_ids), 1),
                                                             device=self.device).squeeze(1)
            armatures = self.joint_armatures[env_ids].unsqueeze(1).repeat(1, self.num_dof)
            self.sim.set_dof_armature(armatures, env_ids)

        if self.cfg.domain_rand.randomize_coulomb_friction:
            self.randomized_joint_coulomb[env_ids] = torch_rand_float(self.cfg.domain_rand.joint_coulomb_range[0],
                                                                      self.cfg.domain_rand.joint_coulomb_range[1],
                                                                      (len(env_ids), self.num_dof),
                                                                      device=self.device)
            self.randomized_joint_viscous[env_ids] = torch_rand_float(self.cfg.domain_rand.joint_viscous_range[0],
                                                                      self.cfg.domain_rand.joint_viscous_range[1],
                                                                      (len(env_ids), self.num_dof),
                                                                      device=self.device)

    # ---------------------------------------------- Reward ----------------------------------------------

    def _prepare_reward_function(self):
        # prepare list of functions
        self._reward_names = []
        self._reward_functions = []

        for name in self.reward_scales:
            if name == "termination":
                continue

            self._reward_names.append(name)
            self._reward_functions.append(getattr(self, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {name: self._zero_tensor(self.num_envs)
                             for name in self.reward_scales.keys()}

    def _compute_reward(self):
        self.rew_buf[:] = 0.

        for i, name in enumerate(self._reward_names):
            rew = self._reward_functions[i]() * self.reward_scales[name] * self.dt
            self.rew_buf[:] += rew
            self.episode_sums[name][:] += rew

        if self.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.)

        # add termination reward after clipping
        if 'termination' in self.reward_scales:
            assert hasattr(self, '_reward_termination')
            rew = self._reward_termination() * self.reward_scales["termination"] * self.dt
            self.rew_buf[:] += rew
            self.episode_sums["termination"][:] += rew

    def _reward_termination(self):
        raise NotImplementedError
