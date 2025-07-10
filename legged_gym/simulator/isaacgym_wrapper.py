import math
import os
import sys
import time

import numpy as np
import torch
from isaacgym import gymapi, gymutil, gymtorch
from tqdm import tqdm

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.simulator.base_wrapper import BaseWrapper, DriveMode
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import transform_by_quat
from legged_gym.utils.terrain import Terrain


class IsaacGymWrapper(BaseWrapper):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.debug = True
        self.init_done = False

        # create envs, sim and viewer
        self.gym = gymapi.acquire_gym()
        self._create_sim()
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()

        self.lookat_id = 0
        self.cam_handles = []
        if not self.headless:
            self._create_viewer()
            self.clear_lines = True

        self.init_done = True

    # ---------------------------------------------- Sim Creation ----------------------------------------------

    def _create_sim(self):
        """ Creates simulation, terrain and environments """
        # graphics device for rendering, -1 for no rendering
        sim_device, sim_device_id = gymutil.parse_device_str(self.device.type)

        if self.device.type == 'cuda':
            graphics_device_id = -1 if self.headless else 0
        else:
            graphics_device_id = sim_device_id

        # initialize sim params
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(class_to_dict(self.cfg.sim), sim_params)
        sim_params.use_gpu_pipeline = sim_device == 'cuda'
        sim_params.physx.use_gpu = sim_device == 'cuda'

        if self.cfg.asset.default_dof_drive_mode == 1:
            self.drive_mode = DriveMode.pos_target
            sim_params.dt = self.cfg.sim.dt * self.cfg.control.decimation
            sim_params.substeps = self.cfg.control.decimation
        elif self.cfg.asset.default_dof_drive_mode == 2:
            self.drive_mode = DriveMode.vel_target
            sim_params.dt = self.cfg.sim.dt * self.cfg.control.decimation
            sim_params.substeps = self.cfg.control.decimation
            raise ValueError("pos_vel drive mode not implemented yet!")
        elif self.cfg.asset.default_dof_drive_mode == 3:
            sim_params.dt = self.cfg.sim.dt
            sim_params.substeps = 1
            self.drive_mode = DriveMode.torque
        else:
            raise ValueError(f'Invalid drive mode value: {self.cfg.asset.default_dof_drive_mode}')

        self.sim = self.gym.create_sim(sim_device_id,
                                       graphics_device_id,
                                       gymapi.SIM_PHYSX,
                                       sim_params)
        mesh_type = self.cfg.terrain.description_type
        start = time.time()
        print("*" * 80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain)
        else:
            self.terrain = None

        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time.time() - start))
        print("*" * 80)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.height_field_raw.shape[1]
        hf_params.nbColumns = self.terrain.height_field_raw.shape[0]
        hf_params.transform.p.x = -self.terrain.border * self.cfg.terrain.horizontal_scale
        hf_params.transform.p.y = -self.terrain.border * self.cfg.terrain.horizontal_scale
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.height_field_raw.T.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.height_field_raw, device=self.device)
        self.height_guidance = torch.tensor(self.terrain.height_field_guidance, dtype=torch.float, device=self.device)
        self.edge_mask = torch.tensor(self.terrain.edge_mask, device=self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim,
                                   self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'),
                                   tm_params)
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.height_field_raw, dtype=torch.float, device=self.device)
        self.height_guidance = torch.tensor(self.terrain.height_field_guidance, dtype=torch.float, device=self.device)
        self.edge_mask = torch.tensor(self.terrain.edge_mask, device=self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # save body names from the asset
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        self._body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self._dof_names = self.gym.get_asset_dof_names(robot_asset)

        self._randomize_rigid_body_props()

        # handlers created
        self._actor_handles = []
        self._envs = []

        for env_i in tqdm(range(self.num_envs), desc="Creating env..."):
            # create env instance
            env_handle = self.gym.create_env(self.sim, gymapi.Vec3(0., 0., 0.), gymapi.Vec3(0., 0., 0.), int(math.sqrt(self.num_envs)))
            self._envs.append(env_handle)

            # process rigid shape properties
            self._process_rigid_shape_props(rigid_shape_props_asset, env_i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props_asset)

            # this is just for env creation, set to different start positions to avoid PhyX buffer overflow(IsaacGym)
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(np.random.uniform(0, 100), np.random.uniform(0, 100), 0)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "bridge", env_i, self.cfg.asset.self_collisions, 0)
            self._actor_handles.append(actor_handle)

            # process dof properties
            self._process_dof_props(dof_props_asset, env_i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props_asset)

            # process rigid body properties
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            self._process_rigid_body_props(body_props, env_i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

    def _process_rigid_shape_props(self, props, env_id: int):
        if self.cfg.domain_rand.randomize_friction:
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                props[s].compliance = self.compliance_coeffs[env_id]
                props[s].restitution = self.restitution_coeffs[env_id]

    def _process_dof_props(self, props, env_id: int):
        if env_id != 0:
            return

        self.dof_pos_limits = self._zero_tensor(self.num_dof, 2)
        self.dof_vel_limits = self._zero_tensor(self.num_dof)
        self.dof_torque_limits = self._zero_tensor(self.num_dof)

        for i in range(len(props)):
            self.dof_pos_limits[i, 0] = props["lower"][i].item()
            self.dof_pos_limits[i, 1] = props["upper"][i].item()
            self.dof_vel_limits[i] = props["velocity"][i].item()
            self.dof_torque_limits[i] = props["effort"][i].item()

            if self.cfg.asset.use_soft_limits:
                props["velocity"][i] *= self.cfg.asset.sim_dof_limit_mul
                props["effort"][i] *= self.cfg.asset.sim_dof_limit_mul

            # props["stiffness"][i] = self.cfg.asset.stiffness
            props["damping"][i] = self.cfg.asset.angular_damping
            # props["friction"][i] = self.cfg.asset.friction
            # props["armature"][i] = self.cfg.asset.armature

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            props[0].mass += self.payload_masses[env_id]

        if self.cfg.domain_rand.randomize_link_mass:
            for i in range(1, self.num_bodies):
                props[i].mass *= self.link_mass_multiplier[env_id, i - 1]

        if self.cfg.domain_rand.randomize_com:
            props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0],
                                       self.com_displacements[env_id, 1],
                                       self.com_displacements[env_id, 2])

        for i, p in enumerate(props):
            self._link_mass[env_id, i] = props[i].mass
            self._link_com_shift[env_id, i, 0] = p.com.x
            self._link_com_shift[env_id, i, 1] = p.com.y
            self._link_com_shift[env_id, i, 2] = p.com.z

    def create_indices(self, names, is_link):
        indices = self._zero_tensor(len(names), dtype=torch.long)

        if is_link:
            for i, n in enumerate(names):
                indices[i] = self.gym.find_actor_rigid_body_handle(self._envs[0], self._actor_handles[0], n)
        else:
            for i, n in enumerate(names):
                indices[i] = self._dof_names.index(n)
        return indices

    def _init_buffers(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))  # in world frame
        self._dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self._rigid_body_states = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, -1, 13)
        self._contact_forces = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis

    # ---------------------------------------------- IO Interface ----------------------------------------------
    def get_trimesh(self):
        if self.terrain is None:
            raise NotImplementedError
        else:
            vertices = self.terrain.vertices - np.array([[self.cfg.terrain.border_size, self.cfg.terrain.border_size, 0]])
            triangles = self.terrain.triangles

        return vertices, triangles

    def set_root_state(self, env_ids, root_pos, root_quat, root_lin_vel, root_ang_vel):
        self._root_state[env_ids, 0:3] = root_pos
        self._root_state[env_ids, 3:7] = root_quat
        self._root_state[env_ids, 7:10] = root_lin_vel
        self._root_state[env_ids, 10:13] = root_ang_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_state),
                                                     gymtorch.unwrap_tensor(env_ids_int32),
                                                     len(env_ids_int32))

    def set_dof_state(self, env_ids, dof_pos, dof_vel):
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))

    def _set_dof_prop(self, prop_name, value, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)

        for i, env_id in enumerate(env_ids):
            dof_props = self.gym.get_actor_dof_properties(self._envs[env_id], self._actor_handles[env_id])

            for dof_i in range(self.num_dof):
                dof_props[dof_i][prop_name] = value[i, dof_i]

            self.gym.set_actor_dof_properties(self._envs[env_id], self._actor_handles[env_id], dof_props)

    def set_dof_kp(self, kp, env_ids=None):
        # Only used when you use pos_target drive mode
        raise NotImplementedError

    def set_dof_kv(self, kp, env_ids=None):
        # Only used when you use pos_target drive mode
        raise NotImplementedError

    def set_dof_stiffness(self, stiffness, env_ids=None):
        self._set_dof_prop('stiffness', stiffness, env_ids)

    def set_dof_damping_coef(self, damping_coef, env_ids=None):
        self._set_dof_prop('damping', self.cfg.asset.angular_damping * damping_coef, env_ids)

    def set_dof_friction(self, friction, env_ids=None):
        self._set_dof_prop('friction', friction, env_ids)

    def set_dof_armature(self, armature, env_ids=None):
        self._set_dof_prop('armature', armature, env_ids)

    def refresh_variable(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    @property
    def root_pos(self):
        return self._root_state[:, :3]

    @property
    def root_quat(self):
        return self._root_state[:, 3:7]

    @property
    def root_lin_vel(self):
        return self._root_state[:, 7:10]

    @property
    def root_ang_vel(self):
        return self._root_state[:, 10:13]

    @property
    def dof_pos(self):
        return self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]

    @property
    def dof_vel(self):
        return self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

    @property
    def contact_forces(self):
        return self._contact_forces

    @property
    def link_pos(self):
        return self._rigid_body_states[..., 0:3]

    @property
    def link_quat(self):
        return self._rigid_body_states[..., 3:7]

    @property
    def link_vel(self):
        return self._rigid_body_states[..., 7:10]

    @property
    def link_ang_vel(self):
        return self._rigid_body_states[..., 10:13]

    @property
    def link_COM(self):
        return self.link_pos + transform_by_quat(self._link_com_shift, self.link_quat).unflatten(0, (self.num_envs, -1))

    @property
    def link_mass(self):
        return self._link_mass

    # ---------------------------------------------- Step Interface ----------------------------------------------

    def apply_perturbation(self, force, torque):
        self.gym.apply_rigid_body_force_tensors(self.sim,
                                                gymtorch.unwrap_tensor(force),
                                                gymtorch.unwrap_tensor(torque),
                                                gymapi.ENV_SPACE)

    def step_environment(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)

    # def _control_dof_position(self, target_dof_pos):
    #     self.robot.control_dofs_position(target_dof_pos, self._dof_indices_local)
    #     self.scene.step(update_visualizer=False)

    def control_dof_torque(self, torques):
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))

    # ------------------------------------------------ Graphics ------------------------------------------------

    def _create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        self.enable_viewer_sync = True
        self.free_cam = False
        self.lookat_vec = torch.tensor([-0, 2, 1], device=self.device)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "free_cam")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")

    def create_camera_sensor(self, env_i: int, name_link_attached_to, cam_props, cam_trans):
        camera_handle = self.gym.create_camera_sensor(self._envs[env_i], cam_props)
        root_handle = self.gym.find_actor_rigid_body_handle(self._envs[env_i], self._actor_handles[env_i], name_link_attached_to)
        self.gym.attach_camera_to_body(camera_handle, self._envs[env_i], root_handle, cam_trans, gymapi.FOLLOW_TRANSFORM)
        self.cam_handles.append(camera_handle)

    def _handle_key_event(self, event):
        for evt in event:
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()

            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

            elif evt.action == "free_cam" and evt.value > 0:
                self.free_cam = not self.free_cam
                if self.free_cam:
                    self.lookat(self.lookat_id)

            elif evt.action == "pause" and evt.value > 0:
                pause = True
                while pause:
                    time.sleep(0.1)
                    self.gym.draw_viewer(self.viewer, self.sim, True)

                    for evt in self.gym.query_viewer_action_events(self.viewer):
                        if evt.action == "pause" and evt.value > 0:
                            pause = False

                    if self.gym.query_viewer_has_closed(self.viewer):
                        sys.exit()

            elif not self.free_cam:
                if evt.action == "prev_id" and evt.value > 0:
                    self.lookat_id = (self.lookat_id - 1) % self.num_envs
                    self.lookat(self.lookat_id)
                elif evt.action == "next_id" and evt.value > 0:
                    self.lookat_id = (self.lookat_id + 1) % self.num_envs
                    self.lookat(self.lookat_id)

    def render(self):
        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        if not self.free_cam:
            self.lookat(self.lookat_id)

        # check for keyboard events
        self._handle_key_event(self.gym.query_viewer_action_events(self.viewer))

        # fetch results
        if self.device.type != 'cpu':
            self.gym.fetch_results(self.sim, True)

        self.gym.poll_viewer_events(self.viewer)

        # step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

        if not self.free_cam:
            p = self.gym.get_viewer_camera_transform(self.viewer, None).p
            cam_trans = torch.tensor([p.x, p.y, p.z], requires_grad=False, device=self.device)
            look_at_pos = self._root_state[self.lookat_id, :3].clone()
            self.lookat_vec = cam_trans - look_at_pos

    def refresh_graphics(self, clear_lines):
        self.render()

        if clear_lines and self.clear_lines:
            self.gym.clear_lines(self.viewer)

    def lookat(self, i):
        self.lookat_id = i % self.num_envs
        look_at_pos = self._root_state[self.lookat_id, :3].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(*cam_pos), gymapi.Vec3(*look_at_pos))

    def draw_points(self, points, radius=0.02, color=(0, 1, 0), sphere_lines=4, z_shift=0.02):
        sphere_geom = gymutil.WireframeSphereGeometry(radius, sphere_lines, sphere_lines, None, color=color)

        for pt in points:
            sphere_pose = gymapi.Transform(gymapi.Vec3(pt[0], pt[1], pt[2] + z_shift), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self._envs[self.lookat_id], sphere_pose)

    def render_camera(self, depth_tensor):
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for env_i in range(self.num_envs):
            depth_tensor[env_i] = -gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, self._envs[env_i], self.cam_handles[env_i], gymapi.IMAGE_DEPTH))
        self.gym.end_access_image_tensors(self.sim)
