import math
import os

import genesis as gs
import torch
import warp as wp
from genesis.engine.solvers import RigidSolver
from genesis.ext.isaacgym import terrain_utils
from rich import print

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.simulator.base_wrapper import BaseWrapper
from legged_gym.utils.math import torch_rand_float
from legged_gym.utils.terrain import Terrain


class GenesisWrapper(BaseWrapper):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.debug = True

        self._parse_cfg(args)

        # create envs, sim and viewer
        gs.init(backend=gs.gpu if self.device.type == 'cuda' else gs.cpu, logging_level='info')

        n_rendered_envs = args.n_rendered_envs if hasattr(args, 'n_rendered_envs') else 1
        self._create_scene(n_rendered_envs)
        self._process_robot_props()
        self._domain_rand()

        # # if running with a viewer, set up keyboard shortcuts and camera
        # if self.headless:
        #     self.viewer = None
        # else:
        #     self.viewer = self.scene.visualizer.viewer
        #
        #     self.enable_viewer_sync = True
        #     self.free_cam = True
        #     self.lookat_vec = np.array([-0, 2, 1])
        #
        #     if self.cfg.play.control:
        #         self.input_handler = JoystickHandler(self) if self.cfg.play.use_joystick else KeyboardHandler(self)
        #     else:
        #         self.input_handler = BaseHandler(self)
        #
        # self.lookat_id = 0

    # ---------------------------------------------- Sim Creation ----------------------------------------------
    def _parse_cfg(self, args):
        self.device = torch.device(args.device)
        self.headless = args.headless

        self.num_envs = self.cfg.env.num_envs

        self.dt = self.cfg.control.decimation * self.cfg.sim.dt

    def _create_scene(self, n_rendered_envs):
        """ Creates simulation, terrain and environments
        """
        if hasattr(self.cfg.control, 'use_genesis_torque_controller') and self.cfg.control.use_genesis_torque_controller:
            dt = self.dt / self.cfg.control.decimation
            substeps = 1
        else:
            dt = self.dt
            substeps = self.cfg.control.decimation

        if self.cfg.asset.disable_gravity:
            self.cfg.sim.gravity[2] = 0

        sim_options = gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            gravity=self.cfg.sim.gravity,
            floor_height=0.0,
            requires_grad=False,
        )

        rigid_options = gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            enable_self_collision=True,
            max_collision_pairs=100,
            constraint_solver=gs.constraint_solver.Newton,
        )
        viewer_options = gs.options.ViewerOptions(
            camera_pos=(2.0, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            refresh_rate=60,
            max_FPS=60
        )
        vis_options = gs.options.VisOptions(n_rendered_envs=n_rendered_envs)

        # create Scene
        self._scene = gs.Scene(
            sim_options=sim_options,
            rigid_options=rigid_options,
            viewer_options=viewer_options,
            vis_options=vis_options,
            show_viewer=not self.headless,
            show_FPS=False
        )

        # add terrain to the scene
        if hasattr(self.cfg.terrain, 'description_type'):
            terrain_desc_type = self.cfg.terrain.description_type
        else:
            raise ValueError("Terrain description type not specified!")

        if terrain_desc_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, terrain_utils)
        else:
            self.terrain = None

        if terrain_desc_type == 'plane':
            self._scene.add_entity(gs.morphs.Plane())
        elif terrain_desc_type == 'heightfield':
            self._create_heightfield()
        elif terrain_desc_type == 'trimesh':
            self._create_trimesh()
        else:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [plane, heightfield, trimesh]")

        # add robot to the scene
        self._robot = self._scene.add_entity(
            gs.morphs.URDF(
                file=self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
                merge_fixed_links=self.cfg.asset.collapse_fixed_joints,
                links_to_keep=self.cfg.asset.links_to_keep
            ),
            visualize_contact=self.debug
        )

        # build the scene
        self._scene.build(n_envs=self.num_envs)

        # save solver for advanced domain randomization
        # self.rigid_solver = self.scene.sim.active_solvers[0]
        # assert isinstance(self.rigid_solver, gs.engine.solvers.rigid.rigid_solver_decomp.RigidSolver), "Solver type is wrong???"

        for solver in self._scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        self._scene.add_entity(gs.morphs.Terrain(
            pos=(-self.terrain.border, -self.terrain.border, 0.0),
            horizontal_scale=self.cfg.terrain.horizontal_scale,
            vertical_scale=self.cfg.terrain.vertical_scale,
            height_field=self.terrain.height_field_raw,
        ))

        # terrain_entity = self.scene.add_entity(gs.morphs.Terrain(
        #     n_subterrains=(5, 1),
        #     subterrain_size=(8, 8),
        #     horizontal_scale=0.25,
        #     vertical_scale=0.005,
        #     subterrain_types=[['pyramid_sloped_terrain', ]] * 5,
        # ))
        self.height_samples = torch.tensor(self.terrain.height_field_raw, dtype=torch.float, device=self.device)
        self.height_guidance = torch.tensor(self.terrain.height_field_guidance, dtype=torch.float, device=self.device)
        self.edge_mask = torch.tensor(self.terrain.edge_mask, dtype=torch.float, device=self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """

        # save the trimesh to file to be loaded by genesis
        import open3d
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(self.terrain.vertices)
        mesh.triangles = open3d.utility.Vector3iVector(self.terrain.triangles)
        mesh_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'utils/terrain/terrain_mesh.obj')
        open3d.io.write_triangle_mesh(mesh_path, mesh)

        # load the trimesh using genesis
        self._scene.add_entity(gs.morphs.Mesh(
            file=mesh_path,
            scale=1.0,
            fixed=True,
            convexify=True,
            decompose_nonconvex=True,
            merge_submeshes_for_collision=True,
            decimate=False,
            pos=(-self.terrain.cfg.border_size, -self.terrain.cfg.border_size, 0.0),
        ))
        self.height_samples = torch.tensor(self.terrain.height_field_raw, dtype=torch.float, device=self.device)
        self.height_guidance = torch.tensor(self.terrain.height_field_guidance, dtype=torch.float, device=self.device)
        self.edge_mask = torch.tensor(self.terrain.edge_mask, dtype=torch.float, device=self.device)

    def _process_robot_props(self):
        self.num_bodies = self._robot.n_links

        self._body_names = [link.name for link in self._robot.links]
        self._dof_names = []
        for n in self.cfg.init_state.default_joint_angles:
            found = False
            for joint in self._robot.joints:
                if n == joint.name:
                    self._dof_names.append(n)
                    found = True
            if not found:
                raise ValueError(f'Joint {n} not found in the robot???')
        self.num_dof = len(self._dof_names)

        # create indices
        self._base_idx_scene_level = [self._robot.get_link(self.cfg.asset.base_link_name).idx, ]
        self._body_indices_scene_level = self._create_indices(self._body_names, True, True)  # contains base link!
        self._dof_indices_local = self._create_indices(self._dof_names, False, False)

        # read dof properties
        self.dof_pos_limits = torch.stack(self._robot.get_dofs_limit(self._dof_indices_local), dim=1)
        self.torque_limits = self._robot.get_dofs_force_range(self._dof_indices_local)[1]

    def create_indices(self, names, is_link):
        return self._create_indices(names, is_link, False)

    def _create_indices(self, names, is_link, scene_level):
        indices = self._zero_tensor(len(names), dtype=torch.long)

        for i, n in enumerate(names):
            if scene_level:
                indices[i] = self._robot.get_link(n).idx
            elif is_link:
                indices[i] = self._robot.get_link(n).idx_local
            else:
                indices[i] = self._robot.get_joint(n).dof_idx_local

        return indices

    def _domain_rand(self):
        env_ids = torch.arange(self.num_envs, device=self.device)

        # randomize rigid body properties
        self._randomize_rigid_body_props()

        if self.cfg.domain_rand.randomize_base_mass:
            self.rigid_solver.set_links_mass_shift(self.payload_masses, self._base_idx_scene_level, env_ids)
            # self._robot.set_mass_shift(self.payload_masses, self._base_idx_scene_level, env_ids)  # TODO: use this !!!

        if self.cfg.domain_rand.randomize_link_mass:
            # notice the [1:] slicing, this is because base is not considered
            link_mass = [self._robot.get_link(n).get_mass() for n in self._body_names]
            link_mass_shift = (self.link_mass_multiplier - 1) * torch.tensor(link_mass[1:], device=self.device).unsqueeze(0)
            self.rigid_solver.set_links_mass_shift(link_mass_shift, self._body_indices_scene_level[1:], env_ids)

        if self.cfg.domain_rand.randomize_com:
            self.rigid_solver.set_links_COM_shift(self.com_displacements.unsqueeze(1), self._base_idx_scene_level, env_ids)

        if self.cfg.domain_rand.randomize_friction:
            self.rigid_solver.set_geoms_friction_ratio(self.friction_coeffs, self._body_indices_scene_level, env_ids)

            if not self.suppress_warning:
                print(f"[bold red]⚠️ Restitution is not supported by Genesis currently?![/bold red]")  # Rich formatting

    # ---------------------------------------------- IO Interface ----------------------------------------------

    def set_root_state(self, env_ids, pos, quat, lin_vel, ang_vel):
        self._robot.set_pos(pos, zero_velocity=False, envs_idx=env_ids)
        self._robot.set_quat(quat, zero_velocity=False, envs_idx=env_ids)

        if not (lin_vel is None and ang_vel is None):
            print(f"[bold red]⚠️ Setting base lin_vel and ang_vel is not supported by Genesis currently![/bold red]")  # Rich formatting

    def set_dof_state(self, env_ids, dof_pos, dof_vel):
        self._robot.set_dofs_position(
            position=dof_pos,
            dofs_idx_local=self._dof_indices_local,
            zero_velocity=False,
            envs_idx=env_ids,
        )

        self._robot.set_dofs_velocity(
            velocity=dof_vel,
            dofs_idx_local=self._dof_indices_local,
            envs_idx=env_ids,
        )

    def refresh_variable(self):
        pass

    @property
    def root_pos(self):
        return self._robot.get_pos()

    @property
    def root_quat(self):
        quat = self._robot.get_quat()  # (w, x, y, z)
        return torch.cat([quat[:, 1:], quat[:, :1]], dim=1)

    @property
    def root_lin_vel(self):
        return self._robot.get_vel()

    @property
    def root_ang_vel(self):
        return self._robot.get_ang()

    @property
    def dof_pos(self):
        return self._robot.get_dofs_position(self._dof_indices_local)

    @property
    def dof_vel(self):
        return self._robot.get_dofs_velocity(self._dof_indices_local)

    @property
    def contact_forces(self):
        return self._robot.get_links_net_contact_force()

    @property
    def link_pos(self):
        return self._robot.get_links_pos()

    @property
    def link_quat(self):
        quat = self._robot.get_links_quat()  # (w, x, y, z)
        return torch.cat([quat[:, 1:], quat[:, :1]], dim=1)

    @property
    def link_vel(self):
        return self._robot.get_links_vel()

    # ---------------------------------------------- Step Interface ----------------------------------------------

    def apply_perturbation(self, force, torque):
        print(f"[bold red]⚠️ Apply external force is not implemented yet! [/bold red]")  # Rich formatting

    def step_environment(self):
        self._scene.step(update_visualizer=False)

    def control_dof_torque(self, torques):
        self._robot.control_dofs_force(torques, self._dof_indices_local)

    # def _control_dof_position(self, target_dof_pos):
    #     self.robot.control_dofs_position(target_dof_pos, self._dof_indices_local)
    #     self.scene.step(update_visualizer=False)

    # ------------------------------------------------ Graphics ------------------------------------------------

    def _visualization(self):
        pass

    def _render(self):
        if not self.free_cam:
            self.lookat(self.lookat_id)

        # check for keyboard events
        self.input_handler.handle_device_input()

        self.viewer.update()

        if not self.free_cam:
            self.lookat_vec = self.viewer.camera_pos - self._base_pos[self.lookat_id, :3].cpu().clone().numpy()

    def lookat(self, i):
        look_at_pos = self._base_pos[i, :3].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        self.viewer.set_camera_pose(pos=position, lookat=lookat)

    def prepare_camera(self):
        if not self.cfg.depth.use_camera:
            return

        # camera properties
        cfg = self.cfg.depth
        width, height = cfg.original
        horizontal_fov = cfg.horizontal_fov

        # camera randomization
        self.cam_offset_pos_design = torch.tensor([cfg.position], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        self.camera_position = self.cam_offset_pos_design + torch.cat([
            torch_rand_float(cfg.position_range[0][0], cfg.position_range[0][1], (self.num_envs, 1), device=self.device),
            torch_rand_float(cfg.position_range[1][0], cfg.position_range[1][1], (self.num_envs, 1), device=self.device),
            torch_rand_float(cfg.position_range[2][0], cfg.position_range[2][1], (self.num_envs, 1), device=self.device),
        ], dim=-1)

        camera_angle_design = torch.tensor([[0, cfg.angle, 0]], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        camera_angle = camera_angle_design.clone()
        camera_angle[:, 1:2] += torch_rand_float(cfg.angle_range[0], cfg.angle_range[1], (self.num_envs, 1), device=self.device)

        self.cam_offset_quat_design = gs.transform_quat_by_quat(
            gs.xyz_to_quat(camera_angle_design),
            gs.xyz_to_quat(torch.deg2rad(torch.tensor([[-90, 0, -90]], device=self.device))).repeat(self.num_envs, 1)
        )

        self.cam_offset_quat = gs.transform_quat_by_quat(
            gs.xyz_to_quat(camera_angle),
            gs.xyz_to_quat(torch.deg2rad(torch.tensor([[-90, 0, -90]], device=self.device))).repeat(self.num_envs, 1)
        )

        u_0, v_0 = width / 2, height / 2
        f = width / 2 / math.tan(math.radians(horizontal_fov) / 2)

        # simple pinhole model
        K = wp.mat44(
            f, 0.0, u_0, 0.0,
            0.0, f, v_0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        self.K_inv = wp.inverse(K)
        self.c_x, self.c_y = int(u_0), int(v_0)

    # def _draw_height_field(self, draw_guidance=True):
    #     # for debug use!!!
    #     # draw height lines
    #     if not self.terrain.cfg.measure_heights:
    #         return
    #
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.02, 3, 3, None, color=(0, 1, 0))
    #
    #     for i in range(self.height_samples.shape[0]):
    #         for j in range(self.height_samples.shape[1]):
    #             x = (i - self.terrain.border) * self.cfg.terrain.horizontal_scale
    #             y = (j - self.terrain.border) * self.cfg.terrain.horizontal_scale
    #
    #             if draw_guidance:
    #                 z = self.height_guidance[i, j] * self.cfg.terrain.vertical_scale + 0.02
    #             else:
    #                 z = self.height_samples[i, j] * self.cfg.terrain.vertical_scale + 0.02
    #             sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #             gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], sphere_pose)
    #
    # def _draw_edge(self):
    #     # for debug use!!!
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)
    #     non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 3, 3, None, color=(0, 1, 0))
    #     edge_geom = gymutil.WireframeSphereGeometry(0.02, 3, 3, None, color=(1, 0, 0))
    #
    #     for i in range(self.edge_mask.shape[0]):
    #         for j in range(self.edge_mask.shape[1]):
    #             x = (i - self.terrain.border) * self.cfg.terrain.horizontal_scale
    #             y = (j - self.terrain.border) * self.cfg.terrain.horizontal_scale
    #
    #             z = self.height_samples[i, j] * self.cfg.terrain.vertical_scale + 0.02
    #             sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #
    #             if self.edge_mask[i, j]:
    #                 gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], sphere_pose)
    #             else:
    #                 gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], sphere_pose)
    #
    # def _draw_camera(self):
    #     non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 8, 8, None, color=(1, 0, 0))
    #     cam_pos = tf_apply(self.base_quat, self.base_pos[:, :3], self.cam_offset_pos)[self.lookat_id]
    #     pose = gymapi.Transform(gymapi.Vec3(cam_pos[0], cam_pos[1], cam_pos[2]), r=None)
    #     gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #
    # def _draw_goals(self):
    #     for i, goal in enumerate(self.env_goals[self.lookat_id].cpu().numpy()):
    #         goal_xy = goal[:2] + self.terrain.cfg.border_size
    #         pts = (goal_xy / self.terrain.cfg.horizontal_scale).astype(int)
    #         goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
    #         pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
    #
    #         if i == self.cur_goal_idx[self.lookat_id].cpu().item():
    #             gymutil.draw_lines(self.sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #
    #             if self.reached_goal_ids[self.lookat_id]:
    #                 gymutil.draw_lines(self.sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #         else:
    #             gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #
    #     if not self.cfg.depth.use_camera:
    #         sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
    #         pose_robot = self.base_pos[self.lookat_id, :3].cpu().numpy()
    #         for i in range(5):
    #             norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
    #             target_vec_norm = self.target_pos_rel / (norm + 1e-5)
    #             pose_arrow = pose_robot[:2] + 0.1 * (i + 3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
    #             pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
    #             gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #
    # def _draw_feet_at_edge(self):
    #     if hasattr(self, 'feet_at_edge'):
    #         non_edge_geom = gymutil.WireframeSphereGeometry(0.022, 8, 8, None, color=(0, 1, 0))
    #         edge_geom = gymutil.WireframeSphereGeometry(0.022, 8, 8, None, color=(1, 0, 0))
    #         stumble_geom = gymutil.WireframeSphereGeometry(0.04, 8, 8, None, color=(1, 1, 0))
    #
    #         feet_pos = self.rigid_body_states[self.lookat_id, self.feet_indices, :3]
    #         force = self.contact_forces[self.lookat_id, self.feet_indices]
    #         stumble = torch.norm(force[:, :2], dim=1) > 4 * torch.abs(force[:, 2])
    #
    #         for i in range(4):
    #             pose = gymapi.Transform(gymapi.Vec3(feet_pos[i, 0], feet_pos[i, 1], feet_pos[i, 2]), r=None)
    #
    #             if self.feet_at_edge[self.lookat_id, i]:
    #                 gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #             else:
    #                 gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #
    #             if stumble[i]:
    #                 gymutil.draw_lines(stumble_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #
    # def _draw_feet_proj(self, use_guidance=False):
    #     geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))
    #
    #     # draw feet projection on the ground
    #     feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
    #     heights = self._get_heights(feet_pos + self.cfg.terrain.border_size, use_guidance=use_guidance)
    #     feet_pos[:, :, 2] = heights
    #     feet_pos = feet_pos[self.lookat_id]
    #
    #     for i in range(4):
    #         pose = gymapi.Transform(gymapi.Vec3(feet_pos[i, 0], feet_pos[i, 1], feet_pos[i, 2]), r=None)
    #         gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    # ------------------------------------------------ JoyStick ------------------------------------------------
