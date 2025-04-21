import os

import genesis as gs
import numpy as np
import torch
from genesis.ext.isaacgym import terrain_utils
from rich import print

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.simulator.base_wrapper import BaseWrapper, DriveMode
from legged_gym.utils.terrain import Terrain


def wrapper_unsafe(func, *args, **kwargs):
    return func(*args, unsafe=True, **kwargs)


class GenesisWrapper(BaseWrapper):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.debug = True
        self.init_done = False

        # create envs, sim and viewer
        gs.init(backend=gs.gpu if self.device.type == 'cuda' else gs.cpu, logging_level='info')

        n_rendered_envs = args.n_rendered_envs if hasattr(args, 'n_rendered_envs') else 1
        self._create_scene(n_rendered_envs)
        self._process_robot_props()
        self._domain_rand()

        if not self.headless:
            self.viewer = self._scene.visualizer.viewer
            self.enable_viewer_sync = True
            self.free_cam = False
            self.lookat_vec = np.array([-0, 2, 1])
            self.last_lookat_pos = np.array([0, 0, 0])
            self.vis_tasks = []
            self.clear_lines = True

        self.lookat_id = 0

        self.init_done = True

    # ---------------------------------------------- Sim Creation ----------------------------------------------

    def _create_scene(self, n_rendered_envs):
        """ Creates simulation, terrain and environments
        """

        # compute dt and substeps based on drive mode
        if self.cfg.asset.default_dof_drive_mode == 1:
            self.drive_mode = DriveMode.pos_target
            dt = self.cfg.sim.dt * self.cfg.control.decimation
            substeps = self.cfg.control.decimation
        elif self.cfg.asset.default_dof_drive_mode == 2:
            self.drive_mode = DriveMode.vel_target
            dt = self.cfg.sim.dt * self.cfg.control.decimation
            substeps = self.cfg.control.decimation
            raise ValueError("pos_vel drive mode not implemented yet!")
        elif self.cfg.asset.default_dof_drive_mode == 3:
            dt = self.cfg.sim.dt
            substeps = 1
            self.drive_mode = DriveMode.torque
        else:
            raise ValueError(f'Invalid drive mode value: {self.cfg.asset.default_dof_drive_mode}')

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
            batch_dofs_info=True,  # TODO: bug not fixed?

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
        vis_options = gs.options.VisOptions(rendered_envs_idx=list(range(n_rendered_envs)))

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
            # TODO: maybe remove this in later release of Genesis?
            self.cfg.terrain.num_cols = 3 * sum(self.cfg.terrain.terrain_dict.values())

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
                links_to_keep=self.cfg.asset.links_to_keep,
                prioritize_urdf_material=True,
                pos=(0., 0., 1.0)
            ),
            visualize_contact=self.debug
        )

        # build the scene
        self._scene.build(n_envs=self.num_envs)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        horizontal_scale = self.cfg.terrain.horizontal_scale_downsample
        self._scene.add_entity(gs.morphs.Terrain(
            pos=(-self.cfg.terrain.border_size, -self.cfg.terrain.border_size, 0.0),
            horizontal_scale=horizontal_scale,
            vertical_scale=self.cfg.terrain.vertical_scale,
            height_field=self.terrain.height_field_raw_downsample,
        ))

        # self._scene.add_entity(gs.morphs.Terrain(
        #     n_subterrains=(5, 1),
        #     subterrain_size=(8, 8),
        #     horizontal_scale=0.25,
        #     vertical_scale=0.005,
        #     subterrain_types=[['random_uniform_terrain', ]] * 5,
        # ))
        self.height_samples = torch.tensor(self.terrain.height_field_raw, dtype=torch.float, device=self.device)
        self.height_guidance = torch.tensor(self.terrain.height_field_guidance, dtype=torch.float, device=self.device)
        self.edge_mask = torch.tensor(self.terrain.edge_mask, dtype=torch.bool, device=self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        # raise NotImplementedError("Terrain creation by trimesh is not supported by Genesis currently!")
        print(("Terrain creation by trimesh is not supported by Genesis currently! Switching to heightfield!"))
        return self._create_heightfield()

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
        self.edge_mask = torch.tensor(self.terrain.edge_mask, dtype=torch.bool, device=self.device)

    def _process_robot_props(self):
        self.rigid_solver = self._robot.solver

        self.num_bodies = self._robot.n_links
        self._body_names = [link.name for link in self._robot.links]
        self._dof_names = []
        for n in self.cfg.init_state.default_joint_angles:
            found = False
            for joint in self._robot.joints:
                if type(joint) is gs.datatypes.List:
                    joint = joint[0]

                if n == joint.name:
                    self._dof_names.append(n)
                    found = True
            if not found:
                raise ValueError(f'Joint {n} not found in the robot???')
        self.num_dof = len(self._dof_names)

        # create indices
        self._base_idx = torch.tensor([self._robot.get_link(self.cfg.asset.base_link_name).idx_local], dtype=torch.long, device=self.device)
        self._base_idx_scene_level = torch.tensor([self._robot.get_link(self.cfg.asset.base_link_name).idx], dtype=torch.long, device=self.device)
        self._body_indices = self.create_indices(self._body_names, True)  # contains base link!
        self._dof_indices = torch.tensor([self._robot.get_joint(n).dof_idx_local for n in self._dof_names], dtype=torch.long, device=self.device)

        # read dof properties
        self.dof_pos_limits = torch.stack(wrapper_unsafe(self._robot.get_dofs_limit, self._dof_indices), dim=1)
        self.torque_limits = wrapper_unsafe(self._robot.get_dofs_force_range, self._dof_indices)[1]

        # set joint stiffness
        joint_stiffness = self.cfg.asset.stiffness + self._zero_tensor(self.num_envs, self.num_dof)
        wrapper_unsafe(self._robot.set_dofs_stiffness, joint_stiffness, self._dof_indices)

        # set joint damping
        joint_damping = self.cfg.asset.angular_damping + self._zero_tensor(self.num_envs, self.num_dof)
        wrapper_unsafe(self._robot.set_dofs_damping, joint_damping, self._dof_indices)

        # set joint armature
        joint_armature = self.cfg.asset.armature + self._zero_tensor(self.num_envs, self.num_dof)
        wrapper_unsafe(self._robot.set_dofs_armature, joint_armature, self._dof_indices)

        if self.cfg.asset.friction > 0 and not self.suppress_warning:
            print(f"[bold red]⚠️ genesis has no joint friction?! [/bold red]")

    def create_indices(self, names, is_link):
        indices = self._zero_tensor(len(names), dtype=torch.long)

        for i, n in enumerate(names):
            if is_link:
                indices[i] = self._robot.get_link(n).idx_local
            else:
                for dof_i, dof_n in enumerate(self._dof_names):
                    if n == dof_n:
                        indices[i] = dof_i
                        break

                if n != dof_n:
                    raise ValueError(f"dof name \"{n}\" not found in self._dof_names")

        return indices

    def _domain_rand(self):
        env_ids = torch.arange(self.num_envs, device=self.device)

        # randomize rigid body properties
        self._randomize_rigid_body_props()

        if self.cfg.domain_rand.randomize_base_mass:
            self._robot.set_mass_shift(self.payload_masses, self._base_idx.clone(), env_ids)  # TODO: use this !!!

        if self.cfg.domain_rand.randomize_link_mass:
            # notice the [1:] slicing, this is because base is not considered
            link_mass = [self._robot.get_link(n).get_mass() for n in self._body_names]
            link_mass_shift = (self.link_mass_multiplier - 1) * torch.tensor(link_mass[1:], device=self.device).unsqueeze(0)
            self._robot.set_mass_shift(link_mass_shift, self._body_indices[1:].clone(), env_ids)

        if self.cfg.domain_rand.randomize_com:
            self._robot.set_COM_shift(self.com_displacements.unsqueeze(1), self._base_idx.clone(), env_ids)

        if self.cfg.domain_rand.randomize_friction:
            self._robot.set_friction_ratio(self.friction_coeffs.repeat(1, self.num_bodies), self._body_indices.clone(), env_ids)

            if not self.suppress_warning:
                print(f"[bold red]⚠️ Restitution is not supported by Genesis currently?![/bold red]")

        for i, l in enumerate(self._robot.links):
            self._link_mass[:, i] = l.get_mass()

        self._link_mass[:, 0:1] += self.payload_masses
        self._link_mass[:, 1:] *= self.link_mass_multiplier

    # ---------------------------------------------- IO Interface ----------------------------------------------

    def get_trimesh(self):
        vertices = self.terrain.vertices - np.array([[self.cfg.terrain.border_size, self.cfg.terrain.border_size, 0]])
        triangles = self.terrain.triangles
        return vertices, triangles

    def refresh_variable(self):
        pass

    def set_root_state(self, env_ids, pos, quat, lin_vel, ang_vel):
        self._robot.set_pos(pos, zero_velocity=True, envs_idx=env_ids)
        quat = quat[..., [3, 0, 1, 2]]  # [x, y, z, w] -> [w, x, y, z]
        self._robot.set_quat(quat, zero_velocity=True, envs_idx=env_ids)

        if not (self.suppress_warning or (lin_vel is None and ang_vel is None)):
            print(f"[bold red]⚠️ Setting base lin_vel and ang_vel is not supported by Genesis currently! [/bold red]")

    def set_dof_state(self, env_ids, dof_pos, dof_vel):
        self._robot.set_dofs_position(
            position=dof_pos,
            dofs_idx_local=self._dof_indices,
            zero_velocity=False,
            envs_idx=env_ids,
        )

        self._robot.set_dofs_velocity(
            velocity=dof_vel,
            dofs_idx_local=self._dof_indices,
            envs_idx=env_ids,
        )

    def set_dof_kp(self, kp, env_ids=None):
        self._robot.set_dofs_kp(kp, self._dof_indices, env_ids)

    def set_dof_kv(self, kd, env_ids=None):
        self._robot.set_dofs_kv(kd, self._dof_indices, env_ids)

    def set_dof_damping_coef(self, damping_coef, env_ids=None):
        damping = self._robot.get_dofs_damping(self._dof_indices, env_ids)
        self._robot.set_dofs_damping(damping * damping_coef, self._dof_indices, env_ids)

    def set_dof_friction_coef(self, friction_coef, env_ids=None):
        if not self.suppress_warning:
            print(f"[bold red]⚠️ genesis has no joint friction?! [/bold red]")

    def set_dof_armature(self, armature, env_ids=None):
        self._robot.set_dofs_armature(armature, self._dof_indices, env_ids)

    @property
    def root_pos(self):
        return self._robot.get_pos()

    @property
    def root_quat(self):
        return self._robot.get_quat()[..., [1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)

    @property
    def root_lin_vel(self):
        return self._robot.get_vel()

    @property
    def root_ang_vel(self):
        return self._robot.get_ang()

    @property
    def dof_pos(self):
        return self._robot.get_dofs_position(self._dof_indices)

    @property
    def dof_vel(self):
        return self._robot.get_dofs_velocity(self._dof_indices)

    @property
    def contact_forces(self):
        return self._robot.get_links_net_contact_force()

    @property
    def link_pos(self):
        return self._robot.get_links_pos()

    @property
    def link_quat(self):
        return self._robot.get_links_quat()[..., [1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)

    @property
    def link_vel(self):
        return self._robot.get_links_vel()

    @property
    def link_COM(self):
        return self.rigid_solver.get_links_COM([l.idx for l in self._robot.links])

    @property
    def link_mass(self):
        return self._link_mass

    # ---------------------------------------------- Step Interface ----------------------------------------------

    def apply_perturbation(self, force, torque, env_ids=None):
        self.rigid_solver.apply_links_external_force(force[:, :1], self._base_idx_scene_level, env_ids)
        self.rigid_solver.apply_links_external_torque(torque[:, :1], self._base_idx_scene_level, env_ids)

    def control_dof_torque(self, torques):
        self._robot.control_dofs_force(torques, self._dof_indices)

    def step_environment(self):
        self._scene.step(update_visualizer=False)

    def control_dof_position(self, target_dof_pos):
        self._robot.control_dofs_position(target_dof_pos, self._dof_indices)

    # ------------------------------------------------ Graphics ------------------------------------------------

    def render(self):
        if not self.free_cam:
            self.lookat(self.lookat_id)

        self._scene.clear_debug_objects()
        self._render_vis_tasks()

        if self.enable_viewer_sync:
            self.viewer.update()

        self.last_lookat_pos = self._robot.get_pos()[self.lookat_id].cpu().numpy()

    def refresh_graphics(self, clear_lines):
        pass

    def lookat(self, i):
        self.lookat_id = i % self.num_envs

        self.lookat_vec = self.viewer.camera_pos - self.last_lookat_pos
        look_at_pos = self._robot.get_pos()[self.lookat_id].cpu().numpy()
        cam_pos = look_at_pos + self.lookat_vec
        self.viewer.set_camera_pose(pos=cam_pos, lookat=look_at_pos)

    def draw_points(self, points, radius=0.02, color=(0, 1, 0), sphere_lines=4, z_shift=0.02):
        if type(points) == list:
            points = np.array(points)
        elif type(points) == np.ndarray:
            points = points.copy()
        else:
            raise ValueError("points must be a list or np.ndarray")

        assert points.ndim == 2 and points.shape[1] == 3

        if points.shape[0] == 0:
            return

        points[:, 2] += z_shift

        self.vis_tasks.append((points, radius, (*color, 0.5)))

    def _render_vis_tasks(self):
        while len(self.vis_tasks) > 0:
            self._scene.draw_debug_spheres(*self.vis_tasks.pop(0))
