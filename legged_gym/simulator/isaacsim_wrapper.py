import os

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from .base_wrapper import BaseWrapper, DriveMode
from ..utils.terrain import Terrain


class IsaacSimWrapper(BaseWrapper):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.debug = True
        self.init_done = False

        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(headless=self.headless, livestream=-1, enable_cameras=True)  # TODO: not finished yet
        self.simulation_app = app_launcher.app

        self._create_sim()
        self._process_robot_props()
        self._domain_rand()

        self.init_done = True

    def _create_sim(self):
        from isaaclab import sim as sim_utils
        from isaaclab import scene, utils

        render_fps = 60

        # compute simulation dt and substeps according to drive mode
        if self.cfg.asset.default_dof_drive_mode == 1:
            self.drive_mode = DriveMode.pos_target
            sim_dt = self.cfg.sim.dt * self.cfg.control.decimation
            sim_substeps = self.cfg.control.decimation
        elif self.cfg.asset.default_dof_drive_mode == 2:
            self.drive_mode = DriveMode.vel_target
            sim_dt = self.cfg.sim.dt * self.cfg.control.decimation
            sim_substeps = self.cfg.control.decimation
            raise ValueError("pos_vel drive mode not implemented yet!")
        elif self.cfg.asset.default_dof_drive_mode == 3:
            sim_dt = self.cfg.sim.dt
            sim_substeps = 1
            self.drive_mode = DriveMode.torque
        else:
            raise ValueError(f'Invalid drive mode value: {self.cfg.asset.default_dof_drive_mode}')

        # setting gravity vector to disable gravity
        if self.cfg.asset.disable_gravity:
            self.cfg.sim.gravity[2] = 0

        physx_cfg = sim_utils.PhysxCfg(
            solver_type=self.cfg.sim.physx.solver_type,
            bounce_threshold_velocity=0.5,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
        )

        render_cfg = sim_utils.RenderCfg()

        sim_config = sim_utils.SimulationCfg(
            device=str(self.device),
            dt=sim_dt,
            render_interval=max(int(1 / render_fps / sim_dt), self.cfg.control.decimation),
            gravity=self.cfg.sim.gravity,
            use_fabric=True,
            physx=physx_cfg,
            render=render_cfg,
        )

        # create a simulation context to control the simulator
        if sim_utils.SimulationContext.instance() is None:
            self.sim = sim_utils.SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        # generate scene
        with utils.Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            scene_cfg = scene.InteractiveSceneCfg(
                num_envs=self.cfg.env.num_envs,
                env_spacing=self.cfg.env.env_spacing,
                lazy_sensor_update=True,
                replicate_physics=True,
                filter_collisions=True,
            )

            self._setup_scene(scene_cfg)
            self._setup_robot(scene_cfg)
            self._scene = scene.InteractiveScene(scene_cfg)
            self._robot = self._scene["robot"]
            self._contact_sensor = self._scene["contact_sensor"]
            self.sim.reset()
            print("[INFO]: Setup complete...")

            # self.sim.reset()
            # sim_dt = self.sim.get_physics_dt()
            # import time
            # while self.simulation_app.is_running():
            #     time_start = time.time()
            #
            #     efforts = torch.randn_like(robot.data.joint_pos) * 5.0
            #     robot.set_joint_effort_target(efforts)
            #     robot.write_data_to_sim()
            #     self.sim.step()
            #     self.scene.update(sim_dt)
            #     while time.time() - time_start < sim_dt * 1:
            #         self.sim.render()

        print("[INFO]: Scene manager: ", self._scene)

    def _setup_scene(self, scene_cfg):
        from isaaclab import sim as sim_utils
        from isaaclab import assets

        # add lights
        scene_cfg.dome_light = assets.AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        )

        # add terrain
        mesh_type = self.cfg.terrain.description_type
        print("*" * 80)
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain)
        else:
            self.terrain = None

        # if mesh_type == 'plane':
        scene_cfg.ground = self._create_ground_plane()  # TODO: fix here
        # elif mesh_type == 'heightfield':
        #     self._create_heightfield()
        # elif mesh_type == 'trimesh':
        #     self._create_trimesh()
        # elif mesh_type is not None:
        #     raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("*" * 80)

    def _create_ground_plane(self):
        from isaaclab import sim as sim_utils
        from isaaclab import assets
        terrain_cfg = self.cfg.terrain

        # Ground-plane
        cfg_ground = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=terrain_cfg.static_friction,
                dynamic_friction=terrain_cfg.dynamic_friction,
                restitution=terrain_cfg.restitution,
            )
        )
        return assets.AssetBaseCfg(prim_path="/World/GroundPlane", spawn=cfg_ground)

    def _setup_robot(self, scene_cfg):
        from isaaclab import sim as sim_utils
        from isaaclab import assets, actuators, sensors

        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)

        # If the provide asset_path is a URDF file, find the USD file under the asset_root directory
        if asset_path.endswith('.urdf'):
            asset_root = os.path.join(asset_root, 'usd')

            if os.path.exists(asset_root):
                import carb
                carb.log_warn(f'URDF asset is not supported by IsaacSim, loading USD files from {asset_root}')
            else:
                raise NotImplementedError('URDF asset is not supported by IsaacSim, please convert it to USD first!')

            from pathlib import Path
            directory = Path(asset_root)  # Replace with your actual directory path
            asset_path = next(directory.glob('*.usd'), None).as_posix()

        asset_cfg = self.cfg.asset
        physx_cfg = self.cfg.sim.physx
        robot_usd_cfg = sim_utils.UsdFileCfg(
            usd_path=asset_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=asset_cfg.disable_gravity,
                linear_damping=asset_cfg.linear_damping,
                angular_damping=asset_cfg.angular_damping,
                max_linear_velocity=asset_cfg.max_linear_velocity,
                max_angular_velocity=asset_cfg.max_angular_velocity,
                max_depenetration_velocity=physx_cfg.max_depenetration_velocity,
                retain_accelerations=False,  # TODO: what does this do?
            ),
            activate_contact_sensors=True,  # TODO: Do we need it here? "Activate contact reporting on all rigid bodies. Defaults to False."
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=asset_cfg.self_collisions == 0,
                solver_position_iteration_count=physx_cfg.num_position_iterations,
                solver_velocity_iteration_count=physx_cfg.num_velocity_iterations
            ),
        )
        default_joint_angles = self.cfg.init_state.default_joint_angles
        init_state = assets.ArticulationCfg.InitialStateCfg(
            lin_vel=tuple(self.cfg.init_state.lin_vel),
            ang_vel=tuple(self.cfg.init_state.ang_vel),
            joint_pos={j_name: j_angle for j_name, j_angle in default_joint_angles.items()},
            joint_vel={".*": 0.0},
            pos=tuple(self.cfg.init_state.pos),
            rot=tuple([self.cfg.init_state.rot[3], *self.cfg.init_state.rot[:3]]),  # (x,y,z,w) to (w,x,y,z)
        )

        """
        See the following link for details
        https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html#howto-write-articulation-config
        """
        actuators = {
            f'all_actuator': actuators.IdealPDActuatorCfg(
                joint_names_expr=['.*'],
                # effort_limit=?,  # load from USD file
                # velocity_limit=?,  # load from USD file
                stiffness=self.cfg.asset.stiffness,
                damping=self.cfg.asset.angular_damping,  # TODO: what about prismatic joint?
                armature=self.cfg.asset.armature,
                friction=self.cfg.asset.friction,
            )
        }

        scene_cfg.robot = assets.ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=robot_usd_cfg,
            init_state=init_state,
            actuators=actuators,
        )

        scene_cfg.contact_sensor = sensors.ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
        )

        return

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list

        # get kp and kd from config
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping

        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict.keys():
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}")

        # ImplicitActuatorCfg IdealPDActuatorCfg

        robot_articulation_config = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state,
                                                             actuators=actuators)

        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
        )

        # Add a height scanner to the torso to detect the height of the terrain mesh
        height_scanner_config = RayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # Apply a grid pattern that is smaller than the resolution to only return one height value.
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        if (self.terrain_config.mesh_type == "heightfield") or (self.terrain_config.mesh_type == "trimesh"):
            sub_terrains = {}
            terrain_types = self.terrain_config.terrain_types
            terrain_proportions = self.terrain_config.terrain_proportions
            for terrain_type, proportion in zip(terrain_types, terrain_proportions):
                if proportion > 0:
                    if terrain_type == "flat":
                        sub_terrains[terrain_type] = terrain_gen.MeshPlaneTerrainCfg(
                            proportion=proportion
                        )
                    elif terrain_type == "rough":
                        sub_terrains[terrain_type] = terrain_gen.HfRandomUniformTerrainCfg(
                            proportion=proportion, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
                        )
                    elif terrain_type == "low_obst":
                        sub_terrains[terrain_type] = terrain_gen.MeshRandomGridTerrainCfg(
                            proportion=proportion, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
                        )

            terrain_generator_config = TerrainGeneratorCfg(
                curriculum=self.terrain_config.curriculum,
                size=(self.terrain_config.terrain_length, self.terrain_config.terrain_width),
                border_width=self.terrain_config.border_size,
                num_rows=self.terrain_config.num_rows,
                num_cols=self.terrain_config.num_cols,
                horizontal_scale=self.terrain_config.horizontal_scale,
                vertical_scale=self.terrain_config.vertical_scale,
                slope_threshold=self.terrain_config.slope_treshold,
                use_cache=False,
                sub_terrains=sub_terrains,
            )

            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_generator_config,
                max_init_terrain_level=9,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            # terrain_config.env_spacing = self.scene.cfg.env_spacing

        else:
            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing

        self._robot = Articulation(robot_articulation_config)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self._height_scanner = RayCaster(height_scanner_config)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.terrain = terrain_config.class_type(terrain_config)
        self.terrain.env_origins = self.terrain.terrain_origins

        # import ipdb; ipdb.set_trace()

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[terrain_config.prim_path])

        # add lights
        # light_config = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
        # light_config.func("/World/Light", light_config)

        light_config1 = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.98, 0.95, 0.88),
        )
        light_config1.func("/World/DomeLight", light_config1, translation=(1, 0, 10))

    def _process_robot_props(self):
        self._body_names = self._robot.body_names
        self.num_bodies = len(self._body_names)

        self.num_dof = len(self._robot.joint_names)
        self._dof_names = self._robot.joint_names
        self.dof_pos_limits = self._robot.data.joint_pos_limits[0]  # original (n_envs, n_joints, 2), slice the first env

    def create_indices(self, names, is_link):
        indices = self._zero_tensor(len(names), dtype=torch.long)

        for i, n in enumerate(names):
            if is_link:
                for body_i, body_n in enumerate(self._body_names):
                    if n == body_n:
                        indices[i] = body_i
                        break

                if n != body_n:
                    raise ValueError(f"link name \"{n}\" not found in self._body_names")

            else:
                for dof_i, dof_n in enumerate(self._dof_names):
                    if n == dof_n:
                        indices[i] = dof_i
                        break

                if n != dof_n:
                    raise ValueError(f"dof name \"{n}\" not found in self._dof_names")

        return indices

    def _domain_rand(self):
        from isaaclab.utils import configclass
        from isaaclab.managers import EventManager, EventTermCfg, SceneEntityCfg
        from isaaclab.envs import mdp

        # env_ids = torch.arange(self.num_envs, device=self.device)
        self.payload_masses = self._zero_tensor(self.num_envs, 1)
        self._link_mass = self._robot.data.default_mass.to(self.device)

        return

        # randomize rigid body properties
        @configclass
        class EventCfg:
            if self.cfg.domain_rand.randomize_base_mass:
                randomize_base_mass = EventTermCfg(
                    func=mdp.randomize_rigid_body_mass,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=self.cfg.asset.base_link_name),
                        "mass_distribution_params": self.cfg.domain_rand.added_mass_range,
                        "operation": 'add',
                        "distribution": 'uniform',
                        "recompute_inertia": True,
                    },
                )

            # robot_physics_material = EventTermCfg(
            #     func=mdp.randomize_rigid_body_material,
            #     mode="reset",
            #     params={
            #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            #         "static_friction_range": (0.7, 1.3),
            #         "dynamic_friction_range": (1.0, 1.0),
            #         "restitution_range": (1.0, 1.0),
            #         "num_buckets": 250,
            #     },
            # )
            # robot_joint_stiffness_and_damping = EventTermCfg(
            #     func=mdp.randomize_actuator_gains,
            #     mode="reset",
            #     params={
            #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            #         "stiffness_distribution_params": (0.75, 1.5),
            #         "damping_distribution_params": (0.3, 3.0),
            #         "operation": "scale",
            #         "distribution": "log_uniform",
            #     },
            # )
            # reset_gravity = EventTermCfg(
            #     func=mdp.randomize_physics_scene_gravity,
            #     mode="interval",
            #     is_global_time=True,
            #     interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
            #     params={
            #         "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            #         "operation": "add",
            #         "distribution": "gaussian",
            #     },
            # )

        self.event_manager = EventManager(EventTermCfg, self)
        print("[INFO] Event Manager: ", self.event_manager)

        # Randomize at once
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

        return





        self.events_cfg = EventCfg()
        if self.domain_rand_config.get("randomize_link_mass", False):
            self.events_cfg.scale_body_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": tuple(self.domain_rand_config["link_mass_range"]),
                    "operation": "scale",
                },
            )

        # Randomize joint friction
        if self.domain_rand_config.get("randomize_friction", False):
            self.events_cfg.random_joint_friction = EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "friction_distribution_params": tuple(self.domain_rand_config["friction_range"]),
                    "operation": "scale",
                },
            )

        if self.domain_rand_config.get("randomize_base_com", False):
            self.events_cfg.random_base_com = EventTerm(
                func=randomize_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=[
                            "torso_link",
                        ],
                    ),
                    "distribution_params": (
                        torch.tensor([self.domain_rand_config["base_com_range"]["x"][0], self.domain_rand_config["base_com_range"]["y"][0],
                                      self.domain_rand_config["base_com_range"]["z"][0]]),
                        torch.tensor([self.domain_rand_config["base_com_range"]["x"][1], self.domain_rand_config["base_com_range"]["y"][1],
                                      self.domain_rand_config["base_com_range"]["z"][1]])
                    ),
                    "operation": "add",
                    "distribution": "uniform",
                    "num_envs": self.simulator_config.scene.num_envs,
                },
            )

        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)











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

        self._link_mass[:] = self._robot.data.default_mass

        self._link_mass[:, 0:1] += self.payload_masses
        self._link_mass[:, 1:] *= self.link_mass_multiplier

    # ---------------------------------------------- IO Interface ----------------------------------------------

    def refresh_variable(self):
        pass
        # self._robot.update()

    def set_root_state(self, env_ids, pos, quat, lin_vel, ang_vel):
        quat = quat[..., [3, 0, 1, 2]]  # [x, y, z, w] -> [w, x, y, z]
        root_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=1)
        self._robot.write_root_state_to_sim(root_state, env_ids)

    def set_dof_state(self, env_ids, dof_pos, dof_vel):
        self._robot.write_joint_position_to_sim(
            position=dof_pos,
            joint_ids=None,  # all joints
            env_ids=env_ids,
        )

        self._robot.write_joint_velocity_to_sim(
            velocity=dof_vel,
            joint_ids=None,  # all joints
            env_ids=env_ids,
        )

    def set_dof_armature(self, armature, env_ids=None):
        self._robot.write_joint_armature_to_sim(
            armature=armature,
            joint_ids=None,  # all joints
            env_ids=env_ids
        )

    @property
    def root_pos(self):
        return self._robot.data.root_pos_w

    @property
    def root_quat(self):
        return self._robot.data.root_quat_w[..., [1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)

    @property
    def root_lin_vel(self):
        return self._robot.data.root_lin_vel_w

    @property
    def root_ang_vel(self):
        return self._robot.data.root_ang_vel_w

    @property
    def dof_pos(self):
        return self._robot.data.joint_pos

    @property
    def dof_vel(self):
        return self._robot.data.joint_vel

    @property
    def contact_forces(self):
        return self._contact_sensor.data.net_forces_w

    @property
    def link_pos(self):
        return self._robot.data.body_link_pos_w

    @property
    def link_quat(self):
        return self._robot.data.body_link_quat_w[..., [1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)

    # @property
    # def link_vel(self):
    #     return self._robot.get_links_vel()

    @property
    def link_COM(self):
        return self._robot.data.body_com_pos_w

    @property
    def link_mass(self):
        return self._link_mass

    # ---------------------------------------------- Step Interface ----------------------------------------------

    # def apply_perturbation(self, force, torque, env_ids=None):
    #     self.rigid_solver.apply_links_external_force(force[:, :1], self._base_idx_scene_level, env_ids)
    #     self.rigid_solver.apply_links_external_torque(torque[:, :1], self._base_idx_scene_level, env_ids)
    #
    def control_dof_torque(self, torques):
        self._robot.set_joint_effort_target(torques)
        self._robot.write_data_to_sim()

    def step_environment(self):
        self.sim.step()

    # def control_dof_position(self, target_dof_pos):
    #     self._robot.control_dofs_position(target_dof_pos, self._dof_indices)
