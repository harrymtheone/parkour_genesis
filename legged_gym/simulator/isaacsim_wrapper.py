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

        raise NotImplementedError
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
            self.scene = scene.InteractiveScene(scene_cfg)

            print("[INFO]: Setup complete...")
            self.sim.reset()

            sim_dt = self.sim.get_physics_dt()
            import time
            robot = self.scene["robot"]
            while self.simulation_app.is_running():
                time_start = time.time()

                efforts = torch.randn_like(robot.data.joint_pos) * 5.0
                robot.set_joint_effort_target(efforts)
                robot.write_data_to_sim()

                self.sim.step()
                self.scene.update(sim_dt)

                while time.time() - time_start < sim_dt * 1:
                    self.sim.render()

        print("[INFO]: Scene manager: ", self.scene)

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
        from isaaclab import assets
        from isaaclab import actuators

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
