import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from .base_wrapper import BaseWrapper, DriveMode


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
            self.scene = scene.InteractiveScene(scene.InteractiveSceneCfg(
                num_envs=self.cfg.env.num_envs,
                env_spacing=self.cfg.env.env_spacing,
                lazy_sensor_update=True,
                replicate_physics=True,
                filter_collisions=True,
            ))
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)

    def _setup_scene(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)

        spawn = sim_utils.UsdFileCfg(
            usd_path=os.path.join(asset_root, asset_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
        )

        # prepare to override the articulation configuration in RoboVerse/humanoidverse/simulator/isaacsim_articulation_cfg.py
        default_joint_angles = copy.deepcopy(self.robot_config.init_state.default_joint_angles)
        # import ipdb; ipdb.set_trace()
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={
                joint_name: joint_angle for joint_name, joint_angle in default_joint_angles.items()
            },
            joint_vel={".*": 0.0},
        )

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
        actuators = {
            dof_names_list[i]: IdealPDActuatorCfg(
                joint_names_expr=[dof_names_list[i]],
                effort_limit=dof_effort_limit_list[i],
                velocity_limit=dof_vel_limit_list[i],
                stiffness=0,
                damping=0,
                armature=dof_armature_list[i],
                friction=dof_joint_friction_list[i],
            ) for i in range(len(dof_names_list))
        }

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
