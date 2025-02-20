import warp as wp

from legged_gym.utils.helpers import class_to_dict
from .depth_cam import DepthCam


class SensorManager:
    def __init__(self, cfg, device, vertices, triangles):
        self.cfg = cfg
        self.device = device

        # vertices (n_vert, 3), triangles (n_tri)
        self.meshes = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3), indices=wp.array(triangles.flatten(), dtype=int))

        self.depth_sensors = {}
        self.depth_update_interval = -1
        self.lidar_sensors = {}
        self.lidar_update_interval = -1

        self._register_sensors()
        self._init_warp_kernel()

    def update(self, step_counter, root_pos, root_quat, reset_flag):
        def _update(interval, sensors, graph):
            if step_counter % interval == 0:
                for s in sensors:
                    s.update_sensor_pos(root_pos, root_quat)

                wp.capture_launch(graph)

                for s in sensors:
                    s.post_process()
                    s.step(reset_flag)

            else:
                for s in sensors:
                    s.step(reset_flag)

        _update(self.depth_update_interval, self.depth_sensors.values(), self.graph_depth)
        _update(self.lidar_update_interval, self.lidar_sensors.values(), self.graph_lidar)

    def get(self, sensor_name: str):
        if sensor_name.startswith('depth'):
            assert sensor_name in self.depth_sensors, f'Unknown sensor name: {sensor_name}'
            return self.depth_sensors[sensor_name].get()

        elif sensor_name.startswith('lidar'):
            assert sensor_name in self.lidar_sensors, f'Unknown sensor name: {sensor_name}'
            return self.lidar_sensors[sensor_name].get()

        else:
            raise ValueError(f'Unknown sensor type: {sensor_name}')

    def _init_warp_kernel(self):
        wp.capture_begin()
        for sensors in self.depth_sensors.values():
            sensors.launch_kernel()
        self.graph_depth = wp.capture_end()

        wp.capture_begin()
        for sensors in self.lidar_sensors.values():
            sensors.launch_kernel()
        self.graph_lidar = wp.capture_end()

    def _register_sensors(self):
        cfg_dict = class_to_dict(self.cfg.sensors)

        self._process_cfg(cfg_dict)

        for sensor_name in cfg_dict:
            if sensor_name.startswith('depth'):
                self.depth_sensors[sensor_name] = DepthCam(cfg_dict[sensor_name], self.device, self.meshes.id)

            elif sensor_name.startswith('lidar'):
                raise NotImplementedError

    def _process_cfg(self, cfg_dict: dict):
        # pre-process the configuration (fix missing keys)
        self._process_depth_cfg({k: v for k, v in cfg_dict.items() if k.startswith('depth')})

    def _process_depth_cfg(self, cfg_dict: dict):
        self.depth_update_interval = cfg_dict['depth_0']['update_interval']

        for sensor_name, sensor_cfg in cfg_dict.items():
            # check if position and angle is provided
            if not ('position' in sensor_cfg and 'pitch' in sensor_cfg):
                raise ValueError('Depth camera position and angle must be provided!')

            # check missing value, and fill with values of 0th sensor
            for cfg_element_name in ['position_range', 'pitch_range', 'update_interval', 'resolution', 'resized',
                                     'horizontal_fov', 'near_clip', 'far_clip', 'dis_noise_global', 'dis_noise_gaussian']:
                if cfg_element_name not in sensor_cfg:
                    if sensor_name == 'depth_0':
                        raise ValueError('depth_0 must provide all configuration elements!')
                    else:
                        sensor_cfg[cfg_element_name] = cfg_dict['depth_0'][cfg_element_name]

            sensor_cfg['num_envs'] = self.cfg.env.num_envs
            sensor_cfg['buf_len'] = self.cfg.env.len_depth_his

    # # Lag buffer
    # if self.cfg.depth.use_camera:
    #     self.resize_transform = torchvision.transforms.Resize(
    #         (self.cfg.depth.resized[1], self.cfg.depth.resized[0]),
    #         interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
    #         antialias=True
    #     )
    #
    #     self.depth_raw = self._zero_tensor(self.num_envs, 1, *reversed(self.cfg.depth.original))
    #     self.depth_processed = self._zero_tensor(self.num_envs, 1, *reversed(self.cfg.depth.resized))
    #     self.depth_buf = DepthBuffer(self.num_envs, self.cfg.env.len_depth_his, *reversed(self.cfg.depth.resized),
    #                                  dtype=torch.float16, device=self.device)
    #
    #     vertices = self.sim.terrain.vertices - np.array([[self.cfg.terrain.border_size, self.cfg.terrain.border_size, 0]])
    #     self.meshes = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3),
    #                           indices=wp.array(self.sim.terrain.triangles.reshape(-1), dtype=int),
    #                           velocities=None)
    #
    #     # # properties of world voxel grid
    #     # volume_array = (self.terrain.volume_array + 0.5) * self.cfg.reconstruction.grid_size
    #     # self.volume_arr_com = wp.from_numpy(volume_array, dtype=wp.vec3f, device='cuda')
    #     # self.volume_arr_occu = wp.ones((len(volume_array, )), dtype=float, device='cuda')
    #
    #     # cuda graph generated by Warp
    #     self.graph = {}
    #
    #     # depth camera properties
    #     self.sensor_pos_design = self._zero_wp_array(self.num_envs, dtype=wp.vec3f)
    #     self.sensor_quat_design = self._zero_wp_array(self.num_envs, dtype=wp.quatf)
    #     self.sensor_pos = self._zero_wp_array(self.num_envs, dtype=wp.vec3f)
    #     self.sensor_quat = self._zero_wp_array(self.num_envs, dtype=wp.quatf)
    #
    #     # buffer to store computation result
    #     width, height = self.cfg.depth.original
    #     self.cloud_depth = self._zero_tensor(self.num_envs, height, width, 3)
    #     self.cloud_depth_valid = self._zero_tensor(self.num_envs, height, width, dtype=torch.bool)
    #     #
    #     # # voxel grid output buffer
    #     # self.voxel_grid_depth = self._zero_tensor(self.num_envs, *self.cfg.reconstruction.grid_shape, 4)
    #     # self.voxel_grid_terrain = self._zero_tensor(self.num_envs, *self.cfg.reconstruction.grid_shape, 4)
    #
    #     # convert to warp for warp-torch bridging
    #     self._depth_raw_wp = wp.from_torch(self.depth_raw, dtype=wp.float32)
    #     self._cloud_depth_wp = wp.from_torch(self.cloud_depth, dtype=wp.vec3f)
    #     self._cloud_depth_valid_wp = wp.from_torch(self.cloud_depth_valid, dtype=wp.bool)
    #     # self._voxel_grid_depth_wp = wp.from_torch(self.voxel_grid_depth, dtype=wp.vec4f)
    #     # self._voxel_grid_terrain_wp = wp.from_torch(self.voxel_grid_terrain, dtype=wp.vec4f)
    #     # self._voxel_point_accumulation_wp = self._zero_wp_array(self.num_envs, *self.cfg.reconstruction.grid_shape, dtype=wp.vec4f)

    # def _launch_kernel(self, *kernel_name):
    #     for n in kernel_name:
    #         if n == 'depth':
    #             pass
    #
    #         elif n == 'depth_cloud':
    #             # initial launch, building graph
    #             wp.launch(
    #                 kernel=depth_point_cloud_kernel,
    #                 dim=(self.num_envs, *self.cfg.depth.original),
    #                 inputs=[
    #                     self.meshes.id,
    #                     self.sensor_pos_design,
    #                     self.sensor_quat_design,
    #                     self.sensor_pos,
    #                     self.sensor_quat,
    #                     self.K_inv,
    #                     self.c_x,
    #                     self.c_y,
    #                     self.cfg.depth.far_clip,
    #                     self._depth_raw_wp,
    #                     self._cloud_depth_wp,
    #                     self._cloud_depth_valid_wp
    #                 ],
    #                 device=wp.device_from_torch(self.device)
    #             )
    #
    #         elif n == 'cloud_to_voxel_grid':
    #             # prepare properties and buffers
    #             bound = self.cfg.reconstruction.ROI_depth_bound
    #             trans_voxel = wp.vec3f([bound[0][0], bound[1][0], bound[2][0]])
    #             self._voxel_point_accumulation_wp.zero_()
    #             self._voxel_grid_depth_wp.zero_()
    #
    #             wp.launch(
    #                 kernel=point_cloud_depth_to_voxel_grid_accumulation_kernel,
    #                 dim=(self.num_envs, *reversed(self.cfg.depth.original)),
    #                 inputs=[
    #                     wp.vec3i(self.cfg.reconstruction.grid_shape),
    #                     self.cfg.reconstruction.grid_size,
    #                     self.root_states[:, :3],
    #                     self.root_states[:, 3:7],
    #                     trans_voxel,
    #                     self._cloud_depth_wp,
    #                     self._cloud_depth_valid_wp,
    #                     self._voxel_point_accumulation_wp,
    #                 ],
    #                 device=wp.device_from_torch(self.device)
    #             )
    #
    #             wp.launch(
    #                 kernel=point_cloud_to_voxel_grid_kernel,
    #                 dim=(self.num_envs, *self.cfg.reconstruction.grid_shape),
    #                 inputs=[
    #                     self._voxel_point_accumulation_wp,
    #                     self._voxel_grid_depth_wp
    #                 ],
    #                 device=wp.device_from_torch(self.device)
    #             )
    #
    #         elif n == 'tri_mesh_to_voxel_grid':
    #             bound = self.cfg.reconstruction.ROI_body_bound
    #             trans_voxel = wp.vec3f([bound[0][0], bound[1][0], bound[2][0]])
    #             self._voxel_grid_terrain_wp.zero_()
    #
    #             wp.launch(
    #                 kernel=generate_voxel_grid_terrain_kernel,
    #                 dim=(self.num_envs, *self.cfg.reconstruction.grid_shape),
    #                 inputs=[
    #                     self.cfg.reconstruction.grid_size,
    #                     self.root_states[:, :3],
    #                     self.root_states[:, 3:7],
    #                     trans_voxel,
    #                     self.terrain.volume.id,
    #                     self.volume_arr_occu,
    #                     self.volume_arr_com,
    #                     self._voxel_grid_terrain_wp
    #                 ],
    #                 device=wp.device_from_torch(self.device)
    #             )
    #
    #         else:
    #             raise NotImplementedError(f'The kernel {n} is not implemented!')
