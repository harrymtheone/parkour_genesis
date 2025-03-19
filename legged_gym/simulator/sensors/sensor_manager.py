import torch
import warp as wp

from legged_gym.utils.helpers import class_to_dict
from .depth_cam import DepthCam


class SensorManager:
    def __init__(self, cfg, simulator, device: torch.device):
        self.cfg = cfg
        self.simulator = simulator
        self.device = device

        if device.type == 'cuda':
            # vertices (n_vert, 3), triangles (n_tri)
            vertices, triangles = self.simulator.get_trimesh()
            self.meshes = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3), indices=wp.array(triangles.flatten(), dtype=int))

        self.depth_sensors = {}
        self.depth_update_interval = -1
        self.lidar_sensors = {}
        self.lidar_update_interval = -1

        self._register_sensors()

        if device.type == 'cuda':
            self._init_warp_kernel()

    def update(self, step_counter, links_pos, links_quat, reset_flag):
        if self.device.type == 'cpu':
            # IsaacGym builtin sensors
            if step_counter % self.depth_update_interval == 0:
                for s in self.depth_sensors.values():
                    s.update()
                    s.post_process()
                    s.step(reset_flag)
            else:
                for s in self.depth_sensors.values():
                    s.step(reset_flag)

        else:
            # Warp sensors
            def _update(interval, sensors, graph):
                if step_counter % interval == 0:
                    for s in sensors:
                        s.update_sensor_pos(links_pos, links_quat)

                    wp.capture_launch(graph)

                    for s in sensors:
                        s.post_process()
                        s.step(reset_flag)

                else:
                    for s in sensors:
                        s.step(reset_flag)

            _update(self.depth_update_interval, self.depth_sensors.values(), self.graph_depth)
            _update(self.lidar_update_interval, self.lidar_sensors.values(), self.graph_lidar)

    def get(self, sensor_name: str, **kwargs):
        if sensor_name.startswith('depth'):
            assert sensor_name in self.depth_sensors, f'Unknown sensor name: {sensor_name}'
            return self.depth_sensors[sensor_name].get(**kwargs)

        elif sensor_name.startswith('lidar'):
            assert sensor_name in self.lidar_sensors, f'Unknown sensor name: {sensor_name}'
            return self.lidar_sensors[sensor_name].get(**kwargs)

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
            if sensor_name.startswith('depth') and self.device.type == 'cpu':
                from .depth_cam_isaacgym import DepthCamIsaacGym
                self.depth_sensors[sensor_name] = DepthCamIsaacGym(cfg_dict[sensor_name], self.device, self.simulator)

            elif sensor_name.startswith('depth') and self.device.type == 'cuda':
                self.depth_sensors[sensor_name] = DepthCam(cfg_dict[sensor_name], self.device, self.meshes.id, self.simulator)

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
                        raise ValueError(f'depth_0 must provide all configuration elements! Missing {cfg_element_name}')
                    else:
                        sensor_cfg[cfg_element_name] = cfg_dict['depth_0'][cfg_element_name]

            sensor_cfg['num_envs'] = self.cfg.env.num_envs
            sensor_cfg['buf_len'] = self.cfg.env.len_depth_his
