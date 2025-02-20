import random
from typing import List

import torch
import warp as wp

from legged_gym.utils.math import transform_by_trans_quat, transform_quat_by_quat


class SensorBase:
    def __init__(self, cfg_dict, device, mesh_id):
        self.cfg_dict = cfg_dict
        self.num_envs = cfg_dict['num_envs']
        self.device = device
        self.mesh_id = mesh_id

        if self.device.type != 'cuda':
            raise ValueError("Only cuda rendering is currently supported!")

        # offset from base
        self.sensor_offset_pos_design = torch.zeros(self.num_envs, 3, dtype=torch.float, device=device)
        self.sensor_offset_quat_design = torch.zeros(self.num_envs, 4, dtype=torch.float, device=device)
        self.sensor_offset_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=device)
        self.sensor_offset_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=device)

        # position in world frame
        self.sensor_pos_design = self._zero_wp_array(self.num_envs, dtype=wp.vec3f)
        self.sensor_quat_design = self._zero_wp_array(self.num_envs, dtype=wp.quatf)
        self.sensor_pos = self._zero_wp_array(self.num_envs, dtype=wp.vec3f)
        self.sensor_quat = self._zero_wp_array(self.num_envs, dtype=wp.quatf)

        self._initialize_sensors()

    def get(self):
        raise NotImplementedError

    def update_sensor_pos(self, root_pos, root_quat):
        # update depth camera and convert to voxel grid
        self.sensor_pos_design.assign(wp.from_torch(
            transform_by_trans_quat(self.sensor_offset_pos_design, root_pos, root_quat), dtype=wp.vec3f))
        self.sensor_quat_design.assign(wp.from_torch(
            transform_quat_by_quat(root_quat, self.sensor_offset_quat_design), dtype=wp.quatf))
        self.sensor_pos.assign(wp.from_torch(
            transform_by_trans_quat(self.sensor_offset_pos, root_pos, root_quat), dtype=wp.vec3f))
        self.sensor_quat.assign(wp.from_torch(
            transform_quat_by_quat(root_quat, self.sensor_offset_quat), dtype=wp.quatf))

    def launch_kernel(self):
        raise NotImplementedError

    def post_process(self, reset_flag):
        raise NotImplementedError

    def _initialize_sensors(self):
        raise NotImplementedError

    def _zero_tensor(self, *shape, dtype=torch.float, requires_grad=False):
        return torch.zeros(*shape, dtype=dtype, device=self.device, requires_grad=requires_grad)

    def _zero_wp_array(self, *shape, dtype=None, device=None):
        if dtype is None:
            raise ValueError('dtype cannot be None!')

        if device is None:
            device = wp.device_from_torch(self.device)

        return wp.zeros(shape, dtype=dtype, device=device)


class SensorBuffer:
    def __init__(self, n_envs, his_len, data_shape, delay_prop=None, dtype=None, device=None):
        assert device is not None, 'please define the buffer device!'

        self.buf_len = his_len
        self.delay_prop = delay_prop
        dtype = torch.float32 if dtype is None else dtype
        self.buf = torch.zeros(n_envs, self.buf_len, *data_shape, dtype=dtype, device=device)

        self.delay_data_buf: List[torch.Tensor] = []
        self.delay_buf: List[int] = []

    def append(self, data):
        if self.delay_prop is None:
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = data
            return

        delay = round(random.gauss(*self.delay_prop))

        while (len(self.delay_buf) > 0) and (delay <= self.delay_buf[-1]):
            self.delay_data_buf.pop(-1)
            self.delay_buf.pop(-1)

        self.delay_data_buf.append(data.clone())
        self.delay_buf.append(delay + 1)

    def step(self, reset):
        if self.delay_prop is None:
            return

        if len(self.delay_buf) == 0:
            return

        for i, v in enumerate(self.delay_buf):
            self.delay_buf[i] = v - 1

        if self.delay_buf[0] == 0:
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = self.delay_data_buf[0]
            self.delay_buf.pop(0)
            self.delay_data_buf.pop(0)

        self.buf[reset] = 0.

    def get(self):
        return self.buf.clone()
