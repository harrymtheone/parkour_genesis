import math

import torch
import torchvision
import warp as wp

from legged_gym.utils.math import xyz_to_quat, torch_rand_float
from .sensor_base import SensorBase, SensorBuffer
from .warp_kernel import depth_only_kernel


class DepthCam(SensorBase):
    def __init__(self, cfg_dict, device, mesh_id):
        super().__init__(cfg_dict, device, mesh_id)

        self.far_clip = cfg_dict['far_clip']
        self.near_clip = cfg_dict['near_clip']
        self.dis_noise_global = cfg_dict['dis_noise_global']
        self.dis_noise_gaussian = cfg_dict['dis_noise_gaussian']

        self.depth_raw = self._zero_tensor(self.num_envs, *reversed(self.cfg_dict['resolution']))
        self._depth_raw_wp = wp.from_torch(self.depth_raw, dtype=wp.float32)

        self.resize_transform = torchvision.transforms.Resize(
            (self.cfg_dict['resized'][1], self.cfg_dict['resized'][0]),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True
        )

        self.buf = SensorBuffer(self.num_envs,
                                cfg_dict['buf_len'],
                                (cfg_dict['resized'][1], cfg_dict['resized'][0]),
                                delay_prop=cfg_dict['delay_prop'],
                                device=device)

    def get(self, get_raw=False, **kwargs):
        if get_raw:
            return self.depth_raw.clone()
        else:
            return self.buf.get()

    def step(self, reset):
        return self.buf.step(reset)

    def _initialize_sensors(self):
        # camera properties
        cfg = self.cfg_dict
        width, height = cfg['resolution']
        horizontal_fov = cfg['horizontal_fov']

        self.sensor_offset_pos_design[:] = torch.tensor(cfg['position'], dtype=torch.float, device=self.device).unsqueeze(0)
        self.sensor_offset_quat_design[:] = xyz_to_quat(  # for point cloud computation
            torch.deg2rad(torch.tensor([[0, cfg['pitch'], 0]], dtype=torch.float, device=self.device)))

        # camera randomization
        self.sensor_offset_pos[:] = self.sensor_offset_pos_design
        self.sensor_offset_pos[:, 0:1] += torch_rand_float(*cfg['position_range'][0], shape=(self.num_envs, 1), device=self.device)
        self.sensor_offset_pos[:, 1:2] += torch_rand_float(*cfg['position_range'][1], shape=(self.num_envs, 1), device=self.device)
        self.sensor_offset_pos[:, 2:3] += torch_rand_float(*cfg['position_range'][2], shape=(self.num_envs, 1), device=self.device)

        camera_xyz_angle = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        camera_xyz_angle[:, 1:2] = cfg['pitch'] + torch_rand_float(*cfg['pitch_range'], shape=(self.num_envs, 1), device=self.device)
        self.sensor_offset_quat[:] = xyz_to_quat(torch.deg2rad(camera_xyz_angle))

        u_0, v_0 = width / 2, height / 2
        f = u_0 / math.tan(math.radians(horizontal_fov) / 2)

        # simple pinhole model
        K = wp.mat44(
            f, 0.0, u_0, 0.0,
            0.0, f, v_0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        self.K_inv = wp.inverse(K)
        self.c_u, self.c_v = int(u_0), int(v_0)

    def launch_kernel(self):
        wp.launch(
            kernel=depth_only_kernel,
            dim=(self.num_envs, *self.cfg_dict['resolution']),
            inputs=[
                self.mesh_id,
                self.sensor_pos,
                self.sensor_quat,
                self.K_inv,
                self.c_u,
                self.c_v,
                self.far_clip,
                self._depth_raw_wp,
            ],
            device=wp.device_from_torch(self.device)
        )

    def post_process(self):
        # These operations are replicated on the hardware
        depth_image = self.depth_raw.clone()

        # crop 30 pixels from the left and right and 20 pixels from bottom and return croped image
        depth_image = depth_image[:, :-2, 4:-4]

        # add global distance noise
        depth_image[:] += torch_rand_float(-self.dis_noise_global, self.dis_noise_global, (self.num_envs, 1), self.device).unsqueeze(-1)

        # add Gaussian noise
        depth_image += torch.randn_like(depth_image) * self.dis_noise_gaussian

        # distance clip
        depth_image[:] = torch.clip(depth_image, self.near_clip, self.far_clip)

        # resize image
        depth_image = self.resize_transform(depth_image)

        # normalize the depth image to range (-0.5, 0.5)
        depth_image[:] = (depth_image - self.near_clip) / (self.far_clip - self.near_clip) - 0.5

        # self.depth_processed[:] = depth_image
        self.buf.append(depth_image)

    # def _process_voxel_grid(self):
    #     def process(voxel: torch.Tensor, noise_level):
    #         occu = voxel[..., 0:1] > 0
    #
    #         if noise_level == 0:
    #             # normalize com to (-1, 1) and set com of unoccupied voxel to zero
    #             voxel[..., 1:] = torch.where(occu, voxel[..., 1:] - 0.5, 0)
    #         else:
    #             # and add noise to each occupied voxel
    #             noise = (torch.rand_like(voxel[..., 1:]) - 0.5) * noise_level
    #             voxel[..., 1:] = torch.where(occu, voxel[..., 1:] - 0.5 + noise, 0)
    #
    #     if self.global_counter % self.cfg.depth.update_interval == 0:
    #         process(self.voxel_grid_depth, self.cfg.reconstruction.voxel_noise / self.cfg.reconstruction.grid_size)
    #
    #     if self.cfg.reconstruction.recon_each_step or (self.global_counter % self.cfg.depth.update_interval == 0):
    #         process(self.voxel_grid_terrain, 0)
