import math

import torch
import torchvision
import warp as wp

from legged_gym.utils.math import xyz_to_quat, torch_rand_float
from .sensor_base import SensorBase, SensorBuffer


class DepthCam(SensorBase):
    def __init__(self, cfg_dict, device, mesh_id, sim):
        super().__init__(cfg_dict, device, mesh_id, sim)

        self.data_format = cfg_dict['data_format']
        self.far_clip = cfg_dict['far_clip']
        self.near_clip = cfg_dict['near_clip']
        self.dis_noise_global = cfg_dict['dis_noise_global']
        self.dis_noise_gaussian = cfg_dict['dis_noise_gaussian']

        self.depth_raw = self._zero_tensor(self.num_envs, *reversed(self.cfg_dict['resolution']))

        if (self.data_format == 'cloud') or (self.data_format == 'hmap'):
            self.cloud = self._zero_tensor(self.num_envs, *reversed(self.cfg_dict['resolution']), 3)
            self._cloud_wp = wp.from_torch(self.cloud, dtype=wp.vec3f)
            self.cloud_valid = self._zero_tensor(self.num_envs, *reversed(self.cfg_dict['resolution']), dtype=torch.bool)
            self._cloud_valid_wp = wp.from_torch(self.cloud_valid, dtype=wp.bool)

            if self.data_format == 'hmap':
                x1, x2, y1, y2 = self.cfg_dict['bounding_box']
                hmap_shape = self.cfg_dict['hmap_shape']
                assert x2 > x1 and y2 > y1
                self._bounding_box_wp = wp.vec4f(x1, x2, y1, y2)
                self._hmap_grid_size_wp = wp.vec2f((x2 - x1) / hmap_shape[0], (y2 - y1) / hmap_shape[1])

                self.hmap = self._zero_tensor(self.num_envs, *hmap_shape)
                self.hmap_std = self._zero_tensor(self.num_envs, *hmap_shape)
                self._hmap_square = self._zero_tensor(self.num_envs, *hmap_shape)
                self._hmap_counter = self._zero_tensor(self.num_envs, *hmap_shape)

        self.resize_transform = torchvision.transforms.Resize(
            (self.cfg_dict['resized'][1], self.cfg_dict['resized'][0]),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True
        )

        if self.data_format == 'depth' or self.data_format == 'cloud':
            self.buf = SensorBuffer(self.num_envs,
                                    cfg_dict['buf_len'],
                                    (cfg_dict['resized'][1], cfg_dict['resized'][0]),
                                    delay_prop=cfg_dict['delay_prop'],
                                    device=device)
        elif self.data_format == 'hmap':
            self.buf = SensorBuffer(self.num_envs,
                                    cfg_dict['buf_len'],
                                    (2, *cfg_dict['hmap_shape']),
                                    delay_prop=cfg_dict['delay_prop'],
                                    device=device)

    def get(self, get_pos=False, get_depth=False, get_cloud=False, get_hmap=False, **kwargs):
        if get_pos:
            return wp.to_torch(self.sensor_pos)

        if get_depth:
            return self.depth_raw.clone()

        if get_cloud:
            return self.cloud.clone(), self.cloud_valid.clone()

        if get_hmap:
            return self.hmap, self.hmap_std

        return self.buf.get()

    def update_sensor_pos(self, links_pos: torch.Tensor, links_quat: torch.Tensor):
        super().update_sensor_pos(links_pos, links_quat)

        if self.data_format == 'hmap':
            self.hmap[:] = 0.
            self.hmap_std[:] = -1.
            self._hmap_counter[:] = 0.

    def step(self, reset):
        return self.buf.step(reset)

    def post_process(self):
        if self.data_format == 'depth' or self.data_format == 'cloud':
            # These operations are replicated on the hardware
            depth_image = self.depth_raw.clone()

            # crop 30 pixels from the left and right and 20 pixels from bottom and return cropped image
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

        elif self.data_format == 'hmap':
            self.buf.append(torch.stack([self.hmap, self.hmap_std], dim=1))
        else:
            raise NotImplementedError

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
        if self.data_format == 'depth':
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
                    self.depth_raw,
                ],
                device=wp.device_from_torch(self.device)
            )
        elif (self.data_format == 'cloud') or (self.data_format == 'hmap'):
            wp.launch(
                kernel=depth_point_cloud_kernel,
                dim=(self.num_envs, *self.cfg_dict['resolution']),
                inputs=[
                    self.mesh_id,
                    self.sensor_pos_design,
                    self.sensor_quat_design,
                    self.sensor_pos,
                    self.sensor_quat,
                    self.K_inv,
                    self.c_u,
                    self.c_v,
                    self.far_clip,
                    self.depth_raw,
                    self._cloud_wp,
                    self._cloud_valid_wp
                ],
                device=wp.device_from_torch(self.device)
            )
            if self.data_format == 'hmap':
                wp.launch(
                    kernel=cloud_to_height_map_kernel,
                    dim=(self.num_envs, *self.cfg_dict['resolution']),
                    inputs=[
                        self._bounding_box_wp,
                        self._hmap_grid_size_wp,
                        self.link_pos,
                        self.link_quat,
                        self._cloud_wp,
                        self._cloud_valid_wp,
                        self.hmap,
                        self._hmap_square,
                        self._hmap_counter,
                    ],
                    device=wp.device_from_torch(self.device)
                )

                wp.launch(
                    kernel=height_map_mean_std_kernel,
                    dim=(self.num_envs, *self.cfg_dict['hmap_shape']),
                    inputs=[
                        self.hmap,
                        self._hmap_square,
                        self._hmap_counter,
                        self.hmap_std,
                    ],
                    device=wp.device_from_torch(self.device)
                )

        else:
            raise NotImplementedError

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


@wp.kernel
def depth_only_kernel(
        mesh_id: wp.uint64,
        cam_pos_arr: wp.array1d(dtype=wp.vec3f),
        cam_quat_arr: wp.array1d(dtype=wp.quat),
        K_inv: wp.mat44,
        c_u: int,
        c_v: int,
        far_clip: float,
        depth_image: wp.array3d(dtype=float),
):
    # get the index for current pixel
    env_id, u, v = wp.tid()

    cam_pos = cam_pos_arr[env_id]
    cam_quat = cam_quat_arr[env_id]

    # obtain ray vector in image coordinate system
    cam_coords = wp.vec3f(float(u), float(v), 1.0)
    cam_coords_principal = wp.vec3f(float(c_u), float(c_v), 1.0)  # get the vector of principal axis

    # convert to camera coordinate system
    uv = wp.transform_vector(K_inv, cam_coords)
    uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis

    # convert to base frame
    uv_world = wp.vec3f(uv[2], -uv[0], -uv[1])
    uv_principal_world = wp.vec3f(uv_principal[2], -uv_principal[0], -uv_principal[1])

    # tf the direction from camera to world frame and normalize
    ray_dir = wp.normalize(wp.quat_rotate(cam_quat, uv_world))
    ray_dir_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal_world))  # ray direction of principal axis

    # multiplier to project each ray on principal axis for depth instead of range
    multiplier = wp.dot(ray_dir, ray_dir_principal)

    # perform ray casting
    query = wp.mesh_query_ray(mesh_id, cam_pos, ray_dir, far_clip / multiplier)

    dist = far_clip
    if query.result:
        # compute the depth of this pixel
        dist = multiplier * query.t

    depth_image[env_id, v, u] = dist


@wp.kernel
def depth_point_cloud_kernel(
        mesh_id: wp.uint64,
        cam_pos_design_arr: wp.array1d(dtype=wp.vec3f),  # camera position and quaterion by design
        cam_quat_design_arr: wp.array1d(dtype=wp.quat),
        cam_pos_arr: wp.array1d(dtype=wp.vec3f),  # camera position and quaterion by randomization
        cam_quat_arr: wp.array1d(dtype=wp.quat),
        K_inv: wp.mat44,
        c_u: int,
        c_v: int,
        far_clip: float,
        depth_image: wp.array3d(dtype=float),
        cloud: wp.array3d(dtype=wp.vec3f),
        cloud_valid: wp.array3d(dtype=bool)
):
    # get the index for current pixel
    env_id, u, v = wp.tid()

    cam_pos = cam_pos_arr[env_id]
    cam_quat = cam_quat_arr[env_id]

    # obtain ray vector in image coordinate system
    cam_coords = wp.vec3f(float(u), float(v), 1.0)
    cam_coords_principal = wp.vec3f(float(c_u), float(c_v), 1.0)  # get the vector of principal axis

    # convert to camera coordinate system
    uv = wp.transform_vector(K_inv, cam_coords)
    uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis

    # convert to base frame
    uv_world = wp.vec3f(uv[2], -uv[0], -uv[1])
    uv_principal_world = wp.vec3f(uv_principal[2], -uv_principal[0], -uv_principal[1])

    # tf the direction from camera to world frame and normalize
    ray_dir = wp.normalize(wp.quat_rotate(cam_quat, uv_world))
    ray_dir_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal_world))  # ray direction of principal axis

    # multiplier to project each ray on principal axis for depth instead of range
    multiplier = wp.dot(ray_dir, ray_dir_principal)

    # perform ray casting
    query = wp.mesh_query_ray(mesh_id, cam_pos, ray_dir, far_clip / multiplier)

    dist = far_clip
    cloud_valid[env_id, v, u] = query.result
    if query.result:
        # compute the depth of this pixel
        dist = multiplier * query.t

        # compute the position of the pixel in world frame
        cam_pos_design = cam_pos_design_arr[env_id]
        ray_dir_design = wp.normalize(wp.quat_rotate(cam_quat_design_arr[env_id], uv_world))

        pt_pos = cam_pos_design + ray_dir_design * query.t
        cloud[env_id, v, u] = pt_pos

    depth_image[env_id, v, u] = dist


@wp.kernel
def cloud_to_height_map_kernel(
        bbox: wp.vec4f,  # x1, x2, y1, y2, base frame
        hmap_grid_size: wp.vec2f,
        link_pos: wp.array1d(dtype=wp.vec3f),
        link_quat: wp.array1d(dtype=wp.quatf),
        cloud: wp.array3d(dtype=wp.vec3f),
        cloud_valid: wp.array3d(dtype=wp.bool),
        hmap: wp.array3d(dtype=wp.float32),
        hmap_square: wp.array3d(dtype=wp.float32),
        hmap_counter: wp.array3d(dtype=wp.float32),
):
    # get the index for current pixel
    env_id, u, v = wp.tid()

    if not cloud_valid[env_id, v, u]:
        return

    pt_pos_world = cloud[env_id, v, u]
    pt_pos_base = wp.quat_rotate_inv(link_quat[env_id], pt_pos_world - link_pos[env_id])

    if (pt_pos_base[0] > bbox[0]) and (pt_pos_base[0] < bbox[1]) and (pt_pos_base[1] > bbox[2]) and (pt_pos_base[1] < bbox[3]):
        pt_x = int(wp.floordiv(pt_pos_base[0] - bbox[0], hmap_grid_size[0]))
        pt_y = int(wp.floordiv(pt_pos_base[1] - bbox[2], hmap_grid_size[1]))

        hmap[env_id, pt_x, pt_y] += pt_pos_base[2]
        hmap_square[env_id, pt_x, pt_y] += wp.pow(pt_pos_base[2], 2.0)
        hmap_counter[env_id, pt_x, pt_y] += 1.0


@wp.kernel
def height_map_mean_std_kernel(
        hmap: wp.array3d(dtype=wp.float32),
        hmap_square: wp.array3d(dtype=wp.float32),
        hmap_counter: wp.array3d(dtype=wp.float32),
        hmap_std: wp.array3d(dtype=wp.float32),
):
    # get the index for current pixel
    env_id, i, j = wp.tid()

    if hmap_counter[env_id, i, j] > 0:
        # E[X]
        hmap[env_id, i, j] = hmap[env_id, i, j] / hmap_counter[env_id, i, j]

        # E[X^2]
        hmap_square[env_id, i, j] = hmap_square[env_id, i, j] / hmap_counter[env_id, i, j]

        # std
        hmap_std[env_id, i, j] = wp.sqrt(hmap_square[env_id, i, j] - wp.pow(hmap[env_id, i, j], 2.0))
