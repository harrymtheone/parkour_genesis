import math

import torch
import warp as wp

from .sensor_base import SensorBase


class LiDAR(SensorBase):
    def __init__(self, cfg_dict, device, mesh_id):
        super().__init__(cfg_dict, device, mesh_id)

        self.num_scan_line = cfg_dict['num_scan_lines']
        self.num_points_per_line = cfg_dict['sample_freq'] / cfg_dict['rotation_freq']

        self._initialize_ray_vectors()

    def _initialize_ray_vectors(self):
        # populate a 2D torch array with the ray vectors that are 2d arrays of wp.vec3
        ray_vectors = torch.zeros(
            (self.num_scan_lines, self.num_points_per_line, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_scan_lines):
            for j in range(self.num_points_per_line):
                # Rays go from +HFoV/2 to -HFoV/2 and +VFoV/2 to -VFoV/2
                azimuth_angle = self.horizontal_fov_max - (
                        self.horizontal_fov_max - self.horizontal_fov_min
                ) * (j / (self.num_points_per_line - 1))
                elevation_angle = self.vertical_fov_max - (
                        self.vertical_fov_max - self.vertical_fov_min
                ) * (i / (self.num_scan_lines - 1))
                ray_vectors[i, j, 0] = math.cos(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 1] = math.sin(azimuth_angle) * math.cos(elevation_angle)
                ray_vectors[i, j, 2] = math.sin(elevation_angle)
        # normalize ray_vectors
        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)

        # recast as 2D warp array of vec3
        self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)
