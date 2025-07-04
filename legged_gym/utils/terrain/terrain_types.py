import random
from enum import Enum
from typing import Tuple

from .terrain_utils import SubTerrain, random_uniform_terrain, pyramid_stairs_terrain, parkour_stair_terrain, parkour_flat_terrain, parkour_mini_stair_terrain
from .utils import generate_fractal_noise_2d


class TerrainType(Enum):
    smooth_slope = 0
    rough_slope = 1
    stairs_up = 2
    stairs_down = 3
    discrete = 4
    stepping_stone = 5
    gap = 6
    pit = 7
    parkour = 8
    parkour_gap = 9
    parkour_box = 10
    parkour_step = 11
    parkour_stair = 12
    parkour_mini_stair = 13
    parkour_flat = 14


class BaseTerrain:
    size: Tuple[float, float]

    def __init__(self, name, vertical_scale, horizontal_scale):
        self.vertical_scale = vertical_scale

        pixel_length = int(self.size[0] / horizontal_scale)
        pixel_width = int(self.size[1] / horizontal_scale)

        self.terrain = SubTerrain(
            name,
            width=pixel_length,
            length=pixel_width,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale
        )

    def add_roughness(self, difficulty=1., roughness_height=(0.0, 0.03), downsampled_scale=0.075):
        max_height = (roughness_height[1] - roughness_height[0]) * difficulty + roughness_height[0]
        random_uniform_terrain(self.terrain,
                               min_height=-max_height,
                               max_height=max_height,
                               step=0.005,
                               downsampled_scale=downsampled_scale)

    def add_fractal_roughness(self, terrain, difficulty=1):
        # heightfield_noise = generate_fractal_noise_2d(
        #     xSize=int(terrain.width * self.cfg.horizontal_scale),
        #     ySize=int(terrain.length * self.cfg.horizontal_scale),
        #     xSamples=terrain.width,
        #     ySamples=terrain.length,
        #     zScale=0.08 + 0.07 * difficulty,  # 0.08, 0.15
        #     frequency=10,
        # ) / self.cfg.vertical_scale

        heightfield_noise = generate_fractal_noise_2d(
            (terrain.width, terrain.length),
            terrain.horizontal_scale,
            difficulty
        ) / self.vertical_scale

        terrain.height_field_raw[:] += heightfield_noise


class SmoothSlope(BaseTerrain):
    size = (8, 8)

    def generate(self, difficulty):
        slope = difficulty * 0.4

        # if choice < self.proportions[0] / 2:
        #     slope *= -1
        # terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        # add_fractal_roughness(terrain, scale=5)
        # self.add_roughness(terrain, difficulty)

        self.terrain.terrain_type = TerrainType.smooth_slope
        return self.terrain


class RoughSlope(BaseTerrain):
    size = (8, 8)

    def generate(self, difficulty):
        slope = difficulty * 0.4

        # self.terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        # random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        self.add_fractal_roughness(self.terrain, difficulty)
        # self.add_roughness(terrain, difficulty)

        self.terrain.terrain_type = TerrainType.rough_slope
        return self.terrain


class PyramidStairsUp(BaseTerrain):
    size = (8, 8)

    def generate(self, difficulty):
        stair_up_height = 0.02 + 0.08 * difficulty

        pyramid_stairs_terrain(self.terrain, step_width=0.31, step_height=-stair_up_height, platform_size=3.)
        # self.add_roughness(terrain, difficulty)

        self.terrain.terrain_type = TerrainType.stairs_up
        return self.terrain


class PyramidStairsDown(BaseTerrain):
    size = (8, 8)

    def generate(self, difficulty):
        stair_down_height = 0.02 + 0.08 * difficulty

        pyramid_stairs_terrain(self.terrain, step_width=0.31, step_height=stair_down_height, platform_size=3.)

        # self.add_roughness(terrain, difficulty)

        self.terrain.terrain_type = TerrainType.stairs_down
        return self.terrain


class ParkourStair(BaseTerrain):
    size = (18, 4)

    def generate(self, difficulty):
        stair_height_goal = 0.05 + 0.10 * difficulty

        parkour_stair_terrain(self.terrain,
                              step_height=stair_height_goal,
                              step_depth=random.uniform(0.25, 0.35))  # 0.31
        # self.add_roughness(terrain, 0.5)

        self.terrain.terrain_type = TerrainType.parkour_stair
        self.terrain.centered_origin = False
        return self.terrain


class ParkourMiniStair(BaseTerrain):
    size = (18, 4)

    def generate(self, difficulty):
        stair_height_goal = 0.05 + 0.10 * difficulty

        parkour_mini_stair_terrain(self.terrain,
                                   step_height=stair_height_goal,
                                   # step_depth=random.uniform(0.25, 0.35))  # 0.31
                                   step_depth=0.31)  # 0.31
        self.add_roughness(difficulty)

        self.terrain.terrain_type = TerrainType.parkour_mini_stair
        self.terrain.centered_origin = False
        return self.terrain


class ParkourFlat(BaseTerrain):
    size = (18, 4)

    def generate(self, difficulty):
        parkour_flat_terrain(self.terrain)
        self.add_roughness(difficulty)

        self.terrain.terrain_type = TerrainType.parkour_flat
        self.terrain.centered_origin = False
        return self.terrain
