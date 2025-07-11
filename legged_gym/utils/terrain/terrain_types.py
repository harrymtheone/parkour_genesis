import random
from enum import Enum
from typing import Tuple

import numpy as np

from .terrain_utils import random_uniform_terrain
from .utils import generate_fractal_noise_2d


class TerrainType(Enum):
    smooth_slope = 0
    rough_slope = 1
    stairs_up = 2
    stairs_down = 3
    huge_stair = 4
    discrete = 5
    stepping_stone = 6
    gap = 7
    pit = 8

    parkour_flat = 100
    parkour = 101
    parkour_gap = 102
    parkour_box = 103
    parkour_step = 104
    parkour_stair = 105
    parkour_stair_down = 106
    parkour_mini_stair = 107
    parkour_go_back_stair = 108


class SubTerrain:
    # terrain_name: str
    size: Tuple[float, float]
    horizontal_scale: float = 0.02
    vertical_scale: float = 0.005

    terrain_type: TerrainType
    centered_origin: bool

    def __init__(self):
        self.width = int(self.size[0] / self.horizontal_scale)
        self.length = int(self.size[1] / self.horizontal_scale)

        self.height_field_raw = np.zeros((self.width, self.length), dtype=float)

    def make(self, difficulty):
        raise NotImplementedError

    def add_roughness(self, difficulty=1., roughness_height=(0.0, 0.03), downsampled_scale=0.075):
        max_height = (roughness_height[1] - roughness_height[0]) * difficulty + roughness_height[0]
        random_uniform_terrain(self,
                               min_height=-max_height,
                               max_height=max_height,
                               step=0.005,
                               downsampled_scale=downsampled_scale)

    def add_fractal_roughness(self, difficulty=1):
        # heightfield_noise = generate_fractal_noise_2d(
        #     xSize=int(terrain.width * self.cfg.horizontal_scale),
        #     ySize=int(terrain.length * self.cfg.horizontal_scale),
        #     xSamples=terrain.width,
        #     ySamples=terrain.length,
        #     zScale=0.08 + 0.07 * difficulty,  # 0.08, 0.15
        #     frequency=10,
        # ) / self.cfg.vertical_scale

        heightfield_noise = generate_fractal_noise_2d(
            (self.width, self.length),
            self.horizontal_scale,
            difficulty
        ) / self.vertical_scale

        self.height_field_raw[:] += heightfield_noise


class SmoothSlope(SubTerrain):
    size = (8, 8)
    terrain_type = TerrainType.smooth_slope
    centered_origin = True

    def make(self, difficulty):
        slope = difficulty * 0.4

        # if choice < self.proportions[0] / 2:
        #     slope *= -1
        # terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        # add_fractal_roughness(terrain, scale=5)
        # self.add_roughness(terrain, difficulty)


class RoughSlope(SubTerrain):
    size = (8, 8)
    terrain_type = TerrainType.rough_slope
    centered_origin = True

    def make(self, difficulty):
        slope = difficulty * 0.4

        # self.terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        # random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        self.add_fractal_roughness(difficulty)
        # self.add_roughness(terrain, difficulty)


class PyramidStairsUp(SubTerrain):
    size = (8, 8)
    terrain_type = TerrainType.stairs_up
    centered_origin = True

    def make(self, difficulty):
        stair_up_height = 0.02 + 0.08 * difficulty

        self.pyramid_stairs_terrain(step_width=0.31, step_height=-stair_up_height, platform_size=3.)
        # self.add_roughness(terrain, difficulty)

    def pyramid_stairs_terrain(self, step_width, step_height, platform_size=1.):
        # switch parameters to discrete units
        step_width = int(step_width / self.horizontal_scale)
        step_height = int(step_height / self.vertical_scale)
        platform_size = int(platform_size / self.horizontal_scale)

        height = 0
        start_x = 0
        stop_x = self.width
        start_y = 0
        stop_y = self.length

        while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
            start_x += step_width
            stop_x -= step_width
            start_y += step_width
            stop_y -= step_width
            height += step_height
            self.height_field_raw[start_x: stop_x, start_y: stop_y] = height


class PyramidStairsDown(PyramidStairsUp):
    size = (8, 8)
    terrain_type = TerrainType.stairs_down
    centered_origin = True

    def make(self, difficulty):
        stair_up_height = 0.02 + 0.08 * difficulty

        self.pyramid_stairs_terrain(step_width=0.31, step_height=-stair_up_height, platform_size=3.)
        # self.add_roughness(terrain, difficulty)


class HugeStair(SubTerrain):
    size = (8, 8)
    terrain_type = TerrainType.huge_stair
    centered_origin = True

    def make(self, difficulty):
        stair_down_height = 0.02 + 0.08 * difficulty

        self._make_huge_stair(step_height=stair_down_height, step_depth=0.31)
        self.add_roughness(difficulty)

    def _make_huge_stair(self, step_height, step_depth, padding_size=1.):
        step_height_length = int(step_height / self.vertical_scale)
        step_depth_length = int(step_depth / self.horizontal_scale)
        padding_length = int(padding_size / self.horizontal_scale)

        cur_x = padding_length
        cur_height = 0.

        while cur_x < self.width - padding_length:
            self.height_field_raw[int(cur_x): int(cur_x + step_depth_length), padding_length: -padding_length] = cur_height

            cur_x = cur_x + step_depth_length
            cur_height += step_height_length


# ########################################################################################
# ################################# Goal Guided Terrains #################################
# ########################################################################################


class GoalGuidedTerrain(SubTerrain):
    goals: np.ndarray


class ParkourFlat(GoalGuidedTerrain):
    size = (18, 4)
    terrain_type = TerrainType.parkour_flat
    centered_origin = False

    def make(self, difficulty):
        self.parkour_flat_terrain()
        self.add_roughness(difficulty)

    def parkour_flat_terrain(
            self,
            platform_len=2.,
            x_range=(1.5, 2.4),
            y_range=(-1., 1.)
    ):
        mid_y = self.length // 2  # length is actually y width
        platform_len = round(platform_len / self.horizontal_scale)

        dis_x_min = round(x_range[0] / self.horizontal_scale)
        dis_x_max = round(x_range[1] / self.horizontal_scale)
        dis_y_min = round(y_range[0] / self.horizontal_scale)
        dis_y_max = round(y_range[1] / self.horizontal_scale)

        cur_x = platform_len
        goals = np.zeros((8, 2))
        goals[0] = [platform_len, mid_y]

        for i in range(6):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            cur_x += rand_x
            goals[i + 1] = [cur_x, mid_y + rand_y]

        goals[-1] = [self.width - platform_len // 2, mid_y]
        self.goals = goals * self.horizontal_scale


class ParkourStair(GoalGuidedTerrain):
    size = (18, 4)
    terrain_type = TerrainType.parkour_stair
    centered_origin = False

    def make(self, difficulty):
        stair_height_goal = 0.05 + 0.10 * difficulty

        self.parkour_stair_terrain(
            step_height=stair_height_goal,
            step_width=random.uniform(0.5, 2.0),
            # step_width=3.0,
            step_depth=random.uniform(0.23, 0.35)  # 0.31
        )
        # self.add_roughness(terrain, 0.5)

    def parkour_stair_terrain(
            self,
            step_height,
            step_width,
            step_depth,
            platform_len=2.5,
            goal_deviation=0.3,
    ):
        mid_y = self.length // 2  # length is actually y width

        step_height = round(step_height / self.vertical_scale)
        step_width = round(step_width / self.horizontal_scale)
        step_depth = round(step_depth / self.horizontal_scale)
        platform_len = round(platform_len / self.horizontal_scale)

        dis_x = platform_len
        cur_height = 0

        while dis_x < self.width - platform_len:
            cur_height += step_height
            self.height_field_raw[dis_x:dis_x + step_depth] = cur_height
            dis_x += step_depth

        half_pad_width = int(mid_y - step_width / 2)
        self.height_field_raw[:, :half_pad_width] = round(-20. / self.vertical_scale)
        self.height_field_raw[:, -half_pad_width:] = round(-20. / self.vertical_scale)

        def rand_deviation():
            return round(random.uniform(-goal_deviation, goal_deviation) / self.horizontal_scale)

        goals = np.zeros((4, 2))
        end_x = dis_x - round(1.0 / self.horizontal_scale)
        step_x = int((end_x - platform_len) / 4)

        for i in range(4 - 1):
            goals[i] = [platform_len + (i + 1) * step_x, mid_y + rand_deviation()]

        goals[-1] = [end_x, mid_y]

        self.goals = goals * self.horizontal_scale


class ParkourStairDown(GoalGuidedTerrain):
    size = (18, 4)
    terrain_type = TerrainType.parkour_stair_down
    centered_origin = False

    def make(self, difficulty):
        stair_height_goal = 0.05 + 0.04 * difficulty

        self.parkour_stair_terrain(
            step_height=stair_height_goal,
            step_width_range=(0.5, 2.0),
            step_depth=random.uniform(0.23, 0.35)
        )
        self.add_roughness(0.5)

    def parkour_stair_terrain(
            self,
            step_height,
            step_width_range,
            step_depth,
            platform_len=2.5,
            goal_deviation=0.2,
    ):
        mid_y = self.length // 2  # length is actually y width

        step_height = round(step_height / self.vertical_scale)
        step_depth = round(step_depth / self.horizontal_scale)
        platform_len = round(platform_len / self.horizontal_scale)

        def rand_deviation():
            return round(random.uniform(-goal_deviation, goal_deviation) / self.horizontal_scale)

        goals = np.zeros((5, 2))

        start_x = platform_len
        self.height_field_raw[start_x: start_x + 150] = self.gen_mini_stair(step_height, step_width_range, step_depth)
        goals[0] = [start_x + 75, mid_y + rand_deviation()]

        start_x = start_x + 150 + 50
        self.height_field_raw[start_x: start_x + 150] = self.gen_mini_stair(step_height, step_width_range, step_depth)
        goals[1] = [start_x + 75, mid_y + rand_deviation()]

        start_x = start_x + 150 + 50
        self.height_field_raw[start_x: start_x + 150] = self.gen_mini_stair(step_height, step_width_range, step_depth)
        goals[2] = [start_x + 75, mid_y + rand_deviation()]

        start_x = start_x + 150 + 50
        self.height_field_raw[start_x: start_x + 150] = self.gen_mini_stair(step_height, step_width_range, step_depth)
        goals[3] = [start_x + 75, mid_y + rand_deviation()]

        start_x = start_x + 150
        goals[4] = [start_x + 75, mid_y + rand_deviation()]

        self.goals = goals * self.horizontal_scale

    def gen_mini_stair(self, step_height, step_width_range, step_depth):
        mid_y = self.length // 2  # length is actually y width
        half_step_width = round(random.uniform(*step_width_range) / self.horizontal_scale / 2)
        y_slice = slice(mid_y - half_step_width, mid_y + half_step_width)

        mini_stair_hf = np.zeros((150, self.length))

        # a slope
        mini_stair_hf[:3 * step_depth, y_slice] = np.tile(
            np.linspace(0, 4 * step_height, 3 * step_depth),
            (2 * half_step_width, 1)
        ).T

        # down stairs
        mini_stair_hf[3 * step_depth: -3 * step_depth, y_slice] = 4 * step_height
        mini_stair_hf[-3 * step_depth: -2 * step_depth, y_slice] = 3 * step_height
        mini_stair_hf[-2 * step_depth: -step_depth, y_slice] = 2 * step_height
        mini_stair_hf[-step_depth:, y_slice] = step_height
        return mini_stair_hf


class ParkourMiniStair(GoalGuidedTerrain):
    size = (18, 4)
    terrain_type = TerrainType.parkour_mini_stair
    centered_origin = False

    def make(self, difficulty):
        stair_height_goal = 0.05 + 0.10 * difficulty

        self.parkour_mini_stair_terrain(
            step_height=stair_height_goal,
            step_width_range=(0.5, 2.0),
            step_depth=random.uniform(0.25, 0.35),
            # step_depth=0.31
        )
        self.add_roughness(difficulty)

    def parkour_mini_stair_terrain(
            self,
            step_height,
            step_width_range,
            step_depth,
            platform_len=2.,
            goal_deviation=0.2,
    ):
        mid_y = self.length // 2  # length is actually y width

        step_height = round(step_height / self.vertical_scale)
        step_depth = round(step_depth / self.horizontal_scale)
        platform_len = round(platform_len / self.horizontal_scale)

        def rand_deviation():
            return round(random.uniform(-goal_deviation, goal_deviation) / self.horizontal_scale)

        goals = np.zeros((5, 2))

        start_x = platform_len
        self.height_field_raw[start_x: start_x + 150] = self.gen_mini_stair(step_height, step_width_range, step_depth)
        goals[0] = [start_x + 75, mid_y + rand_deviation()]

        start_x = start_x + 150 + 50
        self.height_field_raw[start_x: start_x + 150] = self.gen_mini_stair(step_height, step_width_range, step_depth)
        goals[1] = [start_x + 75, mid_y + rand_deviation()]

        start_x = start_x + 150 + 50
        self.height_field_raw[start_x: start_x + 150] = self.gen_mini_stair(step_height, step_width_range, step_depth)
        goals[2] = [start_x + 75, mid_y + rand_deviation()]

        start_x = start_x + 150 + 50
        self.height_field_raw[start_x: start_x + 150] = self.gen_mini_stair(step_height, step_width_range, step_depth)
        goals[3] = [start_x + 75, mid_y + rand_deviation()]

        start_x = start_x + 150
        goals[4] = [start_x + 75, mid_y + rand_deviation()]

        self.goals = goals * self.horizontal_scale

    def gen_mini_stair(self, step_height, step_width_range, step_depth):
        mid_y = self.length // 2  # length is actually y width
        half_step_width = round(random.uniform(*step_width_range) / self.horizontal_scale / 2)
        y_slice = slice(mid_y - half_step_width, mid_y + half_step_width)

        mini_stair_hf = np.zeros((150, self.length))
        mini_stair_hf[:step_depth, y_slice] = step_height
        mini_stair_hf[step_depth: 2 * step_depth, y_slice] = 2 * step_height
        mini_stair_hf[2 * step_depth: 3 * step_depth, y_slice] = 3 * step_height
        mini_stair_hf[3 * step_depth: -3 * step_depth, y_slice] = 4 * step_height
        mini_stair_hf[-3 * step_depth: -2 * step_depth, y_slice] = 3 * step_height
        mini_stair_hf[-2 * step_depth: -step_depth, y_slice] = 2 * step_height
        mini_stair_hf[-step_depth:, y_slice] = step_height
        return mini_stair_hf


class ParkourGoBackStair(GoalGuidedTerrain):
    size = (18, 4)
    terrain_type = TerrainType.parkour_go_back_stair
    centered_origin = False

    def make(self, difficulty):
        stair_height_goal = 0.05 + 0.1 * difficulty

        self.parkour_stair_terrain(
            step_height=stair_height_goal,
            step_width=random.uniform(0.5, 2.0),
            step_depth=random.uniform(0.23, 0.35)
        )
        # self.add_roughness(terrain, 0.5)

    def parkour_stair_terrain(
            self,
            step_height,
            step_width,
            step_depth,
            platform_len=2.5,
    ):
        mid_y = self.length // 2  # length is actually y width

        step_height = round(step_height / self.vertical_scale)
        step_width = round(step_width / self.horizontal_scale)
        step_depth = round(step_depth / self.horizontal_scale)
        platform_len = round(platform_len / self.horizontal_scale)

        dis_x = platform_len
        cur_height = 0

        while dis_x < self.width // 3:
            cur_height += step_height
            self.height_field_raw[dis_x:dis_x + step_depth] = cur_height
            dis_x += step_depth

        half_pad_width = int(mid_y - step_width / 2)
        self.height_field_raw[:, :half_pad_width] = round(-20. / self.vertical_scale)
        self.height_field_raw[:, -half_pad_width:] = round(-20. / self.vertical_scale)
        self.height_field_raw[dis_x: dis_x + platform_len] = cur_height

        goals = np.zeros((3, 2))
        goals[0] = [dis_x + platform_len // 2, mid_y]
        goals[1] = [dis_x + platform_len // 2 - 50, mid_y + random.choice([-1, 1]) * 10]
        goals[2] = [platform_len // 2, mid_y]
        self.goals = goals * self.horizontal_scale
