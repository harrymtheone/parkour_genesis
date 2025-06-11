import random

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.width = width
        self.length = length
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.height_field_raw = np.zeros((self.width, self.length), dtype=float)
        self.height_field_guidance = None

        self.row = None
        self.col = None
        self.terrain_type = None
        self.centered_origin = True
        self.goals = None


def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None, ):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(
        heights_range,
        (
            int(terrain.width * terrain.horizontal_scale / downsampled_scale),
            int(terrain.length * terrain.horizontal_scale / downsampled_scale),
        ),
    )

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    interpolator = RegularGridInterpolator((x, y), height_field_downsampled, method='linear', bounds_error=False, fill_value=None)

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)

    xv, yv = np.meshgrid(x_upsampled, y_upsampled, indexing='ij')

    points_to_interpolate = np.stack((xv.ravel(), yv.ravel()), axis=-1)
    z_upsampled = np.rint(interpolator(points_to_interpolate).reshape(xv.shape))

    terrain.height_field_raw += z_upsampled.astype(np.int16)
    return terrain


def pyramid_stairs_terrain(terrain: SubTerrain, step_width, step_height, platform_size=1.):
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height

    return terrain


def gap_terrain(terrain: SubTerrain, gap_size, outer_platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    outer_platform_size = int(outer_platform_size / terrain.horizontal_scale)

    center_x = terrain.width // 2
    center_y = terrain.length // 2
    x1 = (terrain.width - outer_platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.length - outer_platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -200
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0

    goals = np.zeros((2, 2))
    goals[0] = [terrain.width // 2, terrain.length // 2]
    goals[1] = [x2 + 1.0 / terrain.horizontal_scale, terrain.length // 2]
    terrain.goals = goals * terrain.horizontal_scale


def pit_terrain(terrain: SubTerrain, depth, bottom_size=1.):
    depth = int(depth / terrain.vertical_scale)
    bottom_size = int(bottom_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - bottom_size
    x2 = terrain.width // 2 + bottom_size
    y1 = terrain.length // 2 - bottom_size
    y2 = terrain.length // 2 + bottom_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

    # goals = np.zeros((2, 2))
    # goals[0] = [x2 - 1.0 / terrain.horizontal_scale, y2 - 1.0 / terrain.horizontal_scale]
    # goals[1] = [x2 + 1.0 / terrain.horizontal_scale, y2 - 1.0 / terrain.horizontal_scale]
    # goals[2] = [x2 + 1.0 / terrain.horizontal_scale, y1 + 1.0 / terrain.horizontal_scale]
    # goals[3] = [x2 - 1.0 / terrain.horizontal_scale, y1 + 1.0 / terrain.horizontal_scale]
    # goals[4] = [x1 - 1.0 / terrain.horizontal_scale, y1 + 1.0 / terrain.horizontal_scale]
    # goals[5] = [x1 - 1.0 / terrain.horizontal_scale, y2 - 1.0 / terrain.horizontal_scale]
    # goals[6] = [x1 + 1.0 / terrain.horizontal_scale, y2 - 1.0 / terrain.horizontal_scale]
    #
    # goals[7] = [(x1 + x2) // 2, (y1 + y2) // 2]
    #
    # if random.random() > 0.5:
    #     goals[:7] = np.flip(goals[:7], axis=0)
    #
    # terrain.goals = goals * terrain.horizontal_scale

    goals = np.zeros((2, 2))
    goals[0] = [terrain.width // 2, terrain.length // 2]
    goals[1] = [x2 + 1.0 / terrain.horizontal_scale, terrain.length // 2]
    terrain.goals = goals * terrain.horizontal_scale


def parkour_terrain(terrain: SubTerrain,
                    platform_len=2.5,
                    platform_height=0.,
                    x_range=[1.8, 1.9],
                    y_range=[0., 0.1],
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    num_stones = 8

    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)

    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len - stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0

    for i in range(num_stones - 2):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2 * (left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x - last_stone_len // 2:dis_x + last_stone_len // 2,
            dis_y - stone_width // 2: dis_y + stone_width // 2] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2,
            dis_y - stone_width // 2: dis_y + stone_width // 2] = heights.astype(int) + dis_z

        goals[i + 1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2 * np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_gap_terrain(terrain: SubTerrain,
                        platform_len=2.5,
                        platform_height=0.,
                        gap_size=0.3,
                        x_range=[1.6, 2.4],
                        y_range=[-1.2, 1.2],
                        half_valid_width=[0.6, 1.2],
                        gap_depth=-200,
                        pad_width=0.1,
                        pad_height=0.5,
                        flat=False):
    num_gaps = 8

    goals = np.zeros((num_gaps, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)

    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth

    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps - 2):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2,
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x - gap_size // 2: dis_x + gap_size // 2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width:] = gap_depth

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_box_terrain(terrain: SubTerrain,
                        platform_len=0.5,
                        box_length_range=(1., 2.0),
                        box_width_range=(1.5, 3),
                        box_height_range=(0.2, 0.5),
                        dist_x_range=(3, 4),
                        guidance_width=0.2):
    num_box = 4

    # prepare magnitudes
    mid_y = terrain.length // 2  # length is actually y width

    guidance_width = round(guidance_width / terrain.horizontal_scale)
    platform_len = round(platform_len / terrain.horizontal_scale)
    box_length_range = np.round(np.array(box_length_range) / terrain.horizontal_scale)
    box_width_range = np.round(np.array(box_width_range) / terrain.horizontal_scale)
    dist_x_range = np.round(np.array(dist_x_range) / terrain.horizontal_scale)
    box_height_range = np.round(np.array(box_height_range) / terrain.vertical_scale)

    # buffer
    terrain.height_field_guidance = terrain.height_field_raw.copy()
    goals = np.zeros((num_box + 1, 2))

    cur_x = platform_len
    for i in range(num_box):
        rand_x = random.randint(*dist_x_range)
        rand_len = random.randint(*box_length_range) // 2
        rand_wid = random.randint(*box_width_range) // 2
        rand_height = random.randint(*box_height_range)
        cur_x += rand_x

        # height field raw
        x_slice = slice(cur_x - rand_len, cur_x + rand_len)
        y_slice = slice(mid_y - rand_wid, mid_y + rand_wid)
        terrain.height_field_raw[x_slice, y_slice] = rand_height

        # height field guidance
        terrain.height_field_guidance[x_slice, y_slice] = rand_height
        guidance_height = np.tile(np.linspace(0, rand_height, guidance_width), (rand_wid * 2, 1)).T
        x_slice = slice(cur_x - rand_len - guidance_width, cur_x - rand_len)
        terrain.height_field_guidance[x_slice, y_slice] = guidance_height

        # goal
        goals[i] = [cur_x, mid_y]

    goals[-1] = [terrain.width, mid_y]
    terrain.goals = goals * terrain.horizontal_scale


def parkour_step_terrain(terrain: SubTerrain,
                         platform_len=2.5,
                         rand_x_range=(0.2, 0.4),  # x distance between two steps
                         rand_y_range=(-0.15, 0.15),  # y distance between two steps center
                         step_width_range=(2, 3),
                         step_height=0.2,
                         guidance_width=0.2):
    num_steps = 6

    # prepare magnitudes
    platform_len = round(platform_len / terrain.horizontal_scale)
    mid_y = terrain.length // 2  # length is actually y width
    guidance_width = round(guidance_width / terrain.horizontal_scale)
    rand_x_range = np.round((np.array(rand_x_range) + step_height) / terrain.horizontal_scale)
    rand_y_range = np.round(np.array(rand_y_range) / terrain.horizontal_scale)
    step_height = round(step_height / terrain.vertical_scale)
    half_step_width = round(random.uniform(*step_width_range) / 2 / terrain.horizontal_scale)

    # buffer
    goals = np.zeros((num_steps + 1, 2))
    terrain.height_field_guidance = terrain.height_field_raw.copy()

    dis_x = platform_len
    stair_height = last_stair_height = 0
    for i in range(num_steps):
        rand_x = np.random.randint(*rand_x_range)
        rand_y = np.random.randint(*rand_y_range)

        if i < num_steps // 2:
            stair_height += step_height
        elif i > num_steps // 2:
            stair_height -= step_height

        # height field raw
        x_slice = slice(dis_x, dis_x + rand_x)
        y_slice = slice(mid_y + rand_y - half_step_width, mid_y + rand_y + half_step_width)
        terrain.height_field_raw[x_slice, y_slice] = stair_height

        # height field guidance
        terrain.height_field_guidance[x_slice, y_slice] = stair_height

        if i < num_steps // 2:
            guidance_height = np.tile(np.linspace(last_stair_height, stair_height, guidance_width), (2 * half_step_width, 1)).T
            terrain.height_field_guidance[dis_x - guidance_width: dis_x, y_slice] = guidance_height

        goals[i] = [dis_x + rand_x // 2, mid_y + rand_y]

        dis_x += rand_x
        last_stair_height = stair_height

    goals[-1] = [dis_x + round(1 / terrain.horizontal_scale), mid_y]
    terrain.goals = goals * terrain.horizontal_scale


def parkour_stair_terrain(
        terrain: SubTerrain,
        platform_len=2.5,
        num_steps=16,
        step_height=0.2,
        step_depth=0.2,
        num_goals=5,
        goal_deviation=0.5,
        only_up=True
):
    if only_up:
        num_steps = 40
        terrain.height_field_guidance = terrain.height_field_raw.copy()
        mid_y = terrain.length // 2  # length is actually y width

        step_height = round(step_height / terrain.vertical_scale)
        step_depth = round(step_depth / terrain.horizontal_scale)

        platform_len = round(platform_len / terrain.horizontal_scale)

        dis_x = platform_len
        stair_height = last_stair_height = 0

        for i in range(num_steps):
            stair_height += step_height
            terrain.height_field_raw[dis_x:dis_x + step_depth] = stair_height

            terrain.height_field_guidance[dis_x:dis_x + step_depth] = stair_height
            guidance_height = np.tile(np.linspace(last_stair_height, stair_height, step_depth // 2), (terrain.length, 1)).T

            if i < num_steps:
                terrain.height_field_guidance[dis_x - step_depth // 2: dis_x] = guidance_height
            elif i > num_steps:
                terrain.height_field_guidance[dis_x: dis_x + step_depth // 2] = guidance_height

            dis_x += step_depth
            last_stair_height = stair_height

        def rand_deviation():
            return round(random.uniform(-goal_deviation, goal_deviation) / terrain.horizontal_scale)

        goals = np.zeros((num_goals, 2))
        end_x = dis_x - round(1.0 / terrain.horizontal_scale)
        step_x = int((end_x - platform_len) / num_goals)

        for i in range(1, num_goals - 1):
            goals[i] = [platform_len + i * step_x, mid_y + rand_deviation()]

        goals[0] = [platform_len, mid_y]
        goals[-1] = [end_x, mid_y]

        terrain.goals = goals * terrain.horizontal_scale

    else:
        terrain.height_field_guidance = terrain.height_field_raw.copy()
        mid_y = terrain.length // 2  # length is actually y width

        step_height = round(step_height / terrain.vertical_scale)
        step_depth = round(step_depth / terrain.horizontal_scale)

        platform_len = round(platform_len / terrain.horizontal_scale)

        dis_x = platform_len
        stair_height = last_stair_height = 0

        for i in range(num_steps * 2):
            if i < num_steps:
                stair_height += step_height
            elif i > num_steps:
                stair_height -= step_height
            else:
                mid_x_stair = dis_x

            terrain.height_field_raw[dis_x:dis_x + step_depth] = stair_height

            terrain.height_field_guidance[dis_x:dis_x + step_depth] = stair_height
            if i < num_steps:
                guidance_height = np.tile(np.linspace(last_stair_height, stair_height, step_depth // 2), (terrain.length, 1)).T
                terrain.height_field_guidance[dis_x - step_depth // 2: dis_x] = guidance_height

            dis_x += step_depth
            last_stair_height = stair_height

        goals = np.zeros((2, 2))
        goals[0] = [mid_x_stair, mid_y]
        goals[1] = [dis_x + round(1 / terrain.horizontal_scale), mid_y]
        terrain.goals = goals * terrain.horizontal_scale


def parkour_mini_stair_terrain(
        terrain: SubTerrain,
        platform_len=2.5,
        num_steps=16,
        step_height=0.2,
        step_depth=0.2,
        goal_deviation=0.2,
):
    terrain.height_field_guidance = terrain.height_field_raw.copy()
    mid_y = terrain.length // 2  # length is actually y width

    step_height = round(step_height / terrain.vertical_scale)
    step_depth = round(step_depth / terrain.horizontal_scale)
    platform_len = round(platform_len / terrain.horizontal_scale)

    mini_stair_hf = np.zeros((150, 100))
    mini_stair_hf[:step_depth] = step_height
    mini_stair_hf[step_depth: 2 * step_depth] = 2 * step_height
    mini_stair_hf[2 * step_depth: 3 * step_depth] = 3 * step_height
    mini_stair_hf[3 * step_depth: -3 * step_depth] = 4 * step_height
    mini_stair_hf[-3 * step_depth: -2 * step_depth] = 3 * step_height
    mini_stair_hf[-2 * step_depth: -step_depth] = 2 * step_height
    mini_stair_hf[-step_depth:] = step_height

    # mini guidance terrain
    mini_stair_hf_guidance = np.zeros((200, 100))
    mini_stair_hf_guidance[50:] = mini_stair_hf
    mini_stair_hf_guidance[50 - step_depth // 2: 50] = np.tile(np.linspace(0 * step_height, 1 * step_height, step_depth // 2), (100, 1)).T
    mini_stair_hf_guidance[50 + step_depth - step_depth // 2: 50 + step_depth] = np.tile(np.linspace(1 * step_height, 2 * step_height, step_depth // 2), (100, 1)).T
    mini_stair_hf_guidance[50 + 2 * step_depth - step_depth // 2: 50 + 2 * step_depth] = np.tile(np.linspace(2 * step_height, 3 * step_height, step_depth // 2), (100, 1)).T
    mini_stair_hf_guidance[50 + 3 * step_depth - step_depth // 2: 50 + 3 * step_depth] = np.tile(np.linspace(3 * step_height, 4 * step_height, step_depth // 2), (100, 1)).T

    def rand_deviation():
        return round(random.uniform(-goal_deviation, goal_deviation) / terrain.horizontal_scale)

    goals = np.zeros((5, 2))

    start_x = platform_len
    terrain.height_field_raw[start_x: start_x + 150, 50:-50] = mini_stair_hf
    goals[0] = [start_x + 75, mid_y + rand_deviation()]

    start_x = start_x + 150 + 50
    terrain.height_field_raw[start_x: start_x + 150, 50:-50] = mini_stair_hf
    goals[1] = [start_x + 75, mid_y + rand_deviation()]

    start_x = start_x + 150 + 50
    terrain.height_field_raw[start_x: start_x + 150, 50:-50] = mini_stair_hf
    goals[2] = [start_x + 75, mid_y + rand_deviation()]

    start_x = start_x + 150 + 50
    terrain.height_field_raw[start_x: start_x + 150, 50:-50] = mini_stair_hf
    goals[3] = [start_x + 75, mid_y + rand_deviation()]

    start_x = start_x + 150
    goals[4] = [start_x + 75, mid_y + rand_deviation()]

    terrain.goals = goals * terrain.horizontal_scale

    # guidance terrain
    start_x = platform_len - 50
    terrain.height_field_guidance[start_x: start_x + 200, 50:-50] = mini_stair_hf_guidance

    start_x = start_x + 200
    terrain.height_field_guidance[start_x: start_x + 200, 50:-50] = mini_stair_hf_guidance

    start_x = start_x + 200
    terrain.height_field_guidance[start_x: start_x + 200, 50:-50] = mini_stair_hf_guidance

    start_x = start_x + 200
    terrain.height_field_guidance[start_x: start_x + 200, 50:-50] = mini_stair_hf_guidance


def parkour_flat_terrain(terrain: SubTerrain,
                         platform_len=2.,
                         x_range=(1.5, 2.4),
                         y_range=(-1., 1.)):
    mid_y = terrain.length // 2  # length is actually y width
    platform_len = round(platform_len / terrain.horizontal_scale)

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    cur_x = platform_len
    goals = np.zeros((8, 2))
    goals[0] = [platform_len, mid_y]

    for i in range(6):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        cur_x += rand_x
        goals[i + 1] = [cur_x, mid_y + rand_y]

    goals[-1] = [terrain.width - platform_len // 2, mid_y]
    terrain.goals = goals * terrain.horizontal_scale
