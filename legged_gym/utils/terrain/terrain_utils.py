import numpy as np
from scipy.interpolate import RegularGridInterpolator


class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.width = width
        self.length = length
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
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

    # goals = np.zeros((1, 2))
    # goals[0, 0] = center_x + x2 + 50
    # goals[0, 1] = center_y
    # terrain.goals = goals * terrain.horizontal_scale


def pit_terrain(terrain: SubTerrain, depth, num_goal=8, bottom_size=1.):
    depth = int(depth / terrain.vertical_scale)
    bottom_size = int(bottom_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - bottom_size
    x2 = terrain.width // 2 + bottom_size
    y1 = terrain.length // 2 - bottom_size
    y2 = terrain.length // 2 + bottom_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

    # goals = np.zeros((num_goal, 2))
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


def parkour_terrain(terrain: SubTerrain,
                    platform_len=2.5,
                    platform_height=0.,
                    num_stones=8,
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
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones + 2, 2))
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

    for i in range(num_stones):
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
                        num_gaps=8,
                        gap_size=0.3,
                        x_range=[1.6, 2.4],
                        y_range=[-1.2, 1.2],
                        half_valid_width=[0.6, 1.2],
                        gap_depth=-200,
                        pad_width=0.1,
                        pad_height=0.5,
                        flat=False):
    goals = np.zeros((num_gaps + 2, 2))
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
    for i in range(num_gaps):
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
                        num_goals=8,
                        platform_len=1.5,
                        box_length=(1., 2.0),
                        box_width=(1.5, 3),
                        box_height=(0.2, 0.5),
                        x_range=(3, 4)):
    mid_y = terrain.length // 2  # length is actually y width
    platform_len = round(platform_len / terrain.horizontal_scale)

    box_length_min = round(box_length[0] / terrain.horizontal_scale)
    box_length_max = round(box_length[1] / terrain.horizontal_scale)
    box_width_min = round(box_width[0] / terrain.horizontal_scale)
    box_width_max = round(box_width[1] / terrain.horizontal_scale)
    box_height_min = round(box_height[0] / terrain.vertical_scale)
    box_height_max = round(box_height[1] / terrain.vertical_scale)

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)

    cur_x = platform_len

    goals = np.zeros((num_goals, 2))
    goals[0] = [platform_len - 1, mid_y]

    for i in range(num_goals - 1):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_len = np.random.randint(box_length_min, box_length_max)
        rand_wid = np.random.randint(box_width_min, box_width_max)
        rand_height = np.random.randint(box_height_min, box_height_max)
        cur_x += rand_x

        if cur_x + rand_len // 2 > terrain.width - platform_len:
            goals[i + 1] = goals[i]

        else:
            goals[i + 1] = [cur_x, mid_y]

            terrain.height_field_raw[cur_x - rand_len // 2: cur_x + rand_len // 2,
            mid_y - rand_wid // 2: mid_y + rand_wid // 2] = rand_height

    goals[-1] = [terrain.width, mid_y]
    terrain.goals = goals * terrain.horizontal_scale


def parkour_step_terrain(terrain: SubTerrain,
                         platform_len=2.5,
                         platform_height=0.,
                         num_stones=8,
                         x_range=[0.2, 0.4],
                         y_range=[-0.15, 0.15],
                         half_valid_width=[0.15, 0.2],
                         step_height=0.2,
                         pad_width=0.1,
                         pad_height=0.5):
    goals = np.zeros((num_stones + 2, 2))
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round((x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x:dis_x + rand_x, ] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width:] = 0

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_stair_terrain(terrain: SubTerrain,
                          platform_len=2.5,
                          num_steps=16,
                          num_goals=8,
                          step_height=0.2,
                          step_width=0.2):
    terrain.height_field_guidance = terrain.height_field_raw.copy()
    mid_y = terrain.length // 2  # length is actually y width

    step_height = round(step_height / terrain.vertical_scale)
    step_width = round(step_width / terrain.horizontal_scale)

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

        terrain.height_field_raw[dis_x:dis_x + step_width] = stair_height

        terrain.height_field_guidance[dis_x:dis_x + step_width] = stair_height
        guidance_height = np.tile(np.linspace(last_stair_height, stair_height, step_width // 2), (terrain.length, 1)).T

        if i < num_steps:
            terrain.height_field_guidance[dis_x - step_width // 2: dis_x] = guidance_height
        elif i > num_steps:
            terrain.height_field_guidance[dis_x: dis_x + step_width // 2] = guidance_height

        dis_x += step_width
        last_stair_height = stair_height

    goals = np.zeros((num_goals, 2))
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    goals[1] = [mid_x_stair, mid_y]
    # goals[1] = [mid_x_stair, mid_y * 2 * random.random()]  # TODO: unknow bug??????????
    goals[2] = [dis_x + round(1 / terrain.horizontal_scale), mid_y]
    for i in range(3, num_goals):
        goals[i] = goals[i - 1]

    terrain.goals = goals * terrain.horizontal_scale


def parkour_flat_terrain(terrain: SubTerrain,
                         platform_len=2.,
                         num_goals=12,
                         x_range=(1.5, 2.4),
                         y_range=(-1., 1.)):
    mid_y = terrain.length // 2  # length is actually y width
    platform_len = round(platform_len / terrain.horizontal_scale)

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    cur_x = platform_len
    goals = np.zeros((num_goals, 2))
    goals[0] = [platform_len, mid_y]

    for i in range(num_goals - 2):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        cur_x += rand_x
        goals[i + 1] = [cur_x, mid_y + rand_y]

    goals[-1] = [terrain.width - platform_len // 2, mid_y]
    terrain.goals = goals * terrain.horizontal_scale
