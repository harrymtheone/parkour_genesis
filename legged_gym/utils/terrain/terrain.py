from enum import Enum

import scipy

from .terrain_utils import *
from .utils import convert_heightfield_to_trimesh, edge_detection, generate_fractal_noise_2d


class Terrain:
    class terrain_type(Enum):
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
        parkour_flat = 13

    def __init__(self, cfg, terrain_utils):
        self.cfg = cfg
        self.terrain_utils = terrain_utils

        if cfg.description_type in ["none", 'plane']:
            return

        self.proportions = np.array(list(cfg.terrain_dict.values()))
        self.proportions = np.cumsum(self.proportions / np.sum(self.proportions))
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        self.goals = None
        self.num_goals = None

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.height_field_raw: np.array
        self.height_field_guidance: np.array

        self.curriculum(max_difficulty=cfg.max_difficulty)

        if self.cfg.horizontal_scale < self.cfg.horizontal_scale_downsample:
            downsample_factor = self.cfg.horizontal_scale / self.cfg.horizontal_scale_downsample
            self.height_field_raw_downsample = scipy.ndimage.zoom(self.height_field_raw, (downsample_factor, downsample_factor), order=0)
            print(f'Downsample height_field_raw from {self.height_field_raw.shape} to {self.height_field_raw_downsample.shape}')

            self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw_downsample,
                                                                           self.cfg.horizontal_scale_downsample,
                                                                           self.cfg.vertical_scale,
                                                                           self.cfg.slope_treshold)
        else:
            self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw,
                                                                           self.cfg.horizontal_scale,
                                                                           self.cfg.vertical_scale,
                                                                           self.cfg.slope_treshold)

        self.edge_mask = edge_detection(self.height_field_raw,
                                        self.cfg.horizontal_scale,
                                        self.cfg.vertical_scale,
                                        self.cfg.slope_treshold)
        half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
        structure = np.ones((half_edge_width * 2 + 1, half_edge_width * 2 + 1))
        self.edge_mask = scipy.ndimage.binary_dilation(self.edge_mask, structure=structure)

        print(f'Created {self.vertices.shape[0]} vertices')
        print(f'Created {self.triangles.shape[0]} triangles')

        if cfg.description_type == "trimesh" and self.cfg.simplify_grid:
            import pyfqmr
            mesh_simplifier = pyfqmr.Simplify()
            mesh_simplifier.setMesh(self.vertices, self.triangles)
            mesh_simplifier.simplify_mesh(target_count=int(0.05 * self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

            self.vertices, self.triangles, _ = mesh_simplifier.getMesh()
            self.vertices = self.vertices.astype(np.float32)
            self.triangles = self.triangles.astype(np.uint32)

            print(f'Simplified to {self.vertices.shape[0]} vertices')
            print(f'Simplified to {self.triangles.shape[0]} triangles')

    def curriculum(self, max_difficulty=False):
        terrain_mat = np.empty((self.cfg.num_rows, self.cfg.num_cols), dtype=object)

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                choice = j / self.cfg.num_cols + 0.001
                difficulty = i / (self.cfg.num_rows - 1) if self.cfg.num_rows > 1 else 0.5

                if max_difficulty:
                    # terrain = self.make_terrain(choice, 0.8 + 0.2 * difficulty)
                    terrain = self.make_terrain(choice, 0.9999)
                else:
                    terrain = self.make_terrain(choice, difficulty)

                terrain_mat[i, j] = terrain

        # examine number of goals
        self.num_goals = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=int)
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                terrain = terrain_mat[row, col]

                if terrain.goals is None:
                    self.num_goals[row, col] = 0
                else:
                    self.num_goals[row, col] = len(terrain_mat[row, col].goals)

        max_goal_num = max(1, self.num_goals.max())
        self.goals = np.zeros((self.cfg.num_rows, self.cfg.num_cols, max_goal_num, 3))

        self.add_terrains_to_map(terrain_mat)

        for r in range(self.goals.shape[0]):
            for c in range(self.goals.shape[1]):
                for goal_i in range(self.goals.shape[2]):
                    goal_xy = self.goals[r, c, goal_i, :2] + self.cfg.border_size
                    pts = (goal_xy / self.cfg.horizontal_scale).astype(int)
                    self.goals[r, c, goal_i, 2] = self.height_field_raw[pts[0], pts[1]] * self.cfg.vertical_scale

    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.roughness_height[1] - self.cfg.roughness_height[0]) * difficulty + self.cfg.roughness_height[0]
        random_uniform_terrain(terrain,
                               min_height=-max_height,
                               max_height=max_height,
                               step=0.005,
                               downsampled_scale=self.cfg.downsampled_scale)

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
        ) / self.cfg.vertical_scale

        terrain.height_field_raw[:] += heightfield_noise

    def make_terrain(self, choice, difficulty):
        if choice < self.proportions[7]:
            # legged_gym terrain
            pixel_length_per_env = int(self.cfg.terrain_size[0] / self.cfg.horizontal_scale)
            pixel_width_per_env = int(self.cfg.terrain_size[1] / self.cfg.horizontal_scale)
        else:
            # parkour terrain
            pixel_length_per_env = int(self.cfg.terrain_parkour_size[0] / self.cfg.horizontal_scale)
            pixel_width_per_env = int(self.cfg.terrain_parkour_size[1] / self.cfg.horizontal_scale)

        terrain = SubTerrain("terrain",
                             width=pixel_length_per_env,
                             length=pixel_width_per_env,
                             vertical_scale=self.cfg.vertical_scale,
                             horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        stair_up_height = 0.02 + 0.08 * difficulty
        stair_down_height = 0.02 + 0.08 * difficulty
        stair_height_goal = 0.07 + 0.09 * difficulty  # 跑酷楼梯的高度
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 0.1 + 0.4 * difficulty
        pit_depth = 0.1 + 0.3 * difficulty

        if choice < self.proportions[0]:
            terrain.terrain_type = Terrain.terrain_type.smooth_slope
            # if choice < self.proportions[0] / 2:
            #     slope *= -1
            # terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # add_fractal_roughness(terrain, scale=5)
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[1]:
            terrain.terrain_type = Terrain.terrain_type.rough_slope
            # self.terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
            self.add_fractal_roughness(terrain, difficulty)
            # self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                terrain.terrain_type = Terrain.terrain_type.stairs_up
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=-stair_up_height, platform_size=3.)
            else:
                terrain.terrain_type = Terrain.terrain_type.stairs_down
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=stair_down_height, platform_size=3.)

            # self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[4]:
            terrain.terrain_type = Terrain.terrain_type.discrete
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            self.terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                          rectangle_max_size, num_rectangles, platform_size=3.)

        elif choice < self.proportions[5]:
            terrain.terrain_type = Terrain.terrain_type.stepping_stone
            self.terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                       stone_distance=stone_distance, max_height=0., platform_size=4.)

        elif choice < self.proportions[6]:
            terrain.terrain_type = Terrain.terrain_type.gap
            gap_terrain(terrain, gap_size=gap_size, outer_platform_size=6.)
            terrain.centered_origin = False
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[7]:
            terrain.terrain_type = Terrain.terrain_type.pit
            pit_terrain(terrain, depth=pit_depth, bottom_size=4.)
            terrain.centered_origin = False
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[8]:
            terrain.terrain_type = Terrain.terrain_type.parkour
            x_range = [-0.1, 0.1 + 0.3 * difficulty]  # offset to stone_len
            y_range = [0.2, 0.3 + 0.1 * difficulty]
            stone_len = [0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty]  # 2 * round((0.6) / 2.0, 1)
            incline_height = 0.25 * difficulty
            last_incline_height = incline_height + 0.1 - 0.1 * difficulty
            parkour_terrain(terrain,
                            x_range=x_range,
                            y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len,
                            stone_width=1.0,
                            last_incline_height=last_incline_height,
                            pad_height=0,
                            pit_depth=[0.2, 1])
            terrain.centered_origin = False
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[9]:
            terrain.terrain_type = Terrain.terrain_type.parkour_gap
            parkour_gap_terrain(terrain,
                                gap_size=gap_size,
                                gap_depth=[0.2, 1],
                                pad_height=0,
                                x_range=[0.8, 1.5],
                                y_range=self.cfg.y_range,
                                half_valid_width=[0.6, 1.2])
            terrain.centered_origin = False
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[10]:
            terrain.terrain_type = Terrain.terrain_type.parkour_box
            parkour_box_terrain(terrain, box_height_range=[pit_depth - 0.05, pit_depth + 0.05])
            terrain.centered_origin = False
            self.add_roughness(terrain, 0 * difficulty)

        elif choice < self.proportions[11]:
            terrain.terrain_type = Terrain.terrain_type.parkour_step
            parkour_step_terrain(terrain,
                                 step_height=pit_depth,
                                 rand_x_range=(0.8, 1.2),
                                 rand_y_range=self.cfg.y_range,
                                 step_width_range=(1., 1.6))
            terrain.centered_origin = False
            self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[12]:
            terrain.terrain_type = Terrain.terrain_type.parkour_stair
            parkour_stair_terrain(terrain,
                                  step_height=stair_height_goal,
                                  # step_depth=random.uniform(0.25, 0.35))  # 0.31
                                  step_depth=0.31)  # 0.31
            terrain.centered_origin = False
            # self.add_roughness(terrain, difficulty)

        elif choice < self.proportions[13]:
            terrain.terrain_type = Terrain.terrain_type.parkour_flat
            parkour_flat_terrain(terrain)
            terrain.centered_origin = False
            self.add_roughness(terrain, difficulty)

        return terrain

    def add_terrains_to_map(self, terrain_mat):
        # check if terrains are of the same shape
        unique_shape = [t.height_field_raw.shape for t in terrain_mat.T.flatten()]
        unique_shape, counts = np.unique(unique_shape, axis=0, return_counts=True)

        if len(unique_shape) != 2:
            return self.add_terrains_to_map_compact(terrain_mat)

        if unique_shape[0, 0] >= unique_shape[1, 0]:
            raise ValueError('terrain 1 height should be less than terrain 2 height')

        # find the best combination
        (h1, w1), (h2, w2) = unique_shape[0], unique_shape[1]
        min_canvas_area = None

        for row_2 in range(1, counts[1] + 1):
            col_2 = np.ceil(counts[1] / row_2).astype(int)

            row_1 = (row_2 * h2) // h1
            col_1 = np.ceil(counts[0] / row_1).astype(int)

            canvas_height = h2 * row_2
            canvas_width = w2 * col_2 + w1 * col_1
            canvas_area = canvas_height * canvas_width

            if min_canvas_area is None or canvas_area < min_canvas_area:
                min_canvas_area = canvas_area
                best_combination = ((row_1, col_1), (row_2, col_2))
                best_canvas_height = canvas_height
                best_canvas_width = canvas_width

        print('best_combination: ', best_combination)
        terrain_area = np.prod(unique_shape, axis=1)
        terrain_area = np.sum(terrain_area * [counts])
        print('area utilization: ', terrain_area / min_canvas_area)

        self.height_field_raw = np.zeros((best_canvas_height, best_canvas_width))
        self.height_field_guidance = np.zeros((best_canvas_height, best_canvas_width))
        offset_x1, offset_y1, offset_x2, offset_y2 = 0, 0, 0, 0

        # compute origins and goals
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                # get terrain chunk
                terrain = terrain_mat[row, col]

                # store terrain type
                self.terrain_type[row, col] = terrain.terrain_type.value

                # update guidance terrain
                if terrain.height_field_guidance is None:
                    terrain.height_field_guidance = terrain.height_field_raw.copy()

                # fill in the canvas
                if np.all(terrain.height_field_raw.shape == unique_shape[0]):
                    start_x, start_y = offset_x1 * h1, offset_y1 * w1
                    self.height_field_raw[start_x: start_x + h1, start_y: start_y + w1] = terrain.height_field_raw
                    self.height_field_guidance[start_x: start_x + h1, start_y: start_y + w1] = terrain.height_field_guidance
                    offset_x1 += 1

                    # change column
                    if offset_x1 >= best_combination[0][0]:
                        offset_x1 = 0
                        offset_y1 += 1

                else:
                    start_x, start_y = offset_x2 * h2, best_combination[0][1] * w1 + offset_y2 * w2
                    self.height_field_raw[start_x: start_x + h2, start_y: start_y + w2] = terrain.height_field_raw
                    self.height_field_guidance[start_x: start_x + h2, start_y: start_y + w2] = terrain.height_field_guidance
                    offset_x2 += 1

                    # change column
                    if offset_x2 >= best_combination[1][0]:
                        offset_x2 = 0
                        offset_y2 += 1

                # compute terrain origin
                if terrain.centered_origin:
                    env_origin_x = (start_x + 0.5 * terrain.width) * terrain.horizontal_scale
                else:
                    env_origin_x = start_x * terrain.horizontal_scale + 1.0

                env_origin_y = (start_y + 0.5 * terrain.length) * terrain.horizontal_scale

                if self.cfg.origin_zero_z or not terrain.centered_origin:
                    env_origin_z = 0
                else:
                    x1 = int(0.5 * terrain.width - 0.5 / terrain.horizontal_scale)  # within 1 meter square range
                    x2 = int(0.5 * terrain.width + 0.5 / terrain.horizontal_scale)
                    y1 = int(0.5 * terrain.length - 0.5 / terrain.horizontal_scale)
                    y2 = int(0.5 * terrain.length + 0.5 / terrain.horizontal_scale)
                    env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale

                self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]

                if terrain.goals is not None:
                    goal_pos = terrain.goals + [start_x * terrain.horizontal_scale,
                                                start_y * terrain.horizontal_scale]
                    self.goals[row, col, :len(terrain.goals), :2] = goal_pos

        self.height_field_raw = np.pad(self.height_field_raw, (self.border,), 'constant', constant_values=(0,))
        self.height_field_guidance = np.pad(self.height_field_guidance, (self.border,), 'constant', constant_values=(0,))

    def add_terrains_to_map_compact(self, terrain_mat):
        def concat_chunks(chunk_mat, hf_name):
            chunk_col_list = []  # the list to store columns of terrain
            col_len_max = 0  # the maximum length of the columns. Used for padding

            for col in chunk_mat.T:
                hf_raw_col = [getattr(chunk, hf_name) for chunk in col]  # the list to store terrain chunks of the same column
                hf_raw_col = np.concatenate(hf_raw_col, axis=0)
                chunk_col_list.append(hf_raw_col)
                col_len_max = max(col_len_max, hf_raw_col.shape[0])

            # pad terrain_cols to same length
            for i, col in enumerate(chunk_col_list):
                pad_len = col_len_max - col.shape[0]
                chunk_col_list[i] = np.pad(col, ((0, pad_len), (0, 0)), 'constant', constant_values=0)

            return np.concatenate(chunk_col_list, axis=1)

        start_y = self.border
        for col in range(self.cfg.num_cols):
            start_x = self.border

            for row in range(self.cfg.num_rows):
                terrain = terrain_mat[row, col]
                self.terrain_type[row, col] = terrain.terrain_type.value

                if terrain.height_field_guidance is None:
                    terrain.height_field_guidance = terrain.height_field_raw.copy()

                end_x = start_x + terrain.width
                end_y = start_y + terrain.length

                # compute terrain origin
                if terrain.centered_origin:
                    env_origin_x = (start_x - self.border + 0.5 * terrain.width) * terrain.horizontal_scale
                else:
                    env_origin_x = (start_x - self.border) * terrain.horizontal_scale + 1.0

                env_origin_y = (start_y - self.border + 0.5 * terrain.length) * terrain.horizontal_scale

                if self.cfg.origin_zero_z or not terrain.centered_origin:
                    env_origin_z = 0
                else:
                    x1 = int(0.5 * terrain.width - 0.5 / terrain.horizontal_scale)  # within 1 meter square range
                    x2 = int(0.5 * terrain.width + 0.5 / terrain.horizontal_scale)
                    y1 = int(0.5 * terrain.length - 0.5 / terrain.horizontal_scale)
                    y2 = int(0.5 * terrain.length + 0.5 / terrain.horizontal_scale)
                    env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale

                self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]

                if terrain.goals is not None:
                    goal_pos = terrain.goals + [(start_x - self.border) * terrain.horizontal_scale,
                                                (start_y - self.border) * terrain.horizontal_scale]
                    self.goals[row, col, :len(terrain.goals), :2] = goal_pos

                start_x = end_x
            start_y = end_y

        self.height_field_raw = concat_chunks(terrain_mat, 'height_field_raw')
        self.height_field_raw = np.pad(self.height_field_raw, (self.border,), 'constant', constant_values=(0,))
        self.height_field_guidance = concat_chunks(terrain_mat, 'height_field_guidance')
        self.height_field_guidance = np.pad(self.height_field_guidance, (self.border,), 'constant', constant_values=(0,))
