import scipy

from .terrain_types import *
from .terrain_utils import *
from .utils import convert_heightfield_to_trimesh, edge_detection


class Terrain:
    terrain_generators = {
        'smooth_slope': SmoothSlope,
        'rough_slope': RoughSlope,
        'stairs_up': PyramidStairsUp,
        'stairs_down': PyramidStairsDown,
        'huge_stair': HugeStair,
        'discrete': None,
        'stepping_stone': None,
        'gap': None,
        'pit': None,

        'parkour_flat': ParkourFlat,
        'parkour': None,
        'parkour_gap': None,
        'parkour_box': None,
        'parkour_step': None,
        'parkour_stair': ParkourStair,
        'parkour_stair_down': ParkourStairDown,
        'parkour_mini_stair': ParkourMiniStair,
        'parkour_go_back_stair': ParkourGoBackStair,
    }

    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.description_type in ["none", 'plane']:
            return

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)

        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))

        self.height_field_raw: np.array
        self.height_field_guidance: np.array
        self.edge_mask: np.array

        self.goals: np.array
        self.num_goals: np.array

        self._compose_terrain(max_difficulty=cfg.max_difficulty)

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

        self._compute_edge_and_guidance_hmap()

    def _compose_terrain(self, max_difficulty=False):
        terrain_dict = {t: p for t, p in self.cfg.terrain_dict.items() if p > 0.}
        proportions = np.array(list(terrain_dict.values()))
        prop_sum = np.cumsum(proportions / np.sum(proportions))

        terrain_mat = np.empty((self.cfg.num_rows, self.cfg.num_cols), dtype=object)

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                choice = j / self.cfg.num_cols + 0.001

                for selected_terrain_name, prop_sum_i in zip(terrain_dict, prop_sum):
                    if choice < prop_sum_i:
                        break

                terrain = self.terrain_generators[selected_terrain_name]()

                if max_difficulty:
                    terrain.make(difficulty=0.9999)
                else:
                    terrain.make(difficulty=i / (self.cfg.num_rows - 1) if self.cfg.num_rows > 1 else 0.5)

                terrain_mat[i, j] = terrain

        # examine number of goals
        self.num_goals = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=int)
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                terrain = terrain_mat[row, col]

                if isinstance(terrain, GoalGuidedTerrain):
                    self.num_goals[row, col] = len(terrain.goals)

        max_goal_num = max(1, self.num_goals.max())
        self.goals = np.zeros((self.cfg.num_rows, self.cfg.num_cols, max_goal_num, 3))

        self.add_terrains_to_map(terrain_mat)

        for r in range(self.goals.shape[0]):
            for c in range(self.goals.shape[1]):
                for goal_i in range(self.goals.shape[2]):
                    goal_xy = self.goals[r, c, goal_i, :2] + self.cfg.border_size
                    pts = (goal_xy / self.cfg.horizontal_scale).astype(int)
                    self.goals[r, c, goal_i, 2] = self.height_field_raw[pts[0], pts[1]] * self.cfg.vertical_scale

    # def make_terrain(self, selected_terrain_type, difficulty):
    #     discrete_obstacles_height = 0.03 + difficulty * 0.15
    #     stepping_stones_size = 1.5 * (1.05 - difficulty)
    #     stone_distance = 0.05 if difficulty == 0 else 0.1
    #     gap_size = 0.1 + 0.4 * difficulty
    #     pit_depth = 0.1 + 0.3 * difficulty
    #
    #     if selected_terrain_type < self.proportions[4]:
    #         terrain.terrain_type = Terrain.terrain_type.discrete
    #         num_rectangles = 20
    #         rectangle_min_size = 1.
    #         rectangle_max_size = 2.
    #         raise NotImplementedError
    #         self.terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
    #                                                       rectangle_max_size, num_rectangles, platform_size=3.)
    #
    #     elif selected_terrain_type < self.proportions[5]:
    #         terrain.terrain_type = Terrain.terrain_type.stepping_stone
    #         raise NotImplementedError
    #         self.terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
    #                                                    stone_distance=stone_distance, max_height=0., platform_size=4.)
    #
    #     elif selected_terrain_type < self.proportions[6]:
    #         terrain.terrain_type = Terrain.terrain_type.gap
    #         gap_terrain(terrain, gap_size=gap_size, outer_platform_size=6.)
    #         terrain.centered_origin = False
    #         self.add_roughness(terrain, difficulty)
    #
    #     elif selected_terrain_type < self.proportions[7]:
    #         terrain.terrain_type = Terrain.terrain_type.pit
    #         pit_terrain(terrain, depth=pit_depth, bottom_size=4.)
    #         terrain.centered_origin = False
    #         self.add_roughness(terrain, difficulty)
    #
    #     elif selected_terrain_type < self.proportions[8]:
    #         terrain.terrain_type = Terrain.terrain_type.parkour
    #         x_range = [-0.1, 0.1 + 0.3 * difficulty]  # offset to stone_len
    #         y_range = [0.2, 0.3 + 0.1 * difficulty]
    #         stone_len = [0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty]  # 2 * round((0.6) / 2.0, 1)
    #         incline_height = 0.25 * difficulty
    #         last_incline_height = incline_height + 0.1 - 0.1 * difficulty
    #         parkour_terrain(terrain,
    #                         x_range=x_range,
    #                         y_range=y_range,
    #                         incline_height=incline_height,
    #                         stone_len=stone_len,
    #                         stone_width=1.0,
    #                         last_incline_height=last_incline_height,
    #                         pad_height=0,
    #                         pit_depth=[0.2, 1])
    #         terrain.centered_origin = False
    #         self.add_roughness(terrain, difficulty)
    #
    #     elif selected_terrain_type < self.proportions[9]:
    #         terrain.terrain_type = Terrain.terrain_type.parkour_gap
    #         parkour_gap_terrain(terrain,
    #                             gap_size=gap_size,
    #                             gap_depth=[0.2, 1],
    #                             pad_height=0,
    #                             x_range=[0.8, 1.5],
    #                             y_range=self.cfg.y_range,
    #                             half_valid_width=[0.6, 1.2])
    #         terrain.centered_origin = False
    #         self.add_roughness(terrain, difficulty)
    #
    #     elif selected_terrain_type < self.proportions[10]:
    #         terrain.terrain_type = Terrain.terrain_type.parkour_box
    #         parkour_box_terrain(terrain, box_height_range=[pit_depth - 0.05, pit_depth + 0.05])
    #         terrain.centered_origin = False
    #         self.add_roughness(terrain, 0 * difficulty)
    #
    #     elif selected_terrain_type < self.proportions[11]:
    #         terrain.terrain_type = Terrain.terrain_type.parkour_step
    #         parkour_step_terrain(terrain,
    #                              step_height=pit_depth,
    #                              rand_x_range=(0.8, 1.2),
    #                              rand_y_range=self.cfg.y_range,
    #                              step_width_range=(1., 1.6))
    #         terrain.centered_origin = False
    #         self.add_roughness(terrain, difficulty)
    #
    #     return terrain

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
        offset_x1, offset_y1, offset_x2, offset_y2 = 0, 0, 0, 0

        # compute origins and goals
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                # get terrain chunk
                terrain = terrain_mat[row, col]

                # store terrain type
                self.terrain_type[row, col] = terrain.terrain_type.value

                # fill in the canvas
                if np.all(terrain.height_field_raw.shape == unique_shape[0]):
                    start_x, start_y = offset_x1 * h1, offset_y1 * w1
                    self.height_field_raw[start_x: start_x + h1, start_y: start_y + w1] = terrain.height_field_raw
                    offset_x1 += 1

                    # change column
                    if offset_x1 >= best_combination[0][0]:
                        offset_x1 = 0
                        offset_y1 += 1

                else:
                    start_x, start_y = offset_x2 * h2, best_combination[0][1] * w1 + offset_y2 * w2
                    self.height_field_raw[start_x: start_x + h2, start_y: start_y + w2] = terrain.height_field_raw
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

                if isinstance(terrain, GoalGuidedTerrain):
                    goal_pos = terrain.goals + [start_x * terrain.horizontal_scale,
                                                start_y * terrain.horizontal_scale]
                    self.goals[row, col, :len(terrain.goals), :2] = goal_pos

        self.height_field_raw = np.pad(self.height_field_raw, (self.border,), 'constant', constant_values=(0,))

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

                if isinstance(terrain, GoalGuidedTerrain):
                    goal_pos = terrain.goals + [(start_x - self.border) * terrain.horizontal_scale,
                                                (start_y - self.border) * terrain.horizontal_scale]
                    self.goals[row, col, :len(terrain.goals), :2] = goal_pos

                start_x = end_x
            start_y = end_y

        self.height_field_raw = concat_chunks(terrain_mat, 'height_field_raw')
        self.height_field_raw = np.pad(self.height_field_raw, (self.border,), 'constant', constant_values=(0,))

    def _compute_edge_and_guidance_hmap(self):
        self.edge_mask = edge_detection(self.height_field_raw,
                                        self.cfg.horizontal_scale,
                                        self.cfg.vertical_scale,
                                        self.cfg.slope_treshold)
        half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
        structure = np.ones((half_edge_width * 2 + 1, half_edge_width * 2 + 1))
        self.edge_mask = scipy.ndimage.binary_dilation(self.edge_mask, structure=structure)

        eroded = scipy.ndimage.binary_erosion(self.edge_mask, structure=structure)
        boundary = self.edge_mask & ~eroded

        # 2. collect boundary points + values
        ys, xs = np.nonzero(boundary)
        vals = self.height_field_raw[ys, xs]

        # 3. collect interior points
        yi, xi = np.nonzero(self.edge_mask & ~boundary)

        # 4. interpolate
        points = np.column_stack((ys, xs))
        interp_xy = np.column_stack((yi, xi))
        interp_vals = scipy.interpolate.griddata(points, vals, interp_xy, method='linear')  # or 'cubic'

        # 5. fill back in
        self.height_field_guidance = self.height_field_raw.copy()
        self.height_field_guidance[yi, xi] = interp_vals
