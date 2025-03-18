import noise
import numpy as np
from pydelatin import Delatin


# def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
#     """
#     Convert a heightfield array to a triangle mesh represented by vertices and triangles.
#     Optionally, corrects vertical surfaces above the provide slope threshold:
#
#         If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
#                    B(x2,y2)
#                   /|
#                  / |
#                 /  |
#         (x1,y1)A---A'(x2,y1)
#
#     Parameters:
#         height_field_raw (np.array): input heightfield
#         horizontal_scale (float): horizontal scale of the heightfield [meters]
#         vertical_scale (float): vertical scale of the heightfield [meters]
#         slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
#     Returns:
#         vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
#         triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
#     """
#     if slope_threshold is None:
#         raise ValueError('slope threshold cannot be None!!!')
#
#     hf = height_field_raw
#     num_rows = hf.shape[0]
#     num_cols = hf.shape[1]
#
#     y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
#     x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
#     yy, xx = np.meshgrid(y, x)
#
#     slope_threshold *= horizontal_scale / vertical_scale
#     move_x = np.zeros((num_rows, num_cols))
#     move_y = np.zeros((num_rows, num_cols))
#     move_corners = np.zeros((num_rows, num_cols))
#     move_x[:num_rows - 1, :] += hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold
#     move_x[1:num_rows, :] -= hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
#     move_y[:, :num_cols - 1] += hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold
#     move_y[:, 1:num_cols] -= hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
#     move_corners[:num_rows - 1, :num_cols - 1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows - 1, :num_cols - 1] > slope_threshold)
#     move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows - 1, :num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
#     xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
#     yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale
#
#     # create triangle mesh vertices and triangles from the heightfield grid
#     vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
#     vertices[:, 0] = xx.flatten()
#     vertices[:, 1] = yy.flatten()
#     vertices[:, 2] = hf.flatten() * vertical_scale
#     triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
#     for i in range(num_rows - 1):
#         ind0 = np.arange(0, num_cols - 1) + i * num_cols
#         ind1 = ind0 + 1
#         ind2 = ind0 + num_cols
#         ind3 = ind2 + 1
#         start = 2 * i * (num_cols - 1)
#         stop = start + 2 * (num_cols - 1)
#         triangles[start:stop:2, 0] = ind0
#         triangles[start:stop:2, 1] = ind3
#         triangles[start:stop:2, 2] = ind1
#         triangles[start + 1:stop:2, 0] = ind0
#         triangles[start + 1:stop:2, 1] = ind2
#         triangles[start + 1:stop:2, 2] = ind3
#
#     return vertices, triangles


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """ This is the vectorized version. Much faster than the original one. Generated from ChatGPT """

    if slope_threshold is None:
        raise ValueError('slope threshold cannot be None!!!')

    hf = height_field_raw
    num_rows, num_cols = hf.shape

    # Create a mesh grid for the x and y coordinates
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    # Adjust the slope threshold by the ratio of horizontal to vertical scales
    slope_threshold *= horizontal_scale / vertical_scale

    # Compute the adjustments for horizontal and vertical moves
    move_x = np.zeros((num_rows, num_cols))
    move_y = np.zeros((num_rows, num_cols))
    move_corners = np.zeros((num_rows, num_cols))

    move_x[:num_rows - 1, :] += (hf[1:num_rows, :] - hf[:num_rows - 1, :]) > slope_threshold
    move_x[1:num_rows, :] -= (hf[:num_rows - 1, :] - hf[1:num_rows, :]) > slope_threshold
    move_y[:, :num_cols - 1] += (hf[:, 1:num_cols] - hf[:, :num_cols - 1]) > slope_threshold
    move_y[:, 1:num_cols] -= (hf[:, :num_cols - 1] - hf[:, 1:num_cols]) > slope_threshold
    move_corners[:num_rows - 1, :num_cols - 1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows - 1, :num_cols - 1]) > slope_threshold
    move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows - 1, :num_cols - 1] - hf[1:num_rows, 1:num_cols]) > slope_threshold

    # Apply the horizontal shifts
    xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
    yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # Create vertices from the adjusted grid and scaled height values
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale

    # Vectorized triangle creation:
    # Create a grid of vertex indices
    grid = np.arange(num_rows * num_cols, dtype=np.uint32).reshape(num_rows, num_cols)

    # For each cell, extract the indices of its four corners:
    # A = top-left, B = top-right, C = bottom-left, D = bottom-right
    A = grid[:-1, :-1].ravel()
    B = grid[:-1, 1:].ravel()
    C = grid[1:, :-1].ravel()
    D = grid[1:, 1:].ravel()

    # Define two triangles per cell:
    # Triangle 1: [A, D, B]
    # Triangle 2: [A, C, D]
    triangles = np.empty((A.size * 2, 3), dtype=np.uint32)
    triangles[0::2] = np.stack([A, D, B], axis=1)
    triangles[1::2] = np.stack([A, C, D], axis=1)

    return vertices, triangles


def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles


def edge_detection(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    if slope_threshold is None:
        raise ValueError('slope threshold cannot be None!!!')

    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    slope_threshold *= horizontal_scale / vertical_scale
    move_x = np.zeros((num_rows, num_cols))
    move_y = np.zeros((num_rows, num_cols))
    move_x[:num_rows - 1, :] += hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold
    move_x[1:num_rows, :] -= hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
    move_y[:, :num_cols - 1] += hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold
    move_y[:, 1:num_cols] -= hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold

    # compute edge (fixed by hzx)
    move_x[:num_rows - 1, :] += hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold
    move_x[1:num_rows, :] += hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold
    move_x[1:num_rows, :] -= hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
    move_x[:num_rows - 1, :] -= hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
    move_y[:, :num_cols - 1] += hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold
    move_y[:, 1:num_cols] += hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold
    move_y[:, 1:num_cols] -= hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
    move_y[:, :num_cols - 1] -= hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold

    edge_x = move_x != 0
    edge_y = move_y != 0

    return edge_x + edge_y


# def add_fractal_roughness(terrain, levels=8, scale=1.0):
#     """
#     Generates a fractal terrain
#
#     Parameters
#         terrain (SubTerrain): the terrain
#         levels (int, optional): granurarity of the fractal terrain. Defaults to 8.
#         scale (float, optional): scales vertical variation. Defaults to 1.0.
#     """
#     width = terrain.width
#     length = terrain.length
#     height = np.zeros((width, length))
#     for level in range(1, levels + 1):
#         step = 2 ** (levels - level)
#         for y in range(0, width, step):
#             y_skip = (1 + y // step) % 2
#             for x in range(step * y_skip, length, step * (1 + y_skip)):
#                 x_skip = (1 + x // step) % 2
#                 xref = step * (1 - x_skip)
#                 yref = step * (1 - y_skip)
#                 mean = height[y - yref: y + yref + 1: 2 * step, x - xref: x + xref + 1: 2 * step].mean()
#                 variation = 2 ** (-level) * np.random.uniform(-1, 1)
#                 height[y, x] = mean + scale * variation
#
#     height /= terrain.vertical_scale
#     terrain.height_field_raw += height.astype(np.int16)
#     return terrain


def add_fractal_roughness(terrain, difficulty=1):
    """
        Generate 2D fractal noise using Perlin noise.

        Parameters:
        - shape: tuple (height, width) of the 2D array
        - scale: scaling factor for the noise
        - octaves: number of octaves (layers of noise)
        - persistence: controls the amplitude of each octave
        - lacunarity: controls the frequency of each octave
    """

    # flat terrain  (from Zipeng Fu)
    # for structured gait emergence:
    #     number of octaves = 2, fractal lacunarity = 2.0, fractal gain = 0.25, frequency = 10Hz, amplitude = 0.23 m;
    # uneven terrain
    # for unstructured gait emergence:
    #     number of octaves = 2, fractal lacunarity = 2.0, fractal gain = 0.25, frequency = 20Hz, amplitude = 0.27m

    octaves = 2
    lacunarity = 2.0
    gain = 0.25
    frequency = 10
    amplitude = (0.1 + 0.1 * difficulty) / terrain.vertical_scale

    height, width = terrain.height_field_raw.shape
    noise_array = np.zeros((height, width), dtype=terrain.height_field_raw.dtype)

    # Generate Perlin noise at different scales
    for y in range(height):
        for x in range(width):
            noise_value = 0
            amp = amplitude
            freq = frequency

            # Accumulate noise for each octave
            for i in range(octaves):
                noise_value += amp * noise.pnoise2(x / freq, y / freq, octaves=octaves, persistence=gain, lacunarity=lacunarity)
                freq *= lacunarity
                amp *= gain

            noise_array[y][x] = noise_value

    terrain.height_field_raw += noise_array


# def generate_perlin_noise_2d(shape, res):
#     delta = (res[0] / shape[0], res[1] / shape[1])
#     d = (shape[0] // res[0], shape[1] // res[1])
#     grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
#     # Gradients
#     angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
#     gradients = np.dstack((np.cos(angles), np.sin(angles)))
#     g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
#     g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
#     g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
#     g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
#     # Ramps
#     n00 = np.sum(grid * g00, 2)
#     n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
#     n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
#     n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
#     # Interpolation
#     t = 6 * grid ** 5 - 15 * grid ** 4 + 10 * grid ** 3
#     n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
#     n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
#     return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5


def generate_perlin_noise_2d(shape, res):
    delta = (res[0] / shape[0], res[1] / shape[1])
    # Use ceiling to ensure the repeated array is large enough
    d = (int(np.ceil(shape[0] / res[0])), int(np.ceil(shape[1] / res[1])))
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    # Repeat and then crop to match the grid's shape
    g00 = gradients[0:-1, 0:-1].repeat(d[0], axis=0).repeat(d[1], axis=1)[:shape[0], :shape[1], :]
    g10 = gradients[1:, 0:-1].repeat(d[0], axis=0).repeat(d[1], axis=1)[:shape[0], :shape[1], :]
    g01 = gradients[0:-1, 1:].repeat(d[0], axis=0).repeat(d[1], axis=1)[:shape[0], :shape[1], :]
    g11 = gradients[1:, 1:].repeat(d[0], axis=0).repeat(d[1], axis=1)[:shape[0], :shape[1], :]

    # Ramps
    n00 = np.sum(grid * g00, axis=2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=2)

    # Interpolation
    t = 6 * grid ** 5 - 15 * grid ** 4 + 10 * grid ** 3
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11

    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5


def generate_fractal_noise_2d(xSize=20, ySize=20, xSamples=1600, ySamples=1600, frequency=10,
                              fractalOctaves=2, fractalLacunarity=2.0, fractalGain=0.25, zScale=0.23):
    xScale = int(frequency * xSize)
    yScale = int(frequency * ySize)
    amplitude = 1
    noise = np.zeros((xSamples, ySamples))
    for _ in range(fractalOctaves):
        noise += amplitude * generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
        amplitude *= fractalGain
        xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

    return noise
