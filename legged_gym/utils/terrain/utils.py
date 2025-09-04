import numpy as np


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


def edge_detection(
        height_map: np.ndarray,
        horizontal_scale: float,
        vertical_scale: float,
        slope_threshold: float = None
) -> np.ndarray:
    if slope_threshold is None:
        raise ValueError('slope threshold cannot be None!!!')

    hf = height_map
    num_rows, num_cols = hf.shape

    # Scale threshold by horizontal distance
    height_threshold = slope_threshold * horizontal_scale / vertical_scale

    # Initialize edge map
    edges = np.zeros((num_rows, num_cols), dtype=bool)

    # Check x direction: if there's a large height difference between adjacent points,
    # mark BOTH points as edges
    x_diff = np.abs(hf[1:, :] - hf[:-1, :]) > height_threshold
    edges[:-1, :] |= x_diff  # Mark left points of edge pairs
    edges[1:, :] |= x_diff  # Mark right points of edge pairs

    # Check y direction: if there's a large height difference between adjacent points,
    # mark BOTH points as edges
    y_diff = np.abs(hf[:, 1:] - hf[:, :-1]) > height_threshold
    edges[:, :-1] |= y_diff  # Mark top points of edge pairs
    edges[:, 1:] |= y_diff  # Mark bottom points of edge pairs

    return edges


def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    from pydelatin import Delatin
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles


def generate_fractal_noise_2d(terrain_shape, horizontal_scale, difficulty, scale=0.5, octaves=2, persistence=1.0, lacunarity=2.0):
    """
    Generate 2D fractal noise using Perlin noise in a vectorized fashion.

    Parameters:
        shape (tuple): (height, width) of the output noise grid.
        horizontal_scale (float): Base horizontal scaling factor.
        scale (float): Frequency scale of the noise.
        octaves (int): Number of noise layers.
        persistence (float): Amplitude reduction per octave.
        lacunarity (float): Frequency multiplier per octave.

    Returns:
        np.ndarray: The generated fractal noise array.
    """
    import noise
    height, width = terrain_shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    noise_array = np.zeros((height, width))

    for octave in range(octaves):
        # Calculate the frequency and amplitude for the current octave.
        frequency = (horizontal_scale / scale) * (lacunarity ** octave)
        amplitude = persistence ** octave

        # Create a vectorized version of noise.pnoise2 for the current frequency.
        pnoise2_vec = np.vectorize(lambda xx, yy: noise.pnoise2(xx * frequency, yy * frequency, repeatx=width, repeaty=height))
        noise_layer = pnoise2_vec(x, y)
        noise_array += amplitude * noise_layer

    return noise_array * 0.1 * difficulty
