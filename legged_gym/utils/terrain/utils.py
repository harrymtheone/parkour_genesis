import numpy as np
from pydelatin import Delatin


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2,y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    if slope_threshold is None:
        raise ValueError('slope threshold cannot be None!!!')

    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    slope_threshold *= horizontal_scale / vertical_scale
    move_x = np.zeros((num_rows, num_cols))
    move_y = np.zeros((num_rows, num_cols))
    move_corners = np.zeros((num_rows, num_cols))
    move_x[:num_rows - 1, :] += hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold
    move_x[1:num_rows, :] -= hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
    move_y[:, :num_cols - 1] += hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold
    move_y[:, 1:num_cols] -= hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
    move_corners[:num_rows - 1, :num_cols - 1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows - 1, :num_cols - 1] > slope_threshold)
    move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows - 1, :num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
    xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
    yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1:stop:2, 0] = ind0
        triangles[start + 1:stop:2, 1] = ind2
        triangles[start + 1:stop:2, 2] = ind3

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


def add_fractal_roughness(terrain, levels=8, scale=1.0):
    """
    Generates a fractal terrain

    Parameters
        terrain (SubTerrain): the terrain
        levels (int, optional): granurarity of the fractal terrain. Defaults to 8.
        scale (float, optional): scales vertical variation. Defaults to 1.0.
    """
    width = terrain.width
    length = terrain.length
    height = np.zeros((width, length))
    for level in range(1, levels + 1):
        step = 2 ** (levels - level)
        for y in range(0, width, step):
            y_skip = (1 + y // step) % 2
            for x in range(step * y_skip, length, step * (1 + y_skip)):
                x_skip = (1 + x // step) % 2
                xref = step * (1 - x_skip)
                yref = step * (1 - y_skip)
                mean = height[y - yref: y + yref + 1: 2 * step, x - xref: x + xref + 1: 2 * step].mean()
                variation = 2 ** (-level) * np.random.uniform(-1, 1)
                height[y, x] = mean + scale * variation

    height /= terrain.vertical_scale
    terrain.height_field_raw += height.astype(np.int16)
    return terrain
