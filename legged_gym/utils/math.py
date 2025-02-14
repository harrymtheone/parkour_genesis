from typing import Tuple

import torch

import genesis as gs


# --------------------- Quaternion: (x, y, z, w) ---------------------

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], torch.device) -> torch.Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def normalize(x):
    # type: (torch.Tensor) -> torch.Tensor
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * torch.pi
    return angles - 2 * torch.pi * (angles > torch.pi)


@torch.jit.script
def inv_quat(quat):
    # type: (torch.Tensor) -> torch.Tensor
    scaling = torch.tensor([-1, -1, -1, 1], device=quat.device)
    return quat * scaling


@torch.jit.script
def axis_angle_to_quat(angle, axis):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def xyz_to_quat(euler_xyz):
    # type: (torch.Tensor) -> torch.Tensor
    euler_xyz = euler_xyz * torch.tensor(torch.pi) / 180.0
    roll, pitch, yaw = euler_xyz.unbind(-1)
    cosr = (roll * 0.5).cos()
    sinr = (roll * 0.5).sin()
    cosp = (pitch * 0.5).cos()
    sinp = (pitch * 0.5).sin()
    cosy = (yaw * 0.5).cos()
    siny = (yaw * 0.5).sin()
    qw = cosr * cosp * cosy + sinr * sinp * siny
    qx = sinr * cosp * cosy - cosr * sinp * siny
    qy = cosr * sinp * cosy + sinr * cosp * siny
    qz = cosr * cosp * siny - sinr * sinp * cosy
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def quat_to_xyz(quat):
    # type: (torch.Tensor) -> torch.Tensor
    # Extract quaternion components
    qx, qy, qz, qw = quat.unbind(-1)
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(torch.pi / 2),
        torch.asin(sinp),
    )
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1) * 180.0 / torch.tensor(torch.pi)


@torch.jit.script
def transform_by_quat(v, quat):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    quat = quat.reshape(-1, 4)
    v = v.reshape(-1, 3)
    qvec = quat[:, :3]
    t = qvec.cross(v, dim=-1) * 2
    return v + quat[:, 3:] * t + qvec.cross(t, dim=-1)


# @torch.jit.script
def transform_quat_by_quat(v, u):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    assert v.shape == u.shape, f"{v.shape} != {u.shape}"
    shape = u.shape
    u = u.reshape(-1, 4)
    v = v.reshape(-1, 4)
    x1, y1, z1, w1 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]
    x2, y2, z2, w2 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1).view(shape)


@torch.jit.script
def transform_by_yaw(v, yaw):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    quat_yaw = axis_angle_to_quat(
        yaw,
        torch.tensor([0, 0, 1], device=yaw.device, dtype=torch.float)
    )
    return transform_by_quat(v, quat_yaw)


# def depth_to_point_cloud(depth: np.ndarray,
#                          proj: np.matrix,
#                          v_inv: np.matrix,
#                          far_clip: float,
#                          near_clip: float) -> np.ndarray:
#     height, width = depth.shape  # (60, 106)
#     u = np.arange(0, width)[np.newaxis, :].repeat(height, axis=0)
#     v = np.arange(0, height)[:, np.newaxis].repeat(width, axis=1)
#
#     fu, fv = 2 / proj[0, 0], 2 / proj[1, 1]
#     centerU, centerV = width / 2, height / 2
#
#     Z = -depth
#     X = -(u - centerU) / width * Z * fu
#     Y = (v - centerV) / height * Z * fv
#
#     Z, X, Y = Z.flatten(), X.flatten(), Y.flatten()
#     valid = (Z > -far_clip) & (Z < -near_clip)
#
#     position = np.stack((X, Y, Z, np.ones(X.shape)), axis=0).T
#     position = position @ v_inv
#
#     return np.array(position[valid, :3])
#
#
# @torch.jit.script
# def point_cloud_to_voxel_grid_jit(cloud: List[torch.Tensor],
#                                   base_quat: torch.Tensor,
#                                   root_states: torch.Tensor,
#                                   voxel_translation: torch.Tensor,
#                                   grid_shape: Tuple[int, int, int],
#                                   grid_size: float,
#                                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
#     num_cloud = len(cloud)
#     quat_inv = base_quat * torch.tensor([[-1, -1, -1, 1]], dtype=torch.float, device=device)
#     grid_shape_tensor = torch.tensor(grid_shape, dtype=torch.float, device=device)[None, :] - 1
#
#     # init buffer to store voxel grid
#     grid_com = torch.zeros(num_cloud, *grid_shape, 3, dtype=torch.float, device=device)
#     grid_occupied = torch.zeros(num_cloud, *grid_shape, dtype=torch.bool, device=device)
#     voxel_point_count = torch.zeros(num_cloud, *grid_shape, dtype=torch.int, device=device)
#     voxel_sum_position = torch.zeros(num_cloud, *grid_shape, 3, dtype=torch.float, device=device)
#     p_cloud = torch.zeros(max([len(c) for c in cloud]), 3, dtype=torch.float, device=device)
#
#     for i in range(num_cloud):
#         p_cloud[:len(cloud[i])] = cloud[i]
#         p_cloud[len(cloud[i]):] = 0
#
#         # convert cloud to base frame
#         cloud_base = p_cloud - root_states[i, :3]
#         cloud_base = quat_apply_yaw(quat_inv[[i]], cloud_base)  # TODO performance bottleneck: compute before for loop
#
#         # convert points into voxel grid frame
#         cloud_voxel = cloud_base - voxel_translation
#
#         # Determine voxel indices for each point
#         voxel_indices = torch.floor(cloud_voxel / grid_size).to(torch.long)
#         cloud_grid = torch.remainder(cloud_voxel, grid_size)
#
#         # To avoid out-of-bounds indexing, we clamp indices to the grid shape
#         voxel_in_range = (torch.all(voxel_indices > 0, dim=1) &
#                           torch.all(voxel_indices < grid_shape_tensor, dim=1))
#         voxel_indices[:] = torch.clip(voxel_indices, torch.zeros(1, 3, device=device), grid_shape_tensor)
#         p_cloud[~voxel_in_range] = 0
#         cloud_grid[~voxel_in_range] = 0
#
#         # # Accumulate the sum of positions and the count of points per voxel
#         # for idx, voxel in enumerate(voxel_indices):  (TOO SLOW! DO NOT USE THIS ONE!!!)
#         #     voxel_sum_position[tuple(voxel)] += cloud_in_range[idx]  # this one determines the frame of COM
#         #     voxel_point_count[tuple(voxel)] += 1
#
#         # Accumulate the sum of positions and the count of points per voxel
#         voxel_sum_position_flat = voxel_sum_position[i].view(-1, 3)  # Flatten the voxel grid to (32*16*32, 3)
#         voxel_indices_flat = voxel_indices[:, 0] * (grid_shape[1] * grid_shape[2]) + voxel_indices[:, 1] * grid_shape[2] + voxel_indices[:, 2]
#         # voxel_sum_position_flat.scatter_add_(0, voxel_indices_flat.unsqueeze(-1).expand(-1, 3), p_cloud)  # this one defines which frame points in
#         voxel_sum_position_flat.scatter_add_(0, voxel_indices_flat.unsqueeze(-1).expand(-1, 3), cloud_grid)
#
#         voxel_point_count_flat = voxel_point_count[i].view(-1)  # Flatten the voxel grid
#         voxel_in_range_flat = voxel_in_range.to(torch.int).view(-1)
#         voxel_point_count_flat.scatter_add_(0, voxel_indices_flat, voxel_in_range_flat)
#
#     # compute COM and is_occupied of each grid
#     non_empty_voxels = voxel_point_count > 0
#     grid_com[non_empty_voxels] = voxel_sum_position[non_empty_voxels] / voxel_point_count[non_empty_voxels][:, None]
#     grid_occupied[non_empty_voxels] = True
#
#     return grid_occupied, grid_com
#
#
# @torch.jit.script
# def tri_mesh_to_point_cloud_jit(base_quat: torch.Tensor,
#                                 root_states: torch.Tensor,
#                                 vertices_ROI: torch.Tensor,
#                                 cloud: torch.Tensor) -> List[torch.Tensor]:
#     num_env = len(base_quat)
#
#     # convert ROI to world frame
#     ROI_world = quat_apply_yaw(base_quat.repeat(1, 4), vertices_ROI.repeat(num_env, 1))
#     ROI_world = ROI_world.view(num_env, 4, 3) + root_states[:, None, :3]  # (num_env, 4, 3)
#
#     # Define two triangles from the rectangle
#     triangles = torch.stack([
#         ROI_world[:, None, :3],
#         torch.cat([ROI_world[:, None, [0]], ROI_world[:, None, 2:4]], dim=2)
#     ], dim=1)  # (num_env, 2, 3, 3)
#     triangles = triangles.view(-1, 3, 3)  # (num_env * 2, 3, 3)
#
#     # Create batched vectors for both triangles
#     edge0 = triangles[:, 1] - triangles[:, 0]  # Shape (2, 3)
#     edge1 = triangles[:, 2] - triangles[:, 0]  # Shape (2, 3)
#     pt_edge = cloud[None, :, :] - triangles[:, None, 0]  # (2, N, 3)
#
#     # Compute dot products for all points in both triangles
#     dot00 = torch.sum(edge1 * edge1, dim=1, keepdim=True)  # Shape (2, 1)
#     dot01 = torch.sum(edge1 * edge0, dim=1, keepdim=True)  # Shape (2, 1)
#     dot02 = torch.sum(edge1.unsqueeze(1) * pt_edge, dim=2)  # Shape (2, N)
#     dot11 = torch.sum(edge0 * edge0, dim=1, keepdim=True)  # Shape (2, 1)
#     dot12 = torch.sum(edge0.unsqueeze(1) * pt_edge, dim=2)  # Shape (2, N)
#
#     # Compute barycentric coordinates
#     invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)  # Shape (2, 1)
#     u = (dot11 * dot02 - dot01 * dot12) * invDenom  # Shape (2, N)
#     v = (dot00 * dot12 - dot01 * dot02) * invDenom  # Shape (2, N)
#
#     # # Check if the points are inside either of the triangles
#     u, v = u.view(num_env, 2, -1), v.view(num_env, 2, -1)
#     inside = (u >= 0) & (v >= 0) & (u + v <= 1)
#     inside = inside[:, 0] | inside[:, 1]
#
#     cloud = cloud.clone()
#     return [cloud[ins] for ins in inside]  # TODO: Bottleneck
