from typing import Tuple

import numpy as np
import torch


# --------------------- Quaternion: (x, y, z, w) ---------------------

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], torch.device) -> torch.Tensor
    return lower + (upper - lower) * torch.rand(*shape, device=device)


@torch.jit.script
def normalize(x):
    # type: (torch.Tensor) -> torch.Tensor
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)


@torch.jit.script
def wrap_to_pi(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


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
        torch.sign(sinp) * torch.pi / 2,
        torch.asin(sinp),
    )
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return wrap_to_pi(torch.stack([roll, pitch, yaw], dim=-1))


@torch.jit.script
def quat_to_mat(quat):
    # type: (torch.Tensor) -> torch.Tensor
    x, y, z, w = quat.reshape(-1, 4).unbind(-1)
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], dim=-1)
    row1 = torch.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], dim=-1)
    row2 = torch.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1)

    rot_matrix = torch.stack([row0, row1, row2], dim=-2)  # shape (..., 3, 3)
    return rot_matrix


@torch.jit.script
def transform_by_quat(v, quat):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    v = v.reshape(-1, 3)
    quat = quat.reshape(-1, 4)
    qvec = quat[:, :3]
    t = qvec.cross(v, dim=-1) * 2
    return v + quat[:, 3:] * t + qvec.cross(t, dim=-1)


@torch.jit.script
def transform_quat_by_quat(v, u):  # result = u * v, notice the order! Rotate by v first, then u
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    assert v.shape == u.shape, f"{v.shape} != {u.shape}"
    shape = u.shape
    u = u.reshape(-1, 4)
    v = v.reshape(-1, 4)
    x1, y1, z1, w1 = u.unbind(-1)
    x2, y2, z2, w2 = v.unbind(-1)
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
def transform_by_trans_quat(vec, trans, quat):
    return transform_by_quat(vec, quat).view(vec.shape) + trans


@torch.jit.script
def transform_by_yaw(v, yaw):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    quat_yaw = axis_angle_to_quat(
        yaw,
        torch.tensor([0, 0, 1], device=yaw.device, dtype=torch.float)
    )
    return transform_by_quat(v, quat_yaw)


@torch.jit.script
def quat_rotate_inverse(quat, v):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    return transform_by_quat(v, inv_quat(quat))


def density_weighted_sampling(points, num_samples, k=10):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)

    # Estimate density as the mean distance to k-nearest neighbors
    density = np.mean(distances, axis=1)

    # Higher density -> lower probability of being sampled
    probabilities = density / np.sum(density)
    probabilities = 1 - probabilities  # Invert probabilities for uniformity
    probabilities /= np.sum(probabilities)

    # Sample points based on computed probabilities
    num_samples = min(num_samples, len(points) - 1)
    sampled_indices = np.random.choice(len(points), size=num_samples, replace=False, p=probabilities)

    return points[sampled_indices]


def interpolate_projected_gravity_batch(pg0, pg1, t):
    """
    对两个重力投影向量进行批量球面线性插值

    Args:
        pg0: 起始重力投影向量，shape [B, 3]
        pg1: 结束重力投影向量，shape [B, 3]
        t: 插值参数，shape [B] 或 [B, 1]，范围0到1

    Returns:
        插值后的重力投影向量，shape [B, 3]
    """

    batch_size = pg0.shape[0]

    # 处理t的形状
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # [B] -> [B, 1]

    # 归一化输入向量（确保它们是单位向量）
    pg0_norm = pg0 / torch.norm(pg0, dim=-1, keepdim=True)
    pg1_norm = pg1 / torch.norm(pg1, dim=-1, keepdim=True)

    # 计算两向量间的点积（余弦值）
    cos_theta = torch.sum(pg0_norm * pg1_norm, dim=-1, keepdim=True)  # [B, 1]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # 处理向量方向相反的情况（选择较短的弧）
    pg1_adjusted = torch.where(cos_theta < 0, -pg1_norm, pg1_norm)
    cos_theta = torch.abs(cos_theta)

    # 计算角度
    theta = torch.acos(cos_theta)  # [B, 1]
    sin_theta = torch.sin(theta)  # [B, 1]

    # 计算SLERP权重
    ratio0 = torch.sin((1 - t) * theta) / sin_theta  # [B, 1]
    ratio1 = torch.sin(t * theta) / sin_theta  # [B, 1]

    # 执行球面线性插值
    result_slerp = ratio0 * pg0_norm + ratio1 * pg1_adjusted  # [B, 3]

    # 处理向量几乎平行的情况（sin_theta接近0）
    # 在这种情况下使用线性插值然后归一化
    result_linear = (1 - t) * pg0_norm + t * pg1_adjusted
    result_linear = result_linear / torch.norm(result_linear, dim=-1, keepdim=True)

    # 选择适当的插值方法
    parallel_mask = torch.abs(sin_theta) < 1e-6  # [B, 1]
    result = torch.where(parallel_mask, result_linear, result_slerp)

    # 确保结果是单位向量（数值稳定性）
    result = result / torch.norm(result, dim=-1, keepdim=True)

    return result
