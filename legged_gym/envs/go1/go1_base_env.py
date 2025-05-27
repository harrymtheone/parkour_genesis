import torch


@torch.jit.script
def mirror_dof_prop_by_x(prop: torch.Tensor, start_idx: int):
    left_idx = start_idx + torch.tensor([0, 1, 2, 6, 7, 8], dtype=torch.long, device=prop.device)
    right_idx = left_idx + 3

    dof_left = prop[:, left_idx].clone()
    prop[:, left_idx] = prop[:, right_idx]
    prop[:, right_idx] = dof_left

    invert_idx = start_idx + torch.tensor([0, 3, 6, 9], dtype=torch.long, device=prop.device)
    prop[:, invert_idx] *= -1.


@torch.jit.script
def mirror_proprio_by_x(prop: torch.Tensor) -> torch.Tensor:
    prop = prop.clone()

    # base angular velocity, [0:3], [-roll, pitch, -yaw]
    prop[:, 0] *= -1.
    prop[:, 2] *= -1.

    # projected gravity, [3:6], [x, -y, z]
    prop[:, 4] *= -1.

    # commands [6:9], [x, -y, -yaw]
    prop[:, 7:9] *= -1.

    # dof pos
    mirror_dof_prop_by_x(prop, 9)

    # dof vel
    mirror_dof_prop_by_x(prop, 9 + 12)

    # last actions
    mirror_dof_prop_by_x(prop, 9 + 12 + 12)

    return prop


@torch.jit.script
def mirror_priv_by_x(priv: torch.Tensor) -> torch.Tensor:
    priv = priv.clone()

    # base velocity, [0:3], [x, -y, z]
    priv[:, 1] *= -1.

    # feet height map, [3:19], flip
    feet_hmap = priv[:, 3:19].unflatten(1, (4, 2, 2))
    feet_hmap = torch.flip(feet_hmap, dims=[1, 3])
    priv[:, 3:19] = feet_hmap.flatten(1)

    # body height map, [19:35], flip
    body_hmap = priv[:, 19:35].unflatten(1, (4, 4))
    body_hmap = torch.flip(body_hmap, dims=[2])
    priv[:, 19:35] = body_hmap.flatten(1)

    return priv
