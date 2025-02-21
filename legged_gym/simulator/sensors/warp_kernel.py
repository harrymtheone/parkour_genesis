import warp as wp


@wp.kernel
def depth_only_kernel(
        mesh_id: wp.uint64,
        cam_pos_arr: wp.array1d(dtype=wp.vec3f),
        cam_quat_arr: wp.array1d(dtype=wp.quat),
        K_inv: wp.mat44,
        c_x: int,
        c_y: int,
        far_clip: float,
        depth_image: wp.array3d(dtype=float),
):
    # get the index for current pixel
    env_id, x, y = wp.tid()

    cam_pos = cam_pos_arr[env_id]
    cam_quat = cam_quat_arr[env_id]

    cam_coords = wp.vec3f(float(x), float(y), 1.0)
    cam_coords_principal = wp.vec3f(float(c_x), float(c_y), 1.0)  # get the vector of principal axis

    # transform to uv [-1,1]
    uv = wp.transform_vector(K_inv, cam_coords)
    uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis

    # convert to world frame
    uv_world = wp.vec3f(uv[2], -uv[0], -uv[1])
    uv_principal_world = wp.vec3f(uv_principal[2], -uv_principal[0], -uv_principal[1])

    # compute camera ray origin in world frame
    ro = cam_pos

    # tf the direction from camera to world frame and normalize
    rd = wp.normalize(wp.quat_rotate(cam_quat, uv_world))
    rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal_world))  # ray direction of principal axis

    # multiplier to project each ray on principal axis for depth instead of range
    multiplier = wp.dot(rd, rd_principal)

    # wp.printf('cam rd (%.2f, %.2f, %.2f)\n', rd[0], rd[1], rd[2])

    # perform ray casting
    query = wp.mesh_query_ray(mesh_id, ro, rd, far_clip / multiplier)

    dist = far_clip
    if query.result:
        # compute the depth of this pixel
        dist = multiplier * query.t

    depth_image[env_id, y, x] = dist


@wp.kernel
def depth_point_cloud_kernel(
        mesh_id: wp.uint64,
        cam_pos_design_arr: wp.array1d(dtype=wp.vec3f),  # camera position and quaterion by design
        cam_quat_design_arr: wp.array1d(dtype=wp.quat),
        cam_pos_arr: wp.array1d(dtype=wp.vec3f),  # camera position and quaterion by randomization
        cam_quat_arr: wp.array1d(dtype=wp.quat),
        K_inv: wp.mat44,
        c_x: int,
        c_y: int,
        far_clip: float,
        depth_image: wp.array4d(dtype=float),
        cloud: wp.array3d(dtype=wp.vec3f),
        cloud_valid: wp.array3d(dtype=bool)
):
    # get the index for current pixel
    env_id, x, y = wp.tid()

    cam_pos = cam_pos_arr[env_id]
    cam_quat = cam_quat_arr[env_id]

    cam_coords = wp.vec3(float(x), float(y), 1.0)  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
    cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)  # get the vector of principal axis

    # transform to uv [-1,1]
    uv = wp.transform_vector(K_inv, cam_coords)
    uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis

    # compute camera ray origin in world frame
    ro = cam_pos

    # tf the direction from camera to world frame and normalize
    rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
    rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))  # ray direction of principal axis

    # multiplier to project each ray on principal axis for depth instead of range
    multiplier = wp.dot(rd, rd_principal)

    # perform ray casting
    query = wp.mesh_query_ray(mesh_id, ro, rd, far_clip / multiplier)

    dist = far_clip
    cloud_valid[env_id, y, x] = query.result
    if query.result:
        # compute the depth of this pixel
        dist = multiplier * query.t

        # compute the position of the pixel in world frame
        ro_design = cam_pos_design_arr[env_id]
        rd_design = wp.normalize(wp.quat_rotate(cam_quat_design_arr[env_id], uv))

        pt_pos = ro_design + rd_design * query.t
        cloud[env_id, y, x] = pt_pos

    depth_image[env_id, 0, y, x] = dist


@wp.kernel
def point_cloud_depth_to_voxel_grid_accumulation_kernel(
        grid_shape: wp.vec3i,
        grid_size: float,
        trans_base: wp.array1d(dtype=wp.vec3f),
        quat_base: wp.array1d(dtype=wp.quatf),
        trans_voxel: wp.vec3f,
        cloud: wp.array3d(dtype=wp.vec3f),
        cloud_valid: wp.array3d(dtype=bool),
        voxel_grid_accumulation: wp.array4d(dtype=wp.vec4f)
):
    # get the index for current pixel
    env_id, pt_x, pt_y = wp.tid()

    if not cloud_valid[env_id, pt_x, pt_y]:
        return

    # convert point into base frame
    point_base = cloud[env_id, pt_x, pt_y] - trans_base[env_id]
    point_base = wp.quat_rotate(wp.quat_inverse(quat_base[env_id]), point_base)

    # convert point into voxel grid frame
    point_voxel = point_base - trans_voxel

    # Determine voxel indices for current point
    voxel_index_x = int(wp.floor(point_voxel.x / grid_size))
    voxel_index_y = int(wp.floor(point_voxel.y / grid_size))
    voxel_index_z = int(wp.floor(point_voxel.z / grid_size))

    if (voxel_index_x < 0 or voxel_index_x >= grid_shape.x or
            voxel_index_y < 0 or voxel_index_y >= grid_shape.y or
            voxel_index_z < 0 or voxel_index_z >= grid_shape.z):
        return

    # Determine voxel position for current point
    voxel_pos = wp.vec3f((point_voxel.x % grid_size) / grid_size,
                         (point_voxel.y % grid_size) / grid_size,
                         (point_voxel.z % grid_size) / grid_size)

    # fill in the output variable
    wp.atomic_add(voxel_grid_accumulation,
                  env_id, voxel_index_x, voxel_index_y, voxel_index_z,
                  wp.vec4f(1., voxel_pos.x, voxel_pos.y, voxel_pos.z))


@wp.kernel
def point_cloud_to_voxel_grid_kernel(
        voxel_grid_accumulation: wp.array4d(dtype=wp.vec4f),
        voxel_grid: wp.array4d(dtype=wp.vec4f)
):
    env_id, x, y, z = wp.tid()

    voxel = voxel_grid_accumulation[env_id, x, y, z]

    if voxel.x > 0:
        com_x = voxel.y / voxel.x
        com_y = voxel.z / voxel.x
        com_z = voxel.w / voxel.x
        voxel_grid[env_id, x, y, z] = wp.vec4f(1., com_x, com_y, com_z)


@wp.kernel
def generate_voxel_grid_terrain_kernel(
        grid_size: float,
        trans_base: wp.array1d(dtype=wp.vec3f),
        quat_base: wp.array1d(dtype=wp.quatf),
        trans_voxel: wp.vec3f,
        volume_world_id: wp.uint64,
        volume_occu: wp.array1d(dtype=float),
        volume_com: wp.array1d(dtype=wp.vec3f),
        voxel_grid: wp.array4d(dtype=wp.vec4f),
):
    env_id, x, y, z = wp.tid()

    # get sub-voxel position in voxel grid frame
    point_voxel = wp.vec3f(wp.float32(x), wp.float32(y), wp.float32(z))
    point_voxel = (point_voxel + wp.vec3f(0.5, 0.5, 0.5)) * grid_size

    # convert point into base frame
    point_base = point_voxel + trans_voxel

    # convert point into world frame
    point_world = wp.quat_rotate(quat_base[env_id], point_base)
    point_world += trans_base[env_id]

    index = wp.volume_world_to_index(volume_world_id, point_world)

    pos_weight = wp.volume_sample_index(volume_world_id, index, wp.Volume.LINEAR, volume_occu, 0.)
    # TODO: this one should achieve sub-voxel accuracy
    pos_sample = wp.volume_sample_index(volume_world_id, index, wp.Volume.LINEAR, volume_com, wp.vec3f(0., 0., 0.))

    if pos_weight > 0.:
        com = pos_sample / pos_weight
        voxel_grid[env_id, x, y, z] = wp.vec4f(
            1.,
            com.x % grid_size,
            com.y % grid_size,
            com.z % grid_size
        )
