import torch


class DepthCam:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def _initialize_sensors(self):
        if not self.cfg.use_camera:
            return

            # camera properties
        cfg = self.cfg
        width, height = cfg.original
        horizontal_fov = cfg.horizontal_fov

        # camera randomization
        camera_position = np.copy(cfg.position)
        camera_position[0] += np.random.uniform(*cfg.position_range[0])
        camera_position[1] += np.random.uniform(*cfg.position_range[1])
        camera_position[2] += np.random.uniform(*cfg.position_range[2])
        camera_angle = math.radians(cfg.angle + np.random.uniform(*cfg.angle_range))

        u_0, v_0 = width / 2, height / 2
        f = width / 2 / math.tan(math.radians(horizontal_fov) / 2)

        # simple pinhole model
        K = wp.mat44(
            f, 0.0, u_0, 0.0,
            0.0, f, v_0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        self.K_inv = wp.inverse(K)
        self.c_x, self.c_y = int(u_0), int(v_0)

        self.cam_offset_pos_design[i] = torch.tensor(cfg.position, dtype=torch.float, device=self.device)  # for point cloud computation
        self.cam_offset_pos[i] = torch.tensor(camera_position, dtype=torch.float, device=self.device)

        self.cam_offset_quat_design[i] = quat_mul(
            quat_from_euler_xyz(*torch.tensor([0, math.radians(cfg.angle), 0], dtype=torch.float, device=self.device)),  # for point cloud computation
            quat_from_euler_xyz(*torch.deg2rad(torch.tensor([-90, 0, -90], device=self.device)))
        )
        self.cam_offset_quat[i] = quat_mul(
            quat_from_euler_xyz(*torch.tensor([0, camera_angle, 0], dtype=torch.float, device=self.device)),
            quat_from_euler_xyz(*torch.deg2rad(torch.tensor([-90, 0, -90], device=self.device)))
        )

        if not (cfg.use_warp and self.sim_device.startswith('cuda')):
            raise ValueError("Only Warp based sensor is now supported")


        #     camera_props = gymapi.CameraProperties()
        #     camera_props.width = width
        #     camera_props.height = height
        #     camera_props.enable_tensors = True
        #     camera_props.horizontal_fov = horizontal_fov
        #
        #     camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        #     self.cam_handles.append(camera_handle)
        #
        #     local_transform = gymapi.Transform()
        #     local_transform.p = gymapi.Vec3(*camera_position)
        #     local_transform.r = gymapi.Quat.from_euler_zyx(0, camera_angle, 0)
        #     root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
        #
        #     self.gym.attach_camera_to_body(
        #         camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

