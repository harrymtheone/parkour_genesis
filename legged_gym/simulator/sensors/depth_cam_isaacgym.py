import torch
import torchvision
from isaacgym import gymapi

from legged_gym.utils.math import xyz_to_quat, torch_rand_float
from .sensor_base import SensorBuffer


class DepthCamIsaacGym:
    def __init__(self, cfg_dict, device, simulator):
        self.cfg_dict = cfg_dict
        self.device = device
        self.simulator = simulator
        self.num_envs = cfg_dict['num_envs']

        self.name_link_attached_to = cfg_dict['link_attached_to']
        self.far_clip = cfg_dict['far_clip']
        self.near_clip = cfg_dict['near_clip']
        self.dis_noise_global = cfg_dict['dis_noise_global']
        self.dis_noise_gaussian = cfg_dict['dis_noise_gaussian']

        self.sensor_offset_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=device)
        self.sensor_offset_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=device)

        self.depth_raw = torch.zeros(self.num_envs, *reversed(self.cfg_dict['resolution']),
                                     dtype=torch.float, device=device)

        self.resize_transform = torchvision.transforms.Resize(
            (self.cfg_dict['resized'][1], self.cfg_dict['resized'][0]),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True
        )

        self.buf = SensorBuffer(self.num_envs,
                                cfg_dict['buf_len'],
                                (cfg_dict['resized'][1], cfg_dict['resized'][0]),
                                delay_prop=cfg_dict['delay_prop'],
                                device=device)

        self._initialize_sensors()

    def get(self, get_pos=False, get_raw=False, **kwargs):
        if get_pos:
            return self.sensor_pos.numpy()

        if get_raw:
            return self.depth_raw.clone()
        else:
            return self.buf.get()

    def step(self, reset):
        return self.buf.step(reset)

    def update(self):
        # ---------------------------  render by IsaacGym  ---------------------------
        self.simulator.render_camera(self.depth_raw)

    def post_process(self):
        # These operations are replicated on the hardware
        depth_image = self.depth_raw.clone()

        # crop 30 pixels from the left and right and 20 pixels from bottom and return croped image
        depth_image = depth_image[:, :-2, 4:-4]

        # add global distance noise
        depth_image[:] += torch_rand_float(-self.dis_noise_global, self.dis_noise_global, (self.num_envs, 1), self.device).unsqueeze(-1)

        # add Gaussian noise
        depth_image += torch.randn_like(depth_image) * self.dis_noise_gaussian

        # distance clip
        depth_image[:] = torch.clip(depth_image, self.near_clip, self.far_clip)

        # resize image
        depth_image = self.resize_transform(depth_image)

        # normalize the depth image to range (-0.5, 0.5)
        depth_image[:] = (depth_image - self.near_clip) / (self.far_clip - self.near_clip) - 0.5

        # self.depth_processed[:] = depth_image
        self.buf.append(depth_image)

    def _initialize_sensors(self):
        # camera properties
        cfg = self.cfg_dict
        camera_props = gymapi.CameraProperties()
        camera_props.width = cfg['resolution'][0]
        camera_props.height = cfg['resolution'][1]
        camera_props.enable_tensors = True
        camera_props.horizontal_fov = cfg['horizontal_fov']

        # camera randomization
        self.sensor_offset_pos[:] = torch.tensor(cfg['position'], dtype=torch.float, device=self.device).unsqueeze(0)
        self.sensor_offset_pos[:, 0:1] += torch_rand_float(*cfg['position_range'][0], shape=(self.num_envs, 1), device=self.device)
        self.sensor_offset_pos[:, 1:2] += torch_rand_float(*cfg['position_range'][1], shape=(self.num_envs, 1), device=self.device)
        self.sensor_offset_pos[:, 2:3] += torch_rand_float(*cfg['position_range'][2], shape=(self.num_envs, 1), device=self.device)

        camera_xyz_angle = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        camera_xyz_angle[:, 1:2] = cfg['pitch'] + torch_rand_float(*cfg['pitch_range'], shape=(self.num_envs, 1), device=self.device)
        self.sensor_offset_quat[:] = xyz_to_quat(torch.deg2rad(camera_xyz_angle))

        for env_i in range(self.num_envs):
            cam_trans = gymapi.Transform()
            cam_trans.p = gymapi.Vec3(*self.sensor_offset_pos[env_i])
            cam_trans.r = gymapi.Quat(*self.sensor_offset_quat[env_i])

            self.simulator.create_camera_sensor(env_i, self.name_link_attached_to, camera_props, cam_trans)

