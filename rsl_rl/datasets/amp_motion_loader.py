import joblib
from pathlib import Path
import torch
from collections import deque
import numpy as np
from legged_gym.utils.math import interpolate_projected_gravity_batch


class AMPMotionLoader:
    def __init__(self,
                 motion_cfg,
                 env_dt,
                 amp_num_frames,
                 device) -> None:
        folder = Path(motion_cfg['motion_file'])
        self.sample_dt = env_dt
        self.amp_num_frames = amp_num_frames
        self.motion_clip_length = (amp_num_frames - 1) * self.sample_dt
        self.device = device
        self.motion_weight = []
        motion_data = []
        motion_dict = {}
        stand_data = []
        for file_path in folder.rglob('*'):
            motion_dict = joblib.load(file_path)
            self.motion_weight.append(motion_dict.get('weight', 1.0))
            motion_data.append(motion_dict['data'].to(torch.float32))
            if 'stand' in file_path.name:
                stand_data.append(True)
            else:
                stand_data.append(False)
        self.stand_data = torch.tensor(stand_data, dtype=torch.bool).to(self.device)
        self.motion_data_idx = motion_dict['data_idx']
        self.motion_dt = 1 / motion_dict['fps']
        self.motion_weight = torch.tensor(self.motion_weight, dtype=torch.float32).to(self.device)
        self.motion_weight = self.motion_weight / self.motion_weight.sum()

        self.motion_num_frames = torch.tensor([data.shape[0] for data in motion_data], device=self.device)
        motion_frames = self.motion_num_frames.clone()
        motion_frames_shifted = motion_frames.roll(1)
        motion_frames_shifted[0] = 0
        self.motion_frame_starts = motion_frames_shifted.cumsum(0)
        self.motion_length = torch.tensor([(data.shape[0] - 1) * self.motion_dt for data in motion_data], device=self.device)

        motion_data = torch.cat(motion_data, dim=0).to(self.device)
        amp_motion_data = []
        start_idx = 0
        self.slerp_idx = []

        for obs_name, obs_info in motion_cfg['amp_obs_dict'].items():
            if obs_info["using"]:
                idx = self.motion_data_idx[obs_name]
                scale = torch.tensor(obs_info["obs_scale"], dtype=torch.float32).to(self.device)

                if obs_info.get("interpolate", None) == "slerp":
                    self.slerp_idx += list(range(start_idx, start_idx + obs_info["size"]))

                start_idx += obs_info["size"]
                amp_motion_data.append(motion_data[:, idx] * scale)
        self.motion_data = torch.cat(amp_motion_data, dim=1)

        if motion_cfg.get("using_ref_amp_data_init", False):
            self.ref_dof_pos = motion_data[:, self.motion_data_idx["dof_pos"]].float()
            self.ref_dof_vel = motion_data[:, self.motion_data_idx["dof_vel"]].float()
            self.root_rot_vel = motion_data[:, self.motion_data_idx["root_rot_vel"]].float()

        self.preload = motion_cfg.get('preload', False)
        if self.preload:
            self.num_preload_data = motion_cfg.get('num_preload_data', 100000)
            self.preload_motion_state = self.get_motion_state(self.num_preload_data)
            del self.motion_data, motion_data, amp_motion_data

    def sample_motion_idx_time(self, num_samples):
        motion_idx = torch.multinomial(self.motion_weight, num_samples=num_samples, replacement=True).to(self.device)
        motion_time = (self.motion_length[motion_idx] - self.motion_clip_length) * torch.rand(num_samples, device=self.device)
        return motion_idx, motion_time

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)

        return frame_idx0, frame_idx1, blend

    def get_single_motion_state(self, motion_time, motion_length, motion_num_frames, sample_idx):
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_time, motion_length, motion_num_frames, self.motion_dt)
        f_idx0 = frame_idx0 + self.motion_frame_starts[sample_idx]
        f_idx1 = frame_idx1 + self.motion_frame_starts[sample_idx]
        sample_motion_data = self.motion_data[f_idx0] * (1 - blend.unsqueeze(-1)) + self.motion_data[f_idx1] * blend.unsqueeze(-1)

        if len(self.slerp_idx) > 0:
            n = len(self.slerp_idx) // 3
            low_slerp = self.motion_data[f_idx0][:, self.slerp_idx].reshape(-1, 3)
            high_slerp = self.motion_data[f_idx1][:, self.slerp_idx].reshape(-1, 3)
            blend_expand = blend.repeat_interleave(n, 0)
            slerp_data = interpolate_projected_gravity_batch(low_slerp, high_slerp, blend_expand)
            sample_motion_data[:, self.slerp_idx] = slerp_data.reshape(-1, n * 3)
        return sample_motion_data

    def get_motion_state(self, num_samples):
        sample_motion_data_buf = deque(maxlen=self.amp_num_frames)
        motion_idx, motion_time = self.sample_motion_idx_time(num_samples)
        for _ in range(self.amp_num_frames):
            sample_motion_data = self.get_single_motion_state(motion_time, self.motion_length[motion_idx], self.motion_num_frames[motion_idx], motion_idx)
            sample_motion_data_buf.append(sample_motion_data)
            motion_time += self.sample_dt
        motion_state = torch.cat([sample_motion_data_buf[i] for i in range(self.amp_num_frames)], dim=1)
        return motion_state

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload:
                idxs = np.random.choice(self.num_preload_data, size=mini_batch_size)
                s = self.preload_motion_state[idxs, :]
            else:
                s = self.get_motion_state(mini_batch_size)
            yield s

    def sample_ref_idx(self, num_samples):
        return torch.randint(0, self.ref_dof_pos.shape[0], (num_samples,), device=self.device)

    def get_ref_dof_state(self, motion_idx):
        ref_dof_pos = self.ref_dof_pos[motion_idx]
        ref_dof_vel = self.ref_dof_vel[motion_idx]
        return ref_dof_pos, ref_dof_vel

    def get_root_rot_vel(self, motion_idx):
        return self.root_rot_vel[motion_idx]