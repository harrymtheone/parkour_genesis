import torch

from legged_gym.envs.base.utils import CircularBuffer


class DataBuf:
    def __init__(self, n_envs, n_trans_per_env, shape, dtype, device):
        self.n_envs = n_envs
        self.n_trans_per_env = n_trans_per_env
        self.device = device

        self.buf = CircularBuffer(n_trans_per_env, n_envs, shape, dtype=dtype, device=self.device)

    def append(self, value):
        self.buf.append(value)

    def get_all(self):
        return self.buf.get_all()


class HiddenBuf:
    def __init__(self, n_envs, n_trans_per_env, shape, dtype, device):
        self.n_envs = n_envs
        self.n_trans_per_env = n_trans_per_env
        n_layers, _, hidden_size = shape
        self.device = device

        self.buf = CircularBuffer(n_trans_per_env, n_envs, (n_layers, hidden_size), dtype=dtype, device=self.device)

    def append(self, value):
        self.buf.append(value.transpose(0, 1))

    def get_all(self):
        return self.buf.get_all()[0].transpose(0, 1).contiguous()


class ObsTransBuf:
    def __init__(self, n_envs, n_trans_per_env, device):
        self.n_envs = n_envs
        self.n_trans_per_env = n_trans_per_env
        self.device = device

        self.storage = {}

    def init_buffer(self, obs):
        for n, v in obs.items():
            if n in self.storage or v is None:
                continue
            self.storage[n] = CircularBuffer(self.n_trans_per_env, self.n_envs, v.shape[1:], dtype=v.dtype, device=self.device)

    def append(self, obs):
        for n, v in obs.items():
            self.storage[n].append(v)

    def get_all(self):
        return {k: buf.get_all() for k, buf in self.storage.items()}


class OdometerStorage:
    def __init__(self, num_envs, storage_length, device):
        self.device = device
        self.storage_length = storage_length
        self.num_envs = num_envs

        self.storage = {}

        self.init_done = False
        self.cur_idx = 0

    def add_transitions(self, transition):

        if not self.init_done:
            with torch.inference_mode(mode=False):
                self._init_storage(transition)

        for n, v in transition.get_items():
            if n in self.storage:
                self.storage[n].append(v)

        self.cur_idx = min(self.cur_idx + 1, self.storage_length)

    def _init_storage(self, transition):
        for n, v in transition.get_items():
            if n.endswith('hidden_states'):
                self.storage[n] = HiddenBuf(self.num_envs, self.storage_length, v.shape, v.dtype, self.device)

            elif type(v) is torch.Tensor:
                # other data
                self.storage[n] = DataBuf(self.num_envs, self.storage_length, v.shape[1:], v.dtype, self.device)
            else:
                # observations
                self.storage[n] = ObsTransBuf(self.num_envs, self.storage_length, self.device)
                self.storage[n].init_buffer(v)

        self.init_done = True

    def recurrent_mini_batch_generator(self, num_epochs=8):
        valid_mask = ~(torch.cumsum(self.storage['dones'].get_all(), dim=0) > 0)

        for _ in range(num_epochs):
            batch_dict = {'masks': valid_mask}

            for n, v in self.storage.items():
                batch_dict[n] = v.get_all()

            yield batch_dict

    def is_full(self):
        return self.cur_idx == self.storage_length

    def clear(self):
        self.cur_idx = 0
