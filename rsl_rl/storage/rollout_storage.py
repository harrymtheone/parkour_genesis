import math

import torch


class DataBuf:
    def __init__(self, n_trans_per_env, n_envs, shape: tuple, device):
        self.n_trans_per_env = n_trans_per_env
        self.n_envs = n_envs
        self._shape = shape
        self.device = device

        # self.buf = torch.zeros(n_trans_per_env, n_envs, *shape, dtype=dtype, device=self.device)
        self.buf = None

    def numel(self):
        return math.prod(self._shape)

    def init_buf(self, buf):
        self.buf = buf.view(self.n_trans_per_env, self.n_envs, *self._shape)

    def set(self, idx, value):
        self.buf[idx] = value

    def get(self, slice_):
        return self.buf[slice_]

    def flatten_get(self, idx):
        return self.buf.flatten(0, 1)[idx]


class HiddenBuf:
    def __init__(self, n_trans_per_env, n_envs, shape, device):
        self.n_trans_per_env = n_trans_per_env
        self.n_envs = n_envs
        self._shape = shape
        self.device = device

        # self.buf = torch.zeros(n_trans_per_env, n_layers, n_envs, hidden_size, dtype=dtype, device=self.device)
        self.buf = None

    def numel(self):
        return math.prod(self._shape)

    def init_buf(self, buf):
        n_layers, hidden_size = self._shape
        self.buf = buf.view(self.n_trans_per_env, n_layers, self.n_envs, hidden_size)

    def set(self, idx, value):
        self.buf[idx] = value

    def get(self, slice_):
        return self.buf[0][slice_].contiguous()


class ObsTransBuf:
    def __init__(self, cfg, n_trans_per_env, n_envs, device):
        self.cfg = cfg
        self.n_trans_per_env = n_trans_per_env
        self.n_envs = n_envs
        self.device = device

        self.storage = {}

    def numel(self):
        return sum([math.prod(s) for s in self.cfg2dict(self.cfg).values()])

    def init_buf(self, buf):
        cur_idx = 0
        for k, shape in self.cfg2dict(self.cfg).items():
            self.storage[k] = buf[:, :, cur_idx:cur_idx + math.prod(shape)].unflatten(2, shape)
            cur_idx += math.prod(shape)

    @staticmethod
    def cfg2dict(cfg):
        cfg_dict = {}

        for k in dir(cfg):
            if k.startswith('_'):
                continue

            tensor_shape = getattr(cfg, k)
            if tensor_shape is None:
                continue

            tensor_shape = tensor_shape if isinstance(tensor_shape, tuple) else (tensor_shape,)
            cfg_dict[k] = tensor_shape

        return cfg_dict

    def set(self, idx, obs):
        for n, v in obs.items():
            if n in self.storage:
                self.storage[n][idx] = v.clone()

    def flatten_get(self, idx):
        param = [v.flatten(0, 1)[idx] for v in self.storage.values()]
        return self.obs_class(*param)

    def get(self, slice_):
        param = []
        for v in self.storage.values():
            param.append(v[slice_])

        return self.obs_class(*param)


class RolloutStorage:
    def __init__(self, task_cfg, device):
        self.num_envs = task_cfg.env.num_envs
        self.n_trans_per_env = task_cfg.runner.num_steps_per_env
        self.device = device

        # storage size of actor observation
        # self._storage_size = {
        #     'observation': ObsTransBuf.numel(task_cfg.env.obs),
        #     'critic_observation': ObsTransBuf.numel(task_cfg.env.critic_obs),
        #     'rewards': 1,
        #     'dones': 1,
        #     'values': 1,
        #     'returns': 1,
        #     'advantages': 1,
        #     'actions': task_cfg.env.num_actions,
        #     'actions_log_prob': task_cfg.env.num_actions,
        #     'action_mean': task_cfg.env.num_actions,
        #     'action_sigma': task_cfg.env.num_actions,
        #     'use_estimated_values': 1,
        # }
        self._storage_tensor = None

        self.storage = {
            'observation': ObsTransBuf(task_cfg.env.obs, self.num_envs, self.n_trans_per_env, self.device),
            'critic_observation': ObsTransBuf(task_cfg.env.critic_obs, self.num_envs, self.n_trans_per_env, self.device),
            'rewards': DataBuf(self.n_trans_per_env, self.num_envs, (1,), self.device),
            'dones': DataBuf(self.n_trans_per_env, self.num_envs, (1,), self.device),
            'values': DataBuf(self.n_trans_per_env, self.num_envs, (1,), self.device),
            'returns': DataBuf(self.n_trans_per_env, self.num_envs, (1,), self.device),
            'advantages': DataBuf(self.n_trans_per_env, self.num_envs, (1,), self.device),
            'actions': DataBuf(self.n_trans_per_env, self.num_envs, (task_cfg.env.num_actions,), self.device),
            'actions_log_prob': DataBuf(self.n_trans_per_env, self.num_envs, (task_cfg.env.num_actions,), self.device),
            'action_mean': DataBuf(self.n_trans_per_env, self.num_envs, (task_cfg.env.num_actions,), self.device),
            'action_sigma': DataBuf(self.n_trans_per_env, self.num_envs, (task_cfg.env.num_actions,), self.device),
            'use_estimated_values': DataBuf(self.n_trans_per_env, self.num_envs, (1,), self.device),
        }

        # self.returns_default = torch.zeros(n_trans_per_env, num_envs, 1, device=self.device)
        # self.returns_contact = torch.zeros(n_trans_per_env, num_envs, 1, device=self.device)
        # self.advantages = torch.zeros(n_trans_per_env, num_envs, 1, device=self.device)
        #
        # from legged_gym.envs.base.utils import ObsBase
        # self.obs_base_cls = ObsBase
        #
        # self.init_done = False
        # self.step = 0

    def register_storage(self, name: str, data_type: int, shape: tuple = None, cfg=None):
        """ data_type: 0 for raw data, 1 for hidden_states, 2 for obs """

        if data_type == 0:
            assert shape is not None
            self.storage[name] = DataBuf(self.n_trans_per_env, self.num_envs, shape, device=self.device)
        elif data_type == 1:
            assert shape is not None
            self.storage[name] = HiddenBuf(self.n_trans_per_env, self.num_envs, shape, device=self.device)
        elif data_type == 2:
            assert cfg is not None
            self.storage[name] = ObsTransBuf(cfg, self.n_trans_per_env, self.num_envs, device=self.device)
        else:
            raise ValueError(f'data_type {data_type} is not supported')

    def compose_storage(self):
        storage_numel = {k: s.numel() for k, s in self.storage.items()}
        transition_size = sum(storage_numel.values())

        self._storage_tensor = torch.empty(self.n_trans_per_env, self.num_envs, transition_size,
                                           dtype=torch.float32,
                                           device=self.device)

        cur_idx = 0
        for k, s in self.storage.items():
            s.init_buf(self._storage_tensor[:, :, cur_idx:cur_idx + storage_numel[k]])
            cur_idx += storage_numel[k]

    def add_transitions(self, transition):
        if self.step >= self.n_trans_per_env:
            raise AssertionError("Rollout buffer overflow")

        if not self.init_done:
            with torch.inference_mode(mode=False):
                self._init_storage(transition)

        for n, v in transition.get_items():
            if n in self.storage:
                self.storage[n].set(self.step, v)

        self.step += 1

    def _init_storage(self, transition):
        for n, v in transition.get_items():
            if type(v) is bool:
                # bool data
                self.storage[n] = DataBuf(self.num_envs, self.n_trans_per_env, (1,), torch.bool, self.device)
            elif isinstance(v, self.obs_base_cls):
                # observations
                self.storage[n] = ObsTransBuf(self.num_envs, self.n_trans_per_env, self.device)
                self.storage[n].init_buffer(v)

            elif n.endswith('hidden_states'):
                self.storage[n] = HiddenBuf(self.num_envs, self.n_trans_per_env, v.shape, v.dtype, self.device)

            elif isinstance(v, torch.Tensor):
                # other data
                self.storage[n] = DataBuf(self.num_envs, self.n_trans_per_env, v.shape[1:], v.dtype, self.device)
            else:
                raise NotImplementedError(f'Data of type {type(v)} is not implemented yet. Data name {n}')

        self.init_done = True

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values_default, last_values_contact, gamma, lam):
        advantage_default = advantage_contact = 0
        dones_buf = self.storage['dones'].buf.float()
        rewards_default_buf = self.storage['rewards_default'].buf
        rewards_contact_buf = self.storage['rewards_contact'].buf

        values_default_buf = self.storage['values_default'].buf
        values_contact_buf = self.storage['values_contact'].buf
        w_default, w_contact = 1., 0.25

        for step in reversed(range(self.n_trans_per_env)):
            if step == self.n_trans_per_env - 1:
                next_values_default = last_values_default
                next_values_contact = last_values_contact
            else:
                next_values_default = self.storage['values_default'].get(step + 1)
                next_values_contact = self.storage['values_contact'].get(step + 1)

            next_is_not_terminal = 1.0 - dones_buf[step]
            delta_default = rewards_default_buf[step] + next_is_not_terminal * gamma * next_values_default - values_default_buf[step]
            advantage_default = delta_default + next_is_not_terminal * gamma * lam * advantage_default
            self.returns_default[step] = advantage_default + values_default_buf[step]

            delta_contact = rewards_contact_buf[step] + next_is_not_terminal * gamma * next_values_contact - values_contact_buf[step]
            advantage_contact = delta_contact + next_is_not_terminal * gamma * lam * advantage_contact
            self.returns_contact[step] = advantage_contact + values_contact_buf[step]

        # Compute and normalize the advantages
        advantages_default = self.returns_default - values_default_buf
        advantages_default[:] = (advantages_default - advantages_default.mean()) / (advantages_default.std() + 1e-8)

        advantages_contact = self.returns_contact - values_contact_buf
        advantages_contact[:] = (advantages_contact - advantages_contact.mean()) / (advantages_contact.std() + 1e-8)

        self.advantages[:] = w_default * advantages_default + w_contact * advantages_contact

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.step < self.n_trans_per_env - 1:
            raise AssertionError('why the buffer is not full?')

        batch_size = self.num_envs * self.n_trans_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(batch_size)

        returns_default = self.returns_default.flatten(0, 1)
        returns_contact = self.returns_contact.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                batch_dict = {n: v.flatten_get(batch_idx) for n, v in self.storage.items()}
                batch_dict['returns_default'] = returns_default[batch_idx]
                batch_dict['returns_contact'] = returns_contact[batch_idx]
                batch_dict['advantages'] = advantages[batch_idx]

                yield batch_dict

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        valid_mask = ~(torch.cumsum(self.storage['dones'].buf, dim=0) > 0)

        mini_batch_size = self.num_envs // num_mini_batches
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                batch_dict = {'masks': valid_mask[:, start: stop]}

                for n, v in self.storage.items():
                    batch_dict[n] = v.get((slice(None), slice(start, stop)))

                batch_dict['returns_default'] = self.returns_default[:, start: stop]
                batch_dict['returns_contact'] = self.returns_contact[:, start: stop]
                batch_dict['advantages'] = self.advantages[:, start: stop]

                yield batch_dict
