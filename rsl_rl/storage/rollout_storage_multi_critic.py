import torch


class DataBuf:
    def __init__(self, n_envs, n_trans_per_env, shape, dtype, device):
        self.n_envs = n_envs
        self.n_trans_per_env = n_trans_per_env
        self.device = device

        self.buf = torch.zeros(n_trans_per_env, n_envs, *shape, dtype=dtype, device=self.device)

    def set(self, idx, value):
        self.buf[idx] = value

    def get(self, slice_):
        return self.buf[slice_]

    def flatten_get(self, idx):
        return self.buf.flatten(0, 1)[idx]


class HiddenBuf:
    def __init__(self, n_envs, n_trans_per_env, shape, dtype, device):
        self.n_envs = n_envs
        self.n_trans_per_env = n_trans_per_env
        n_layers, _, hidden_size = shape
        self.device = device

        self.buf = torch.zeros(n_trans_per_env, n_layers, n_envs, hidden_size, dtype=dtype, device=self.device)

    def set(self, idx, value):
        self.buf[idx] = value

    def get(self, slice_):
        return self.buf[0][slice_].contiguous()


class ObsTransBuf:
    def __init__(self, n_envs, n_trans_per_env, device):
        self.n_envs = n_envs
        self.n_trans_per_env = n_trans_per_env
        self.device = device

        self.obs_class = None
        self.storage = {}

        self.traj_split_padded = None

    def init_buffer(self, obs):
        self.obs_class = type(obs)

        for n, v in obs.items():
            if n in self.storage or v is None:
                continue
            self.storage[n] = torch.zeros(self.n_trans_per_env, self.n_envs, *v.shape[1:], dtype=v.dtype, device=self.device)

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


class RolloutStorageMultiCritic:
    def __init__(self, num_envs, n_trans_per_env, device):
        self.device = device
        self.num_transitions_per_env = n_trans_per_env
        self.num_envs = num_envs

        self.storage = {}
        self.returns_default = torch.zeros(n_trans_per_env, num_envs, 1, device=self.device)
        self.returns_contact = torch.zeros(n_trans_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(n_trans_per_env, num_envs, 1, device=self.device)

        from legged_gym.envs.base.utils import ObsBase
        self.obs_base_cls = ObsBase

        self.init_done = False
        self.step = 0

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
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
                self.storage[n] = DataBuf(self.num_envs, self.num_transitions_per_env, (1,), torch.bool, self.device)
            elif isinstance(v, self.obs_base_cls):
                # observations
                self.storage[n] = ObsTransBuf(self.num_envs, self.num_transitions_per_env, self.device)
                self.storage[n].init_buffer(v)

            elif n.endswith('hidden_states'):
                self.storage[n] = HiddenBuf(self.num_envs, self.num_transitions_per_env, v.shape, v.dtype, self.device)

            elif type(v) is torch.Tensor:
                # other data
                self.storage[n] = DataBuf(self.num_envs, self.num_transitions_per_env, v.shape[1:], v.dtype, self.device)
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

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
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
        if self.step < self.num_transitions_per_env - 1:
            raise AssertionError('why the buffer is not full?')

        batch_size = self.num_envs * self.num_transitions_per_env
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
