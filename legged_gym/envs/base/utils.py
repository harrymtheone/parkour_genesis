import torch


class ObsBase:
    def get_items(self):
        return self.__dict__.items()

    def slice(self, slice_):
        sliced_v = []
        for v in self.__dict__.values():
            if v is not None:
                sliced_v.append(v[slice_])
        return type(self)(*sliced_v)

    def flatten(self, start, stop):
        sliced_v = []
        for v in self.__dict__.values():
            if v is not None:
                sliced_v.append(v.flatten(start, stop))
        return type(self)(*sliced_v)

    def unflatten(self, dim, shape):
        sliced_v = []
        for v in self.__dict__.values():
            if v is not None:
                sliced_v.append(v.unflatten(dim, shape))
        return type(self)(*sliced_v)

    def transpose(self, dim0, dim1):
        for n, v in self.__dict__.items():
            setattr(self, n, v.transpose(dim0, dim1))
        return self

    def to(self, device):
        for n, v in self.__dict__.items():
            setattr(self, n, v.to(device))
        return self

    def to_tensor(self):
        raise NotImplementedError

    def to_1D_tensor(self):
        return type(self)(*[v.flatten(1) for v in self.__dict__.values()])

    def clip(self, thresh):
        for v in self.__dict__.values():
            if v is not None:
                torch.clip(v, -thresh, thresh, out=v)

    def clone(self):
        return type(self)(*self.__dict__.values())

    def float(self):
        args = []
        for v in self.__dict__.values():
            if v is not None:
                args.append(v.float())
        return type(self)(*args)


class HistoryBuffer:
    def __init__(self, n_envs, his_len, *data_shape, dtype=None, device=None):
        if device is None:
            raise ValueError('please define the buffer device!')

        self.buf_len = his_len
        dtype = torch.float32 if dtype is None else dtype
        self.buf = torch.zeros(n_envs, self.buf_len, *data_shape, dtype=dtype, device=device)

    def append(self, data, reset):
        for _ in range(data.ndim):
            reset = reset.unsqueeze(-1)

        self.buf[:] = torch.where(
            reset,
            torch.stack([data] * self.buf_len, dim=1),
            torch.cat([self.buf[:, 1:], data.unsqueeze(1)], dim=1),
        )

        # self.buf[reset].zero_()
        # self.buf[:] = torch.cat([self.buf[:, 1:], data.unsqueeze(1)], dim=1)

    def get(self):
        return self.buf.clone()


class DelayBuffer:
    def __init__(self, n_envs, data_shape, delay_range, rand_per_step, dtype=None, device=None):
        self.n_envs = n_envs
        self.delay_range = delay_range
        self.rand_per_step = rand_per_step
        self.buf_len = 1 + 1 + delay_range[1]

        dtype = torch.float32 if dtype is None else dtype
        assert device is not None
        self.buf = torch.zeros((n_envs, self.buf_len, *data_shape), dtype=dtype, device=device)

        # delay should +1 because you should call step before get
        self.delay = 1 + torch.randint(delay_range[0], delay_range[1] + 1, (self.n_envs, 1), device=device)
        self._cols = torch.arange(self.buf_len, device=device).unsqueeze(0)
        self._mask = self._cols >= self.delay

    def update_delay_range(self, delay_range):
        self.delay_range = delay_range

        n_envs = self.buf.size(0)
        old_buf_len = self.buf_len
        data_shape = self.buf.shape[2:]

        self.buf_len = 1 + 1 + delay_range[1]
        assert self.buf_len >= old_buf_len
        old_buf = self.buf
        self.buf = torch.zeros((n_envs, self.buf_len, *data_shape), dtype=old_buf.dtype, device=old_buf.device)

        self.buf[:, :old_buf_len] = old_buf

        # delay should +1 because you should call step before get
        self.delay = 1 + torch.randint(delay_range[0], delay_range[1] + 1, (self.n_envs, 1), device=old_buf.device)
        self._cols = torch.arange(self.buf_len, device=old_buf.device).unsqueeze(0)
        self._mask = self._cols >= self.delay

    def append(self, data: torch.Tensor):
        if self.rand_per_step:
            self.delay[:] = 1 + torch.randint_like(self.delay, self.delay_range[0], self.delay_range[1] + 1)
            self._mask[:] = self._cols >= self.delay

        self.buf[self._mask] = data.unsqueeze(1).expand_as(self.buf)[self._mask]

    def step(self):
        self.buf[:, :-1] = self.buf[:, 1:].clone()

    def get(self):
        return self.buf[:, 0].clone()

    def reset(self, dones):
        self.buf[dones].zero_()


class LagBuffer:
    def __init__(self, data_shape, delay_range, rand_per_step=False, dtype=None, device=None):
        dtype = torch.float32 if dtype is None else dtype
        assert device is not None
        n_envs, data_shape = data_shape[0], data_shape[1:]
        self.delay_range = delay_range
        self.buf = torch.zeros(n_envs, delay_range[1], *data_shape, dtype=dtype, device=device)

        self.rand_per_step = rand_per_step
        self.env_idx = torch.arange(n_envs, device=device)
        self.delay_idx = torch.randint(delay_range[0], delay_range[1] + 1, (n_envs,))

    def append(self, data):
        self.buf[:, :-1] = self.buf[:, 1:]
        self.buf[:, -1] = data

    def get(self):
        if self.rand_per_step:
            self.delay_idx[:] = torch.randint_like(self.delay_idx, self.delay_range[0], self.delay_range[1] + 1)

        return self.buf[self.env_idx, self.delay_idx].clone()

    def reset(self, dones):
        self.buf[dones].zero_()
