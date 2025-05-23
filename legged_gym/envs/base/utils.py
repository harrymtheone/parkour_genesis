from __future__ import annotations

from typing import Sequence, Tuple

import torch


class ObsBase:
    @torch.compiler.disable
    def items(self):
        return self.__dict__.items()

    @torch.compiler.disable
    def __getitem__(self, item):
        sliced_v = []
        for v in self.__dict__.values():
            if v is not None:
                sliced_v.append(v[item])
        return type(self)(*sliced_v)

    @torch.compiler.disable
    def clip(self, thresh):
        for v in self.__dict__.values():
            if v is not None:
                torch.clip(v, -thresh, thresh, out=v)

    @torch.compiler.disable
    def flatten(self, start, stop):
        sliced_v = []
        for v in self.__dict__.values():
            if v is not None:
                sliced_v.append(v.flatten(start, stop))
        return type(self)(*sliced_v)

    @torch.compiler.disable
    def unflatten(self, dim, shape):
        sliced_v = []
        for v in self.__dict__.values():
            if v is not None:
                sliced_v.append(v.unflatten(dim, shape))
        return type(self)(*sliced_v)

    # def transpose(self, dim0, dim1):
    #     for n, v in self.__dict__.items():
    #         setattr(self, n, v.transpose(dim0, dim1))
    #     return self
    #
    # def to(self, device):
    #     for n, v in self.__dict__.items():
    #         setattr(self, n, v.to(device))
    #     return self
    #
    # def to_tensor(self):
    #     raise NotImplementedError
    #
    # def to_1D_tensor(self):
    #     return type(self)(*[v.flatten(1) for v in self.__dict__.values()])
    #
    #
    @torch.compiler.disable
    def clone(self):
        return type(self)(*self.__dict__.values())
    #
    # def float(self):
    #     args = []
    #     for v in self.__dict__.values():
    #         if v is not None:
    #             args.append(v.float())
    #     return type(self)(*args)


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

        # self.buf[reset] = 0.
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
        self.buf[dones] = 0.


class CircularBuffer:
    def __init__(self, max_len: int, batch_size: int, data_shape: Sequence[int], device: torch.device):
        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)

        self._max_len = max_len
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._pointer: int = -1
        self._buffer = torch.empty((self._max_len, batch_size, *data_shape), dtype=torch.float, device=self._device)

    def reset(self, batch_ids: Sequence[int] | None = None):
        if batch_ids is None:
            batch_ids = slice(None)

        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor):
        assert data.size(0) == self._batch_size

        self._pointer = (self._pointer + 1) % self._max_len
        self._buffer[self._pointer] = data

        # Check for batches with zero pushes and initialize all values in batch to first append
        fill_ids = torch.where(self._num_pushes < 1)[0]
        if fill_ids.numel() > 0:
            self._buffer[:, fill_ids] = data[fill_ids]

        self._num_pushes[:] += 1

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        # check if the buffer is empty
        # if torch.any(self._num_pushes < 1) or self._buffer is None:
        #     raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        valid_keys = torch.minimum(key, self._num_pushes - 1)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self._max_len)
        return self._buffer[index_in_buffer, self._ALL_INDICES]

    def get_all(self):
        valid_keys = torch.arange(self._max_len, device=self._device)
        # index_in_buffer = torch.remainder(valid_keys - self._pointer, self._max_len)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self._max_len)
        return self._buffer[index_in_buffer]


class DelayBufferCircular:
    def __init__(self, history_length: int, batch_size: int, data_shape: Sequence[int], device: torch.device):
        self._history_length = max(0, history_length)
        self._batch_size = batch_size
        self._device = device

        self._circular_buffer = CircularBuffer(self._history_length + 1, batch_size, data_shape, device)
        self._time_lags = torch.zeros(batch_size, dtype=torch.int, device=device)

    def set_delay_range(self, delay_prop: Tuple[int, int]):
        self._time_lags[:] = torch.randint(
            low=delay_prop[0],
            high=delay_prop[1] + 1,
            size=(self._batch_size,),
            dtype=torch.int,
            device=self._device,
        )

    def reset(self, batch_ids: Sequence[int] | None = None):
        self._circular_buffer.reset(batch_ids)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        self._circular_buffer.append(data)
        return self._circular_buffer[self._time_lags].clone()

    def get(self):
        return self._circular_buffer[self._time_lags].clone()


class DelayBufferHumanoidGym:
    def __init__(self, history_length: int, batch_size: int, data_shape: Sequence[int], device: torch.device):
        self._batch_size = batch_size
        self._device = device
        self._all_envs = torch.arange(batch_size, device=device)

        self._lag_buffer = torch.zeros(history_length, batch_size, *data_shape, dtype=torch.float, device=device)
        self.lag_timestep = torch.zeros(batch_size, dtype=torch.int, device=device)

    def set_delay_range(self, delay_prop: Tuple[int, int]):
        self.lag_timestep[:] = torch.randint(
            low=delay_prop[0],
            high=delay_prop[1] + 1,
            size=(self._batch_size,),
            dtype=torch.int,
            device=self._device,
        )

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        self._lag_buffer[1:] = self._lag_buffer[:-1].clone()
        self._lag_buffer[0] = data

        return self._lag_buffer[self.lag_timestep, self._all_envs]

    def reset(self, batch_ids: Sequence[int] | None = None):
        self._lag_buffer[:, batch_ids] = 0.

    def get(self):
        return self._lag_buffer[self.lag_timestep, self._all_envs]
