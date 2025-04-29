from __future__ import annotations

from typing import Sequence

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
    """Circular buffer for storing a history of batched tensor data.

    This class implements a circular buffer for storing a history of batched tensor data. The buffer is
    initialized with a maximum length and a batch size. The data is stored in a circular fashion, and the
    data can be retrieved in a LIFO (Last-In-First-Out) fashion. The buffer is designed to be used in
    multi-environment settings, where each environment has its own data.

    The shape of the appended data is expected to be (batch_size, ...), where the first dimension is the
    batch dimension. Correspondingly, the shape of the ring buffer is (max_len, batch_size, ...).
    """

    def __init__(self, max_len: int, batch_size: int, device: str):
        """Initialize the circular buffer.

        Args:
            max_len: The maximum length of the circular buffer. The minimum allowed value is 1.
            batch_size: The batch dimension of the data.
            device: The device used for processing.

        Raises:
            ValueError: If the buffer size is less than one.
        """
        if max_len < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")
        # set the parameters
        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)

        # max length tensor for comparisons
        self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
        # number of data pushes passed since the last call to :meth:`reset`
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        # the pointer to the current head of the circular buffer (-1 means not initialized)
        self._pointer: int = -1
        # the actual buffer for data storage
        # note: this is initialized on the first call to :meth:`append`
        self._buffer: torch.Tensor = None  # type: ignore

    """
    Properties.
    """

    @property
    def batch_size(self) -> int:
        """The batch size of the ring buffer."""
        return self._batch_size

    @property
    def device(self) -> str:
        """The device used for processing."""
        return self._device

    @property
    def max_length(self) -> int:
        """The maximum length of the ring buffer."""
        return int(self._max_len[0].item())

    @property
    def current_length(self) -> torch.Tensor:
        """The current length of the buffer. Shape is (batch_size,).

        Since the buffer is circular, the current length is the minimum of the number of pushes
        and the maximum length.
        """
        return torch.minimum(self._num_pushes, self._max_len)

    @property
    def buffer(self) -> torch.Tensor:
        """Complete circular buffer with most recent entry at the end and oldest entry at the beginning.
        Returns:
            Complete circular buffer with most recent entry at the end and oldest entry at the beginning of dimension 1. The shape is [batch_size, max_length, data.shape[1:]].
        """
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
        return torch.transpose(buf, dim0=0, dim1=1)

    """
    Operations.
    """

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the circular buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        # resolve all indices
        if batch_ids is None:
            batch_ids = slice(None)
        # reset the number of pushes for the specified batch indices
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            # set buffer at batch_id reset indices to 0.0 so that the buffer() getter returns the cleared circular buffer after reset.
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor):
        """Append the data to the circular buffer.

        Args:
            data: The data to append to the circular buffer. The first dimension should be the batch dimension.
                Shape is (batch_size, ...).

        Raises:
            ValueError: If the input data has a different batch size than the buffer.
        """
        # check the batch size
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has {data.shape[0]} environments while expecting {self.batch_size}")

        # at the first call, initialize the buffer size
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype, device=self._device)
        # move the head to the next slot
        self._pointer = (self._pointer + 1) % self.max_length
        # add the new data to the last layer
        self._buffer[self._pointer] = data.to(self._device)
        # Check for batches with zero pushes and initialize all values in batch to first append
        if 0 in self._num_pushes.tolist():
            fill_ids = [i for i, x in enumerate(self._num_pushes.tolist()) if x == 0]
            self._num_pushes.tolist().index(0) if 0 in self._num_pushes.tolist() else None
            self._buffer[:, fill_ids, :] = data.to(self._device)[fill_ids]
        # increment number of number of pushes for all batches
        self._num_pushes += 1

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        """Retrieve the data from the circular buffer in last-in-first-out (LIFO) fashion.

        If the requested index is larger than the number of pushes since the last call to :meth:`reset`,
        the oldest stored data is returned.

        Args:
            key: The index to retrieve from the circular buffer. The index should be less than the number of pushes
                since the last call to :meth:`reset`. Shape is (batch_size,).

        Returns:
            The data from the circular buffer. Shape is (batch_size, ...).

        Raises:
            ValueError: If the input key has a different batch size than the buffer.
            RuntimeError: If the buffer is empty.
        """
        # check the batch size
        if len(key) != self.batch_size:
            raise ValueError(f"The argument 'key' has length {key.shape[0]}, while expecting {self.batch_size}")
        # check if the buffer is empty
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        # admissible lag
        valid_keys = torch.minimum(key, self._num_pushes - 1)
        # the index in the circular buffer (pointer points to the last+1 index)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_length)
        # return output
        return self._buffer[index_in_buffer, self._ALL_INDICES]


class DelayBufferNew:
    """Delay buffer that allows retrieving stored data with delays.

    This class uses a batched circular buffer to store input data. Different to a standard circular buffer,
    which uses the LIFO (last-in-first-out) principle to retrieve the data, the delay buffer class allows
    retrieving data based on the lag set by the user. For instance, if the delay set inside the buffer
    is 1, then the second last entry from the stream is retrieved. If it is 2, then the third last entry
    and so on.

    The class supports storing a batched tensor data. This means that the shape of the appended data
    is expected to be (batch_size, ...), where the first dimension is the batch dimension. Correspondingly,
    the delay can be set separately for each batch index. If the requested delay is larger than the current
    length of the underlying buffer, the most recent entry is returned.

    .. note::
        By default, the delay buffer has no delay, meaning that the data is returned as is.
    """

    def __init__(self, history_length: int, batch_size: int, device: str):
        """Initialize the delay buffer.

        Args:
            history_length: The history of the buffer, i.e., the number of time steps in the past that the data
                will be buffered. It is recommended to set this value equal to the maximum time-step lag that
                is expected. The minimum acceptable value is zero, which means only the latest data is stored.
            batch_size: The batch dimension of the data.
            device: The device used for processing.
        """
        # set the parameters
        self._history_length = max(0, history_length)

        # the buffer size: current data plus the history length
        self._circular_buffer = CircularBuffer(self._history_length + 1, batch_size, device)

        # the minimum and maximum lags across all environments.
        self._min_time_lag = 0
        self._max_time_lag = 0
        # the lags for each environment.
        self._time_lags = torch.zeros(batch_size, dtype=torch.int, device=device)

    """
    Operations.
    """

    def set_time_lag(self, time_lag: int | torch.Tensor, batch_ids: Sequence[int] | None = None):
        """Sets the time lag for the delay buffer across the provided batch indices.

        Args:
            time_lag: The desired delay for the buffer.

              * If an integer is provided, the same delay is set for the provided batch indices.
              * If a tensor is provided, the delay is set for each batch index separately. The shape of the tensor
                should be (len(batch_ids),).

            batch_ids: The batch indices for which the time lag is set. Default is None, which sets the time lag
                for all batch indices.

        Raises:
            TypeError: If the type of the :attr:`time_lag` is not int or integer tensor.
            ValueError: If the minimum time lag is negative or the maximum time lag is larger than the history length.
        """
        # resolve batch indices
        if batch_ids is None:
            batch_ids = slice(None)

        # parse requested time_lag
        if isinstance(time_lag, int):
            # set the time lags across provided batch indices
            self._time_lags[batch_ids] = time_lag
        elif isinstance(time_lag, torch.Tensor):
            # check valid dtype for time_lag: must be int or long
            if time_lag.dtype not in [torch.int, torch.long]:
                raise TypeError(f"Invalid dtype for time_lag: {time_lag.dtype}. Expected torch.int or torch.long.")
            # set the time lags
            self._time_lags[batch_ids] = time_lag.to(device=self.device)
        else:
            raise TypeError(f"Invalid type for time_lag: {type(time_lag)}. Expected int or integer tensor.")

        # compute the min and max time lag
        self._min_time_lag = int(torch.min(self._time_lags).item())
        self._max_time_lag = int(torch.max(self._time_lags).item())
        # check that time_lag is feasible
        if self._min_time_lag < 0:
            raise ValueError(f"The minimum time lag cannot be negative. Received: {self._min_time_lag}")
        if self._max_time_lag > self._history_length:
            raise ValueError(
                f"The maximum time lag cannot be larger than the history length. Received: {self._max_time_lag}"
            )

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the data in the delay buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        self._circular_buffer.reset(batch_ids)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Append the input data to the buffer and returns a stale version of the data based on time lag delay.

        If the requested delay is larger than the number of buffered data points since the last reset,
        the function returns the latest data. For instance, if the delay is set to 2 and only one data point
        is stored in the buffer, the function will return the latest data. If the delay is set to 2 and three
        data points are stored, the function will return the first data point.

        Args:
           data: The input data. Shape is (batch_size, ...).

        Returns:
            The delayed version of the data from the stored buffer. Shape is (batch_size, ...).
        """
        # add the new data to the last layer
        self._circular_buffer.append(data)
        # return output
        delayed_data = self._circular_buffer[self._time_lags]
        return delayed_data.clone()
