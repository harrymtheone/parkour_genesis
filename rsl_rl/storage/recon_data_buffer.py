import torch


class ReconDataBuf:
    def __init__(self, rows, cols, buf_len, batch_size, device):
        self.prop_buf = torch.zeros(rows, cols, buf_len, 10, 45, device=device)
        self.depth_buf = torch.zeros(rows, cols, buf_len, 2, 58, 87, device=device)
        self.recon_prev_buf = torch.zeros(rows, cols, buf_len, 32, 16, device=device)
        self.scan_buf = torch.zeros(rows, cols, buf_len, 32, 16, device=device)

        self.buf_len = buf_len
        self.batch_size = batch_size
        self.device = device

        self.prev_data_loss = torch.zeros(buf_len, device=device)
        self.cur_idx = 0

    @torch.compile(mode='default')
    def append_and_sample(self, proprio, depth, recon_prev, scan):
        """
        1. select where to fill in (minimum self.prev_data_loss)
        2. fill in the data, set corresponding self.prev_data_loss to zero
        3. select which data to sample (maximum self.prev_data_loss)
        4. slice to get those data, concatenate them with the data
            just appended (new data should be used at least once)
        5. return the data and the index of the data (both new data and sampled data)
        """

        len_data = len(proprio)
        _, idx_sorted_by_loss = torch.sort(self.prev_data_loss)

        # select where to fill in
        if self.cur_idx + len_data < self.buf_len:
            data_idx = torch.arange(self.cur_idx, self.cur_idx + len_data, device=self.device)
            self.cur_idx += len_data
        elif self.cur_idx < self.buf_len:
            data_idx = torch.arange(self.cur_idx, self.buf_len, device=self.device)
            self.cur_idx = self.buf_len
        else:
            data_idx = idx_sorted_by_loss[-len_data:]

        # append data to the buffer
        self.prop_buf[data_idx] = proprio
        self.depth_buf[data_idx] = depth
        self.recon_prev_buf[data_idx] = recon_prev
        self.scan_buf[data_idx] = scan
        self.prev_data_loss[data_idx] = 0

        # sample data from the buffer
        sampled_idx = idx_sorted_by_loss[:self.batch_size]

        return (torch.cat([data_idx, sampled_idx], dim=0),
                torch.cat([proprio, self.prop_buf[sampled_idx]], dim=0),
                torch.cat([depth, self.depth_buf[sampled_idx]], dim=0),
                torch.cat([recon_prev, self.recon_prev_buf[sampled_idx]], dim=0),
                torch.cat([scan, self.scan_buf[sampled_idx]], dim=0))

    @torch.compile(mode='default')
    def update_loss(self, idx, loss):
        """ After loss calculation, use the index to update the self.prev_data_loss """
        self.prev_data_loss[idx] = loss
