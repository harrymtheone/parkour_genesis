import torch


class RunningMeanStd:
    def __init__(self, shape, device):
        self.n = 0
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.zeros(shape, device=device)

    def update(self, data: torch.Tensor):
        batch_count = data.size(0)
        batch_mean = data.mean()
        batch_var = data.var()

        delta = batch_mean - self.mean
        tot_count = self.n + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.n
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.square() * self.n * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.n = tot_count


class RewardScaling:
    def __init__(self, n_envs, device):
        self.ret = torch.zeros(n_envs, 1, device=device)
        self.ret_rms = RunningMeanStd((), device=device)

    def update(self, rew, gamma):
        self.ret = self.ret * gamma + rew
        self.ret_rms.update(self.ret)

        return rew / (self.ret_rms.var + 1e-8).sqrt()
