import torch
from torch import nn


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def make_linear_layers(*shape, activation_func=None, output_activation=True):
    if activation_func is None:
        raise ValueError('activation_func cannot be None!')

    layers = nn.Sequential()

    for l1, l2 in zip(shape[:-1], shape[1:]):
        layers.append(nn.Linear(l1, l2))
        layers.append(activation_func)

    if not output_activation:
        layers.pop(-1)

    return layers


@torch.compiler.disable  # Prevent compilation
def gru_wrapper(func, *args, **kwargs):
    n_steps = args[0].size(0)
    rtn = func(*[arg.flatten(0, 1) for arg in args], **kwargs)

    if type(rtn) is tuple:
        return [r.unflatten(0, (n_steps, -1)) for r in rtn]
    else:
        return rtn.unflatten(0, (n_steps, -1))


def recurrent_wrapper(func, tensor):
    n_seq = tensor.size(0)
    return func(tensor.flatten(0, 1)).unflatten(0, (n_seq, -1))


class UniMixOneHotCategorical(torch.distributions.OneHotCategorical):
    def __init__(self, logits, unimix_ratio=0.0):
        assert unimix_ratio > 0.

        probs = torch.softmax(logits, dim=-1)
        probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
        logits = torch.log(probs)

        super().__init__(probs=None, logits=logits)

    @property
    def mode(self):
        _mode = super().mode
        return _mode.detach() + self.logits - self.logits.detach()

    def sample(self, sample_shape=torch.Size()):
        sample = super().sample(sample_shape)
        probs = super().probs

        assert sample.shape == probs.shape
        return sample.detach() + probs - probs.detach()


