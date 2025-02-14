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


def make_linear_layers(*shape, activation_func=None):
    if activation_func is None:
        raise ValueError('activation_func cannot be None!')

    layers = nn.Sequential()

    for l1, l2 in zip(shape[:-1], shape[1:]):
        layers.append(nn.Linear(l1, l2))
        layers.append(activation_func)

    return layers


def unpad_trajectories(traj, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    if torch.all(masks):
        return traj
    return traj.transpose(1, 0)[masks.transpose(1, 0)].view(-1, len(traj), *traj.shape[2:]).transpose(1, 0)
