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


class UniversalCritic(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super().__init__()
        activation = nn.ELU()

        self.priv_enc = nn.Sequential(
            nn.Conv1d(in_channels=env_cfg.num_critic_obs, out_channels=64, kernel_size=6, stride=4),
            activation,
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=2),
            activation,
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1),
            activation,
            nn.Flatten()
        )
        self.scan_enc = make_linear_layers(env_cfg.n_scan, 256, 64,
                                           activation_func=activation)
        self.edge_mask_enc = make_linear_layers(env_cfg.n_scan, 256, 64,
                                                activation_func=activation)
        self.critic = make_linear_layers(128 + 64 + 64, *policy_cfg.critic_hidden_dims, 1,
                                         activation_func=nn.ELU(),
                                         output_activation=False)

    def forward(self, priv_his, scan, edge_mask):
        priv_latent = gru_wrapper(self.priv_enc, priv_his.transpose(2, 3))
        scan_enc = self.scan_enc(scan.flatten(2))
        edge_enc = self.edge_mask_enc(edge_mask.flatten(2))
        return self.critic(torch.cat([priv_latent, scan_enc, edge_enc], dim=2))

    def evaluate(self, obs):
        if obs.priv_his.ndim == 3:
            priv_latent = self.priv_enc(obs.priv_his.transpose(1, 2))
            scan_enc = self.scan_enc(obs.scan.flatten(1))
            edge_enc = self.edge_mask_enc(obs.edge_mask.flatten(1))
            return self.critic(torch.cat([priv_latent, scan_enc, edge_enc], dim=1))
        else:
            priv_latent = gru_wrapper(self.priv_enc, obs.priv_his.transpose(2, 3))
            scan_enc = gru_wrapper(self.scan_enc, obs.scan.flatten(2))
            edge_enc = gru_wrapper(self.edge_mask_enc, obs.edge_mask.flatten(2))
            return gru_wrapper(self.critic, torch.cat([priv_latent, scan_enc, edge_enc], dim=2))


class MixtureOfCritic(nn.Module):
    def __init__(self, task_cfg):
        super().__init__()
        activation = nn.ELU()

        self.priv_enc = nn.Sequential(
            nn.Conv1d(in_channels=task_cfg.env.num_critic_obs, out_channels=64, kernel_size=6, stride=4),
            activation,
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=2),
            activation,
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1),
            activation,
            nn.Flatten()
        )
        self.scan_enc = make_linear_layers(task_cfg.env.n_scan, 256, 64,
                                           activation_func=activation)
        self.edge_mask_enc = make_linear_layers(task_cfg.env.n_scan, 256, 64,
                                                activation_func=activation)

        self.critic = nn.ModuleDict()

        for rew_name in dir(task_cfg.rewards.scales):
            if rew_name.startswith('__'):
                continue

            self.critic[rew_name] = make_linear_layers(128 + 64 + 64, *task_cfg.policy.critic_hidden_dims, 1,
                                                       activation_func=nn.ELU(),
                                                       output_activation=False)

    def evaluate(self, obs):
        if obs.priv_his.ndim == 3:
            priv_latent = self.priv_enc(obs.priv_his.transpose(1, 2))
            scan_enc = self.scan_enc(obs.scan.flatten(1))
            edge_enc = self.edge_mask_enc(obs.edge_mask.flatten(1))
            return {rew_name: critic(torch.cat([priv_latent, scan_enc, edge_enc], dim=1)) for rew_name, critic in self.critic.items()}
        else:
            priv_latent = gru_wrapper(self.priv_enc, obs.priv_his.transpose(2, 3))
            scan_enc = gru_wrapper(self.scan_enc, obs.scan.flatten(2))
            edge_enc = gru_wrapper(self.edge_mask_enc, obs.edge_mask.flatten(2))
            return {rew_name: gru_wrapper(critic, torch.cat([priv_latent, scan_enc, edge_enc], dim=2)) for rew_name, critic in self.critic.items()}

    def load(self, state_dict: dict):
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            # Handle ModuleDict mismatch by loading only matching keys
            model_state_dict = self.state_dict()
            filtered_state_dict = {}

            for key, value in state_dict.items():
                if key not in model_state_dict:
                    continue

                if model_state_dict[key].shape != value.shape:
                    raise ValueError(f"Shape mismatch for key '{key}': "
                                     f"model expects {model_state_dict[key].shape}, "
                                     f"but state_dict has {value.shape}")
                filtered_state_dict[key] = value

            # Load only the matching keys, ignore missing and unexpected keys
            self.load_state_dict(filtered_state_dict, strict=False)
