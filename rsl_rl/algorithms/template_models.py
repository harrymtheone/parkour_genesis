from enum import Enum

import torch
from torch import autograd, nn

from rsl_rl.modules.utils import get_activation, make_linear_layers, gru_wrapper


class AMPType(Enum):
    least_square = 0
    wasserstein = 1
    log = 2
    bce = 3


class AMPDiscriminator(nn.Module):
    def __init__(self,
                 num_input=400,
                 hidden_dims=(1024, 512, 256),
                 activation='elu',
                 amp_reward_coef=1.0,
                 amp_type='least_square',
                 lambda_schedule_dict=None,
                 task_rew_schedule_dict=None,
                 device='cuda:0',
                 **kwargs):

        if kwargs:
            print("Dagger.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(AMPDiscriminator, self).__init__()
        self.device = device
        if amp_type not in AMPType.__members__:
            raise ValueError(f"Invalid AMP type: {amp_type}. Must be one of {list(AMPType.__members__.keys())}.")

        activation = get_activation(activation)

        # mlp
        amp_layers = []
        self.num_input = num_input
        curr_in_dim = num_input
        for hidden_dim in hidden_dims:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(activation)
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(self.device)
        self.amp_linear = nn.Linear(hidden_dims[-1], 1).to(self.device)

        self.trunk.train()
        self.amp_linear.train()

        print(f"Amp: {self.trunk}")

        self.amp_reward_coef = amp_reward_coef
        self.amp_type = amp_type

        self._build_lambda_schedule(lambda_schedule_dict)
        self._build_task_rew_schedule(task_rew_schedule_dict)

    def forward(self, x):
        x = x.float()
        d_trunk = self.trunk(x.to(self.device))
        return self.amp_linear(d_trunk)

    def compute_style_rew(self, generated_motion, reset_amp_obs, reset_idx, task_rew):
        with torch.no_grad():
            self.eval()
            generated_motion = generated_motion.clone()
            if reset_idx is not None:
                generated_motion[reset_idx] = reset_amp_obs

            if self.amp_type == AMPType.least_square.name:
                d = self.forward(generated_motion)
                amp_rew = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            elif self.amp_type == AMPType.wasserstein.name:
                d = self.forward(generated_motion)
                amp_rew = self.amp_reward_coef * torch.exp(0.1 * d.clamp(min=-10, max=10))
                # amp_rew = self.amp_reward_coef * torch.exp(torch.tanh(0.4 * d))
                # amp_rew = self.amp_reward_coef * torch.exp(torch.tanh(d))
            elif self.amp_type == AMPType.bce.name:
                d = self.forward(generated_motion)
                d = torch.sigmoid(d)
                amp_rew = -self.amp_reward_coef * torch.log(torch.clamp(1 - d, min=1e-4))
            self.train()
            total_rew = amp_rew.squeeze() + self.task_rew_coef * task_rew
        return total_rew, amp_rew.squeeze()

    def compute_amp_loss(self, ref_motion, generated_motion):
        if self.amp_type == AMPType.least_square.name:
            ref_motion.requires_grad = True
            disc = self.forward(ref_motion)
            ones = torch.ones(disc.size(), device=disc.device)
            grad = autograd.grad(
                outputs=disc, inputs=ref_motion,
                grad_outputs=ones, create_graph=True,
                retain_graph=True, only_inputs=True)[0]
            ref_motion.requires_grad = False
            # Enforce that the grad norm approaches 0.
            grad_pen = self.lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
            ref_d = self.forward(ref_motion)
            gen_d = self.forward(generated_motion)
            ref_loss = torch.nn.MSELoss()(
                ref_d, torch.ones(ref_d.size(), device=disc.device))
            gen_loss = torch.nn.MSELoss()(
                gen_d, -1 * torch.ones(gen_d.size(), device=disc.device))

        elif self.amp_type == AMPType.wasserstein.name:
            # data = self.alpha * ref_motion + (1-self.alpha) * generated_motion
            alpha = torch.rand(1, device=ref_motion.device)
            data = alpha * ref_motion + (1 - alpha) * generated_motion
            data.requires_grad = True
            disc = self.forward(data)
            ones = torch.ones(disc.size(), device=disc.device)
            grad = autograd.grad(
                outputs=disc, inputs=data,
                grad_outputs=ones, create_graph=True,
                retain_graph=True, only_inputs=True)[0]
            # Wasserstein GAN with Gradient Penalty
            grad_pen = self.lambda_ * (grad.norm(2, dim=1) - 1).clip(min=0).pow(2).mean()
            ref_regress = - torch.tanh(0.4 * self.forward(ref_motion)).mean()
            gen_regress = torch.tanh(0.4 * self.forward(generated_motion)).mean()
            ref_loss = ref_regress
            gen_loss = gen_regress

        elif self.amp_type == AMPType.bce.name:
            ref_motion.requires_grad = True
            disc = self.forward(ref_motion)
            ones = torch.ones(disc.size(), device=disc.device)
            grad = autograd.grad(
                outputs=disc, inputs=ref_motion,
                grad_outputs=ones, create_graph=True,
                retain_graph=True, only_inputs=True)[0]
            grad_pen = self.lambda_ * (grad.norm(2, dim=1) - 1).clip(min=0).pow(2).mean()

            ref_motion.requires_grad = False
            ref_d = self.forward(ref_motion)
            gen_d = self.forward(generated_motion)
            ref_loss = torch.nn.BCEWithLogitsLoss()(ref_d, torch.ones(ref_d.size(), device=disc.device))
            gen_loss = torch.nn.BCEWithLogitsLoss()(gen_d, torch.zeros(gen_d.size(), device=disc.device))

        amp_loss = ref_loss + gen_loss
        return amp_loss, grad_pen, gen_d.mean(), ref_d.mean()

    def update_lambda(self, gen_d):
        with torch.no_grad():
            self.lambda_schedule(gen_d)

    def _build_lambda_schedule(self, lambda_schedule_dict):
        schedule_type = lambda_schedule_dict["schedule_type"]
        para = lambda_schedule_dict["lambda1"]
        self.lambda_ = para[0]
        self.lambda_low = para[1]
        self.lambda_high = para[2]
        self.lambda_ema = para[3]

        if schedule_type is None:
            self.lambda_schedule = self.lambda_schedule_init
        elif schedule_type == "linear":
            self.lambda_schedule = self.lambda_schedule_linear()
        elif schedule_type == "exp":
            self.lambda_schedule = self.lambda_schedule_exp()
        elif schedule_type == "inverse":
            self.lambda_schedule = self.lambda_schedule_inverse()

    def lambda_schedule_linear(self):
        d0, d1 = -1, 0
        self.k = (self.lambda_high - self.lambda_low) / (d0 - d1)
        self.d = self.lambda_low

        def linear_schedule(self, gen_d):
            new_lambda = self.k * gen_d + self.d
            new_lambda = max(self.lambda_low, min(new_lambda, self.lambda_high))
            self.lambda_ = (1 - self.lambda_ema) * self.lambda_ + self.lambda_ema * new_lambda

        return linear_schedule.__get__(self, type(self))

    def lambda_schedule_inverse(self):
        # 1/lambda = A*d + B
        d0, d1 = -1, 0
        self.A = (1.0 / self.lambda_high - 1.0 / self.lambda_low) / (d0 - d1)
        self.B = 1.0 / self.lambda_low

        def inverse_schedule(self, gen_d):
            new_lambda = 1 / (self.A * gen_d + self.B)
            new_lambda = max(self.lambda_low, min(new_lambda, self.lambda_high))
            self.lambda_ = (1 - self.lambda_ema) * self.lambda_ + self.lambda_ema * new_lambda

        return inverse_schedule.__get__(self, type(self))

    def lambda_schedule_exp(self):
        # A*B^-d =
        d0, d1 = -1, 0
        self.A = self.lambda_low
        self.B = abs((self.lambda_high / self.lambda_low) / (d0 - d1))

        def exp_schedule(self, gen_d):
            new_lambda = self.A * self.B ** (-gen_d)
            new_lambda = max(self.lambda_low, min(new_lambda, self.lambda_high))
            self.lambda_ = (1 - self.lambda_ema) * self.lambda_ + self.lambda_ema * new_lambda

        return exp_schedule.__get__(self, type(self))

    def lambda_schedule_init(self, gen_d):
        pass

    def _build_task_rew_schedule(self, task_rew_schedule_dict):
        self.task_rew_coef = 1.0
        self.using_task_rew_schedule = task_rew_schedule_dict.get("using_schedule", False)
        if self.using_task_rew_schedule:
            self.tracking_lin_vel_rew_buf = torch.zeros(int(task_rew_schedule_dict["buffer_size"]), device=self.device)
            self.tracking_lin_vel_rew_buf_num = 0
            self.task_rew_coef_update_step = task_rew_schedule_dict["update_step"]
            self.task_rew_coef_min = task_rew_schedule_dict["task_rew_coef_min"]
            self.update_threshold = task_rew_schedule_dict["update_threshold"]
            self.traking_lin_vel_max = task_rew_schedule_dict["traking_lin_vel_max"]

    def update_task_rew_coef(self, rew):
        if not self.using_task_rew_schedule or rew is None:
            return
        self.tracking_lin_vel_rew_buf = self.tracking_lin_vel_rew_buf.roll(rew.shape[0])
        self.tracking_lin_vel_rew_buf[:rew.shape[0]] = rew
        self.tracking_lin_vel_rew_buf_num += rew.shape[0]

        # self.tracking_lin_vel_rew_buf = self.tracking_lin_vel_rew_buf.roll(1)
        # self.tracking_lin_vel_rew_buf[:1] = rew
        # self.tracking_lin_vel_rew_buf_num += 1

        if self.tracking_lin_vel_rew_buf_num >= self.tracking_lin_vel_rew_buf.shape[0]:
            if self.tracking_lin_vel_rew_buf.mean() > self.update_threshold * self.traking_lin_vel_max:
                self.task_rew_coef -= self.task_rew_coef_update_step
                self.task_rew_coef = max(self.task_rew_coef, self.task_rew_coef_min)
                self.tracking_lin_vel_rew_buf_num = 0


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
