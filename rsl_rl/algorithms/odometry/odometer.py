import torch

from rsl_rl.modules.model_odom import OdomTransformer
from rsl_rl.storage.odometer_storage import OdometerStorage

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


def masked_MSE(input_, target, mask):
    return ((input_ - target) * mask).square().sum() / (input_.numel() / mask.numel() * mask.sum())


def masked_L1(input_, target, mask):
    return ((input_ - target) * mask).abs().sum() / (input_.numel() / mask.numel() * mask.sum())


def masked_mean(input_, mask):
    return (input_ * mask).sum() / (input_.numel() / mask.numel() * mask.sum())


class Transition:
    def __init__(self):
        self.observations = None
        self.hidden_states = None
        self.dones = None

    def get_items(self):
        return self.__dict__.items()

    def clear(self):
        for key in self.__dict__:
            setattr(self, key, None)


class Odometer:
    def __init__(self, task_cfg, device=torch.device('cpu')):
        # PPO parameters
        self.task_cfg = task_cfg
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        self.cur_it = 0

        # Odometer components
        self.odom = OdomTransformer(
            task_cfg.env.n_proprio,
            task_cfg.policy.odom_transformer_embed_dim,
            task_cfg.policy.odom_gru_hidden_size,
            task_cfg.policy.estimator_output_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.odom.parameters(), lr=task_cfg.policy.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)
        self.loss_mse = torch.nn.MSELoss()
        self.loss_l1 = torch.nn.L1Loss()
        self.loss_bce = torch.nn.BCEWithLogitsLoss()

        self.selected_indices = None
        self.odom_update_infos = {}

        # Rollout Storage
        self.transition = Transition()
        self.storage = OdometerStorage(self.task_cfg.policy.batch_size, 24, device=self.device)

    def reconstruct(self, obs):
        # act function should run within torch.inference_mode context
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            # store observations
            self.transition.observations = obs
            self.transition.hidden_states = self.odom.hidden_states

            _, recon_refine, priv_est = self.odom.inference_forward(obs.proprio, obs.depth, eval_=True)

            if self.transition.hidden_states is None:
                self.transition.hidden_states = torch.zeros_like(self.odom.hidden_states)

            return recon_refine, priv_est

    def process_env_step(self, dones):
        self.transition.dones = dones.unsqueeze(1)

        # data selection
        if self.selected_indices is None:
            obs = self.transition.observations

            unique_env_classes = torch.unique(obs.env_class)
            num_per_class = self.task_cfg.policy.batch_size // len(unique_env_classes)
            selected_indices = []

            for env_class in unique_env_classes:
                class_indices = torch.where(obs.env_class == env_class)[0]

                if len(class_indices) >= num_per_class:
                    selected = torch.randperm(len(class_indices), device=self.device)[:num_per_class]
                    selected_indices.append(class_indices[selected])
                else:
                    selected_indices.append(class_indices)

            self.selected_indices = torch.cat(selected_indices)

        # Record the transition
        self.transition.observations = self.transition.observations[self.selected_indices]
        self.transition.hidden_states = self.transition.hidden_states[:, self.selected_indices]
        self.transition.dones = self.transition.dones[self.selected_indices]

        self.storage.add_transitions(self.transition)
        self.transition.clear()

        self.odom.reset(dones)

    def update(self):
        if not self.storage.is_full():
            return {}

        num_updates = 0
        mean_recon_rough_loss = 0
        mean_recon_refine_loss = 0
        mean_edge_loss = 0
        mean_priv_loss = 0

        for batch in self.storage.recurrent_mini_batch_generator(num_epochs=4):
            with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
                obervations = batch['observations']

                proprio = obervations['proprio'].half()
                depth = obervations['depth'].half()
                hidden_states = batch['hidden_states'].half()
                rough_scan = obervations['rough_scan'].half()
                scan = obervations['scan'].half()
                priv = obervations['priv_actor'].half()

                recon_rough, recon_refine, priv_est = self.odom(proprio, depth, hidden_states)

            loss_recon_rough = self.loss_mse(recon_rough, rough_scan.unsqueeze(2))
            loss_recon_refine = self.loss_l1(recon_refine[:, 0], scan[:, 0])
            loss_edge = self.loss_bce(recon_refine[:, 1], scan[:, 1])
            loss_priv = self.loss_mse(priv_est, priv)

            # Gradient step
            self.optimizer.zero_grad()
            self.scaler.scale(10. * loss_recon_rough + loss_recon_refine + 0.1 * loss_edge + loss_priv).backward()
            # self.scaler.unscale_(self.optimizer)
            # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            num_updates += 1
            mean_recon_rough_loss += loss_recon_rough.item()
            mean_recon_refine_loss += loss_recon_refine.item()
            mean_edge_loss += loss_edge.item()
            mean_priv_loss += loss_priv.item()

        mean_recon_rough_loss /= num_updates
        mean_recon_refine_loss /= num_updates
        mean_edge_loss /= num_updates
        mean_priv_loss /= num_updates

        return {
            'Loss/recon_rough_loss': mean_recon_rough_loss,
            'Loss/recon_refine_loss': mean_recon_refine_loss,
            'Loss/edge_loss': mean_edge_loss,
            'Loss/priv_loss': mean_priv_loss,
        }

    def _update_obom_each_step(self, obs):
        if self.cur_it < self.task_cfg.policy.update_since:
            return

        # data selection
        if self.selected_indices is None:
            unique_env_classes = torch.unique(obs.env_class)
            num_per_class = self.task_cfg.policy.batch_size // len(unique_env_classes)
            selected_indices = []

            for env_class in unique_env_classes:
                class_indices = torch.where(obs.env_class == env_class)[0]

                if len(class_indices) >= num_per_class:
                    selected = torch.randperm(len(class_indices), device=self.device)[:num_per_class]
                    selected_indices.append(class_indices[selected])
                else:
                    selected_indices.append(class_indices)

            self.selected_indices = torch.cat(selected_indices)

        # odometer update
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.cfg.use_amp):
            prop = obs.proprio[self.selected_indices]
            depth = obs.depth[self.selected_indices]
            rough_scan = obs.rough_scan[self.selected_indices]
            scan = obs.scan[self.selected_indices]
            priv_actor = obs.priv_actor[self.selected_indices]

            recon_rough, recon_refine, est = self.odom.inference_forward(prop, depth, eval_=False)

            loss_recon_rough = self.loss_mse(recon_rough, rough_scan.unsqueeze(1))
            loss_recon_refine = self.loss_l1(recon_refine[:, 0], scan[:, 0])
            loss_edge = self.loss_bce(recon_refine[:, 1], scan[:, 1])
            loss_priv = self.loss_mse(est, priv_actor)
            self.scaler.scale(loss_recon_rough + loss_recon_refine + loss_edge + loss_priv).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()

            self.odom_update_infos['Loss/loss_recon_rough'] = loss_recon_rough.item()
            self.odom_update_infos['Loss/loss_recon_refine'] = loss_recon_refine.item()
            self.odom_update_infos['Loss/loss_edge'] = loss_edge.item()
            self.odom_update_infos['Loss/loss_priv'] = loss_priv.item()

    def play_reconstruct(self, obs, **kwargs):
        with torch.autocast(self.device.type, torch.float16, enabled=self.cfg.use_amp):
            recon_rough, recon_refine, est = self.odom.inference_forward(obs.proprio, obs.depth, eval_=True)

            recon_refine[:, 1] = torch.where(recon_refine[:, 1] < 0., 0., 1.)

            return {
                'recon_rough': recon_rough,
                'recon_refine': recon_refine,
                'estimation': est,
            }

    def load(self, loaded_dict, load_optimizer=True):

        if 'odometer_state_dict' in loaded_dict:
            self.odom.load_state_dict(loaded_dict['odometer_state_dict'])

    def save(self):
        return {'odometer_state_dict': self.odom.state_dict()}
