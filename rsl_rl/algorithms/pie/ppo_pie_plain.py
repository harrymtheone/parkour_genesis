import torch
from torch import nn, optim

from rsl_rl.algorithms.utils import masked_MSE, masked_L1
from rsl_rl.modules.pie import PolicyPlain
from rsl_rl.modules.utils import UniversalCritic
from rsl_rl.storage import RolloutStorageMultiCritic as RolloutStorage
from . import PPO_PIE_MC

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class Transition:
    def __init__(self):
        self.observations = None
        self.critic_observations = None
        self.hidden_states = None
        self.observations_next = None
        self.actions = None
        self.rewards = None
        self.rewards_contact = None
        self.dones = None
        self.values = None
        self.values_contact = None
        self.actions_log_prob = None
        self.action_mean = None
        self.action_sigma = None
        self.use_estimated_values = None

    def get_items(self):
        return self.__dict__.items()

    def clear(self):
        for key in self.__dict__:
            setattr(self, key, None)


class PPO_PIE_Plain(PPO_PIE_MC):
    def __init__(self, task_cfg, device=torch.device('cpu'), env=None, **kwargs):
        super().__init__(task_cfg, device, env, **kwargs)
        self.env = env

        # PPO parameters
        self.cfg = task_cfg.algorithm
        self.task_cfg = task_cfg
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        # PPO components
        self.actor = PolicyPlain(task_cfg.env, task_cfg.policy).to(self.device)
        self.actor.reset_std(self.cfg.init_noise_std, device=self.device)
        if self.actor.is_recurrent:
            self.mixer_hidden_states = None

        self.critic = nn.ModuleDict({
            'default': UniversalCritic(task_cfg.env, task_cfg.policy),
            'contact': UniversalCritic(task_cfg.env, task_cfg.policy),
        }).to(self.device)

        # Create separate optimizers for different update types
        self.ppo_optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.ppo_scaler = GradScaler(enabled=self.cfg.use_amp)
        self.actor_scaler = GradScaler(enabled=self.cfg.use_amp)

        # Rollout Storage
        self.transition = Transition()
        self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)

    def update(self, cur_it=0, **kwargs):
        # Initialize metrics
        mean_value_loss_default = 0
        mean_value_loss_contact = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_vel_est_loss = 0
        mean_ot1_loss = 0
        mean_recon_loss = 0
        mean_symmetry_loss = 0
        kl_change = []
        num_updates = 0

        if self.actor.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)

        for batch in generator:
            # PPO Update
            ppo_metrics = self.update_ppo(cur_it, batch)
            mean_kl += ppo_metrics['kl_mean']
            mean_value_loss_default += ppo_metrics['value_loss_default']
            mean_value_loss_contact += ppo_metrics['value_loss_contact']
            mean_surrogate_loss += ppo_metrics['surrogate_loss']
            mean_entropy_loss += ppo_metrics['entropy_loss']
            kl_change.append(ppo_metrics['kl_mean'])

            # # Symmetry Update
            # sym_metrics = self.update_symmetry(cur_it, batch)
            # mean_symmetry_loss += sym_metrics['symmetry_loss']

            # Estimation Update
            if cur_it % 3 == 0:
                est_metrics = self.update_estimation(cur_it, batch)
                mean_vel_est_loss += est_metrics['vel_est_loss']
                mean_ot1_loss += est_metrics['ot1_loss']
                mean_recon_loss += est_metrics['recon_loss']

            num_updates += 1

        # Average metrics
        mean_value_loss_default /= num_updates
        mean_value_loss_contact /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates
        mean_vel_est_loss /= num_updates
        mean_ot1_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_symmetry_loss /= num_updates

        kl_str = 'kl: '
        for k in kl_change:
            kl_str += f'{k:.3f} | '
        print(kl_str)

        self.storage.clear()

        metrics = {
            'Loss/learning_rate': self.learning_rate,
            'Loss/value_loss_default': mean_value_loss_default,
            'Loss/value_loss_contact': mean_value_loss_contact,
            'Loss/kl_div': mean_kl,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Train/noise_std': self.actor.log_std.exp().mean().item(),
        }

        if cur_it % 3 == 0:
            metrics.update({
                'Loss/vel_est_loss': mean_vel_est_loss,
                'Loss/Ot+1_loss': mean_ot1_loss,
                'Loss/recon_loss': mean_recon_loss,
                'Loss/symmetry_loss': mean_symmetry_loss,
            })

        return metrics

    def update_estimation(self, cur_it, batch, **kwargs):
        """Update estimation-related components (estimation, prediction, VAE, recon losses)"""
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs = batch['observations']
            critic_obs = batch['critic_observations']
            hidden_states = batch['hidden_states'] if self.actor.is_recurrent else None
            mask = batch['masks'] if self.actor.is_recurrent else slice(None)
            obs_next = batch['observations_next']

            # Forward pass
            vel, z, ot1, hmap = self.actor.estimate(obs.prop_his, obs.depth, hidden_states=hidden_states)

            # Estimation loss
            vel_est_loss = masked_MSE(vel, critic_obs.est_gt, mask)
            ot1_loss = masked_MSE(ot1, obs_next.proprio, mask)
            recon_loss = masked_L1(hmap, critic_obs.scan.flatten(2), mask)

            # Total estimation loss
            total_loss = vel_est_loss + ot1_loss + recon_loss

        # Gradient step - moved outside autocast context
        self.actor_optimizer.zero_grad()
        self.actor_scaler.scale(total_loss).backward()
        self.actor_scaler.step(self.actor_optimizer)
        self.actor_scaler.update()

        return {
            'vel_est_loss': vel_est_loss.item(),
            'ot1_loss': ot1_loss.item(),
            'recon_loss': recon_loss.item(),
        }
