import datetime
import os

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from rsl_rl.modules.model_odom import OdomTransformer, Actor
from rsl_rl.modules.utils import UniversalCritic
from rsl_rl.storage import RolloutStorageMultiCritic as RolloutStorage
from .alg_base import BaseAlgorithm

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
        self.critic_observations = None
        self.actor_hidden_states = None
        self.recon = None
        self.priv = None
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


class PPO_Odom(BaseAlgorithm):
    def __init__(self, task_cfg, device=torch.device('cpu'), env=None, **kwargs):
        self.env = env

        # PPO parameters
        self.task_cfg = task_cfg
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        # PPO component
        self.odom = OdomTransformer(task_cfg.env, task_cfg.policy).to(self.device)
        self.actor = Actor(task_cfg.env, task_cfg.policy).to(self.device)
        self.actor.reset_std(self.cfg.init_noise_std)

        self.critic = nn.ModuleDict({
            'default': UniversalCritic(task_cfg.env, task_cfg.policy),
            'contact': UniversalCritic(task_cfg.env, task_cfg.policy),
        }).to(self.device)
        self.optimizer = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)

        # Rollout Storage
        self.transition = Transition()
        self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)

        # dataset for odometry estimation
        self.save_odom_data = False

        if self.save_odom_data:
            self.dataset_capacity = 10000
            self.data_length = 50

            self.buffer = {
                'prop': torch.zeros(task_cfg.env.num_envs, self.data_length, task_cfg.env.n_proprio, dtype=torch.half, device=self.device),
                'depth': torch.zeros(task_cfg.env.num_envs, self.data_length, 1, 64, 64, dtype=torch.half, device=self.device),
                'recon': torch.zeros(task_cfg.env.num_envs, self.data_length, 2, *task_cfg.env.scan_shape, dtype=torch.half, device=self.device),
                'priv': torch.zeros(task_cfg.env.num_envs, self.data_length, task_cfg.policy.estimator_output_dim, dtype=torch.half, device=self.device),
                'masks': torch.zeros(task_cfg.env.num_envs, self.data_length, dtype=torch.bool, device=self.device),
            }
            self.buffer_ptr = torch.zeros(task_cfg.env.num_envs, dtype=torch.long, device=self.device)

            self.dataset = {
                'prop': torch.empty(self.dataset_capacity, self.data_length, task_cfg.env.n_proprio, dtype=torch.half, device='cpu'),
                'depth': torch.empty(self.dataset_capacity, self.data_length, 1, 64, 64, dtype=torch.half, device='cpu'),
                'recon': torch.empty(self.dataset_capacity, self.data_length, 2, *task_cfg.env.scan_shape, dtype=torch.half, device='cpu'),
                'priv': torch.empty(self.dataset_capacity, self.data_length, task_cfg.policy.estimator_output_dim, dtype=torch.half, device='cpu'),
                'masks': torch.empty(self.dataset_capacity, self.data_length, dtype=torch.bool, device='cpu'),
            }
            self.dataset_idx = 0

    def act(self, obs, obs_critic, use_estimated_values: torch.Tensor = None, **kwargs):
        if self.save_odom_data:
            # Store data in buffer at current pointer
            self.buffer['prop'][:, self.buffer_ptr] = obs.proprio.half()
            self.buffer['depth'][:, self.buffer_ptr] = obs.depth.half()
            self.buffer['recon'][:, self.buffer_ptr] = obs.scan.half()
            self.buffer['priv'][:, self.buffer_ptr] = obs.priv_actor.half()
            self.buffer['masks'][:, self.buffer_ptr] = True
            self.buffer_ptr[:] += 1

        # act function should run within torch.inference_mode context
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            # store observations
            self.transition.observations = obs.no_depth()
            self.transition.critic_observations = obs_critic
            self.transition.actor_hidden_states = self.actor.get_hidden_states()

            if torch.any(use_estimated_values):
                recon, priv = self.odom(obs.proprio, obs.depth)
                self.transition.recon = recon
                self.transition.priv = priv
                actions = self.actor.act(obs, recon, priv, use_estimated_values=use_estimated_values)
            else:
                actions = self.actor.act(obs, None, None, use_estimated_values=use_estimated_values)

            if self.transition.actor_hidden_states is None:
                # only for the first step where hidden_state is None
                self.transition.actor_hidden_states = 0 * self.actor.get_hidden_states()

            # store
            self.transition.actions = actions
            self.transition.values = self.critic['default'].evaluate(obs_critic)
            self.transition.values_contact = self.critic['contact'].evaluate(obs_critic)
            self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions)
            self.transition.action_mean = self.actor.action_mean
            self.transition.action_sigma = self.actor.action_std
            self.transition.use_estimated_values = use_estimated_values
            return actions

    def process_env_step(self, rewards, dones, infos, *args):
        self.transition.dones = dones.unsqueeze(1)

        # from Logan
        rew_elements = infos['rew_elements']
        # rew_contact = rew_elements['feet_edge'].clone().unsqueeze(1)
        rew_contact = (rew_elements['feet_contact_forces'] + rew_elements['feet_stumble'] + rew_elements['foothold']).clone().unsqueeze(1)
        rew_default = rewards.clone().unsqueeze(1) - rew_contact
        self.transition.rewards = rew_default
        self.transition.rewards_contact = rew_contact

        # Bootstrapping on time-outs
        if 'time_outs' in infos:
            if 'reach_goals' in infos:
                bootstrapping = (infos['time_outs'] | infos['reach_goals']).unsqueeze(1).to(self.device)
            else:
                bootstrapping = (infos['time_outs']).unsqueeze(1).to(self.device)

            self.transition.rewards += self.cfg.gamma * self.transition.values * bootstrapping
            self.transition.rewards_contact += self.cfg.gamma * self.transition.values_contact * bootstrapping

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor.reset(dones)

        if self.save_odom_data:
            # Buffer logic: if any env is done, transfer it's buffer to dataset
            finished = dones | (self.buffer_ptr >= self.data_length)
            if torch.any(finished):
                num_finished = torch.sum(finished).item()

                if self.dataset_idx + num_finished > self.dataset_capacity:
                    # Save dataset to disk
                    save_dir = os.path.join('/home/harry/projects/parkour_genesis/logs/dataset')
                    os.makedirs(save_dir, exist_ok=True)
                    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    torch.save(self.dataset, os.path.join(save_dir, f'{now}.pt'))

                    # clear the dataset
                    for key in self.dataset:
                        self.dataset[key].zero_()
                    self.dataset_idx = 0
                else:
                    self.dataset['prop'][self.dataset_idx: self.dataset_idx + num_finished] = self.buffer['prop'][finished].cpu()
                    self.dataset['depth'][self.dataset_idx: self.dataset_idx + num_finished] = self.buffer['depth'][finished].cpu()
                    self.dataset['recon'][self.dataset_idx: self.dataset_idx + num_finished] = self.buffer['recon'][finished].cpu()
                    self.dataset['priv'][self.dataset_idx: self.dataset_idx + num_finished] = self.buffer['priv'][finished].cpu()
                    self.dataset['masks'][self.dataset_idx: self.dataset_idx + num_finished] = self.buffer['masks'][finished].cpu()
                    self.dataset_idx += num_finished

                self.buffer_ptr[finished] = 0
                self.buffer['prop'][finished] = 0
                self.buffer['depth'][finished] = 0
                self.buffer['recon'][finished] = 0
                self.buffer['priv'][finished] = 0
                self.buffer['masks'][finished] = False

    def compute_returns(self, last_critic_obs):
        last_values_default = self.critic['default'].evaluate(last_critic_obs).detach()
        last_values_contact = self.critic['contact'].evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values_default, last_values_contact, self.cfg.gamma, self.cfg.lam)

    def update(self, cur_it=0, **kwargs):
        mean_value_loss = 0
        mean_default_value_loss = 0
        mean_contact_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_symmetry_loss = 0

        kl_change = []
        num_updates = 0

        for batch in self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):
            # ########################## policy loss ##########################
            kl_mean, value_loss_default, value_loss_contact, surrogate_loss, entropy_loss, symmetry_loss = self._compute_policy_loss(batch)
            loss = surrogate_loss + self.cfg.value_loss_coef * (value_loss_default + value_loss_contact) - entropy_loss + symmetry_loss

            num_updates += 1
            # policy statistics
            kl_change.append(kl_mean.item())
            mean_kl += kl_mean.item()
            mean_default_value_loss += value_loss_default.item()
            mean_contact_value_loss += value_loss_contact.item()
            mean_value_loss = mean_default_value_loss + mean_contact_value_loss
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_loss.item()
            # mean_symmetry_loss += symmetry_loss.item()

            # Use KL to adaptively update learning rate
            if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
                if kl_mean > self.cfg.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                    self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            # Gradient step
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            # self.scaler.unscale_(self.optimizer)
            # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        mean_kl /= num_updates
        mean_value_loss /= num_updates
        mean_default_value_loss /= num_updates
        mean_contact_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_symmetry_loss /= num_updates

        kl_str = 'kl: '
        for k in kl_change:
            kl_str += f'{k:.3f} | '
        print(kl_str)

        self.storage.clear()
        return_dict = {
            'Loss/learning_rate': self.learning_rate,
            'Loss/kl_div': mean_kl,
            'Loss/value_loss': mean_value_loss,
            'Loss/default_value_loss': mean_default_value_loss,
            'Loss/contact_value_loss': mean_contact_value_loss,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Loss/symmetry_loss': mean_symmetry_loss,
            'Train/noise_std': self.actor.log_std.exp().mean().item(),
        }

        return return_dict

    # @torch.compile
    def _compute_policy_loss(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            actor_hidden_states_batch = batch['actor_hidden_states']
            recon_batch = batch['recon'] if 'recon' in batch else None
            priv_batch = batch['priv'] if 'priv' in batch else None
            mask_batch = batch['masks']
            actions_batch = batch['actions']
            default_values_batch = batch['values']
            contact_values_batch = batch['values_contact']
            advantages_batch = batch['advantages']
            default_returns_batch = batch['returns']
            contact_returns_batch = batch['returns_contact']
            old_actions_log_prob_batch = batch['actions_log_prob']
            use_estimated_values_batch = batch['use_estimated_values']

            self.actor.train_act(
                obs_batch,
                recon_batch,
                priv_batch,
                hidden_states=actor_hidden_states_batch,
                use_estimated_values=use_estimated_values_batch
            )

            with torch.no_grad():
                kl_mean = kl_divergence(
                    Normal(batch['action_mean'], batch['action_sigma']),
                    Normal(self.actor.action_mean, self.actor.action_std)
                ).sum(dim=2, keepdim=True)
                kl_mean = masked_mean(kl_mean, mask_batch)

            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            evaluation_default = self.critic['default'].evaluate(critic_obs_batch)
            evaluation_contact = self.critic['contact'].evaluate(critic_obs_batch)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate_loss = masked_mean(torch.maximum(surrogate, surrogate_clipped), mask_batch)

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped_default = default_values_batch + (
                        evaluation_default - default_values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses_default = (evaluation_default - default_returns_batch).square()
                value_losses_clipped_default = (value_clipped_default - default_returns_batch).square()
                value_losses_default = masked_mean(torch.maximum(value_losses_default, value_losses_clipped_default), mask_batch)

                value_clipped_contact = contact_values_batch + (
                        evaluation_contact - contact_values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses_contact = (evaluation_contact - contact_returns_batch).square()
                value_losses_clipped_contact = (value_clipped_contact - contact_returns_batch).square()
                value_losses_contact = masked_mean(torch.max(value_losses_contact, value_losses_clipped_contact), mask_batch)
            else:
                value_losses_default = masked_MSE(evaluation_default, default_returns_batch, mask_batch)
                value_losses_contact = masked_MSE(evaluation_contact, contact_returns_batch, mask_batch)

            # Entropy loss
            entropy_loss = self.cfg.entropy_coef * masked_mean(self.actor.entropy, mask_batch)

            # # Symmetry loss
            # batch_size = 4
            # action_mean_original = self.actor.action_mean[:batch_size].detach()
            #
            # obs_mirrored_batch = obs_batch[:batch_size].flatten(0, 1).mirror().unflatten(0, (batch_size, -1))
            # self.actor.train_act(
            #     obs_mirrored_batch,
            #     recon_batch,
            #     priv_batch,
            #     hidden_states=actor_hidden_states_batch,
            #     use_estimated_values=use_estimated_values_batch
            # )
            #
            # mu_batch = obs_batch.mirror_dof_prop_by_x(action_mean_original.flatten(0, 1)).unflatten(0, (batch_size, -1))
            # symmetry_loss = 0.1 * self.mse_loss(mu_batch, self.actor.action_mean)

            return kl_mean, value_losses_default, value_losses_contact, surrogate_loss, entropy_loss, 0.

    # @torch.compile
    def _compute_estimation_loss(self, batch: dict, batch_size=128):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            batch_idx = torch.randperm(batch['masks'].size(1))[:batch_size]

            obs_batch = batch['observations'][:, batch_idx]
            obs_enc_hidden_states_batch = batch['obs_enc_hidden_states'][:, batch_idx].contiguous()
            recon_hidden_states_batch = batch['recon_hidden_states'][:, batch_idx].contiguous()
            obs_next_batch = batch['observations_next'][:, batch_idx]
            mask_batch = batch['masks'][:, batch_idx]

            if self.enable_VAE:
                recon_rough, recon_refine, est_latent, est, est_logvar, ot1 = self.actor.reconstruct(
                    obs_batch, obs_enc_hidden_states_batch, recon_hidden_states_batch)

                # Ot+1 prediction and VAE loss
                prediction_loss = masked_MSE(ot1, obs_next_batch.proprio, mask_batch)
                vae_loss = (1 + est_logvar - est_latent.pow(2) - est_logvar.exp()).sum(dim=2, keepdims=True)
                vae_loss = -0.5 * masked_mean(vae_loss, mask_batch)

            else:
                recon_rough, recon_refine, est = self.actor.reconstruct(
                    obs_batch, obs_enc_hidden_states_batch, recon_hidden_states_batch)

            # privileged information estimation loss
            estimation_loss = masked_MSE(est, obs_batch.priv_actor, mask_batch)

            # reconstructor loss
            recon_rough_loss = masked_MSE(recon_rough, obs_batch.scan, mask_batch[:, :, :, None, None])
            recon_refine_loss = masked_L1(recon_refine, obs_batch.scan, mask_batch[:, :, :, None, None])

        if self.enable_VAE:
            return estimation_loss, prediction_loss, vae_loss, recon_rough_loss, recon_refine_loss
        else:
            return estimation_loss, recon_rough_loss, recon_refine_loss

    def play_act(self, obs, **kwargs):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            if 'use_estimated_values' in kwargs:
                use_estimated_values_bool = kwargs['use_estimated_values']
                kwargs['use_estimated_values'] = use_estimated_values_bool & torch.ones(
                    self.task_cfg.env.num_envs, 1, dtype=torch.bool, device=self.device)

            mean = self.actor.act(obs, None, None, **kwargs)
            return {'actions': mean}

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):
        # try:
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std)

    def save(self):
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
