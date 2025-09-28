import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

from rsl_rl.algorithms.alg_base import BaseAlgorithm
from rsl_rl.algorithms.utils import masked_mean, masked_MSE, masked_L1
from rsl_rl.modules.pie import Policy
from rsl_rl.modules.utils import UniversalCritic
from rsl_rl.storage import RolloutStorageMultiCritic as RolloutStorage

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class Transition:
    def __init__(self):
        self.observations = None
        self.critic_observations = None
        self.mixer_hidden_states = None
        self.actor_hidden_states = None
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


class PPO_PIE_MC(BaseAlgorithm):
    def __init__(self, task_cfg, device=torch.device('cpu'), env=None, **kwargs):
        self.env = env

        # PPO parameters
        self.cfg = task_cfg.algorithm
        self.task_cfg = task_cfg
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        # Initialize adaptive KL coefficient
        self.kl_coef_vel = 0.01
        self.kl_coef_z = 0.01

        # PPO components
        self.actor = Policy(task_cfg.env, task_cfg.policy).to(self.device)
        self.actor.reset_std(self.cfg.init_noise_std, device=self.device)
        if self.actor.is_recurrent:
            self.mixer_hidden_states = None
            self.actor_hidden_states = None

        self.critic = nn.ModuleDict({
            'default': UniversalCritic(task_cfg.env, task_cfg.policy),
            'contact': UniversalCritic(task_cfg.env, task_cfg.policy),
        }).to(self.device)

        # Create separate optimizers for different update types
        self.ppo_optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)

        # Rollout Storage
        self.transition = Transition()
        self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, obs_critic, use_estimated_values: torch.Tensor = None, **kwargs):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            # store observations
            self.transition.observations = obs
            self.transition.critic_observations = obs_critic
            if self.actor.is_recurrent:
                self.transition.mixer_hidden_states = self.mixer_hidden_states
                self.transition.actor_hidden_states = self.actor_hidden_states
            self.transition.use_estimated_values = use_estimated_values

            actions, self.mixer_hidden_states, self.actor_hidden_states = self.actor.act(
                proprio=obs.proprio.unsqueeze(0),
                prop_his=obs.prop_his.unsqueeze(0),
                depth=obs.depth.unsqueeze(0),
                mixer_hidden_states=self.mixer_hidden_states,
                actor_hidden_states=self.actor_hidden_states,
            )

            if self.actor.is_recurrent and self.transition.mixer_hidden_states is None:
                # only for the first step where hidden_state is None
                self.transition.mixer_hidden_states = torch.zeros_like(self.mixer_hidden_states)
                self.transition.actor_hidden_states = torch.zeros_like(self.actor_hidden_states)

            # store
            self.transition.actions = actions
            self.transition.values = self.critic['default'](
                obs_critic.priv_his.unsqueeze(0),
                obs_critic.scan.unsqueeze(0),
                obs_critic.edge_mask.unsqueeze(0),
            )
            self.transition.values_contact = self.critic['contact'](
                obs_critic.priv_his.unsqueeze(0),
                obs_critic.scan.unsqueeze(0),
                obs_critic.edge_mask.unsqueeze(0),
            )
            self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions)
            self.transition.action_mean = self.actor.action_mean
            self.transition.action_sigma = self.actor.action_std
            return actions

    def process_env_step(self, rewards, dones, infos, *args):
        self.transition.observations_next = args[0].as_obs_next()
        self.transition.dones = dones.unsqueeze(1)

        # from Logan
        step_rew: dict = infos['step_rew']
        rew_contact = torch.zeros_like(rewards)
        rew_contact += step_rew.get('feet_edge', 0.)
        rew_contact += step_rew.get('feet_contact_forces', 0.)
        rew_contact += step_rew.get('feet_stumble', 0.)
        rew_contact += step_rew.get('foothold', 0.)

        rew_default = rewards - rew_contact
        self.transition.rewards = rew_default.unsqueeze(1)
        self.transition.rewards_contact = rew_contact.unsqueeze(1)

        for name in ['action_mean', 'action_sigma', 'actions', 'actions_log_prob', 'values', 'values_contact']:
            setattr(self.transition, name, getattr(self.transition, name).squeeze(0))

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
        self.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values_default = self.critic['default'](
            last_critic_obs.priv_his.unsqueeze(0),
            last_critic_obs.scan.unsqueeze(0),
            last_critic_obs.edge_mask.unsqueeze(0),
        )
        last_values_contact = self.critic['contact'](
            last_critic_obs.priv_his.unsqueeze(0),
            last_critic_obs.scan.unsqueeze(0),
            last_critic_obs.edge_mask.unsqueeze(0),
        )
        self.storage.compute_returns(last_values_default, last_values_contact, self.cfg.gamma, self.cfg.lam)

    def update(self, cur_it=0, **kwargs):
        # Initialize metrics
        mean_value_loss_default = 0
        mean_value_loss_contact = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_symmetry_loss = 0
        mean_vel_est_loss = 0
        mean_ot1_loss = 0
        mean_recon_loss = 0
        mean_vel_kl_loss = 0
        mean_z_kl_loss = 0
        mean_mu_vel = 0
        mean_mu_z = 0
        mean_std_vel = 0
        mean_std_z = 0
        mean_kl_coef_vel = 0
        mean_kl_coef_z = 0
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

            # Symmetry Update
            # sym_metrics = self.update_symmetry(cur_it, batch)
            # mean_symmetry_loss += sym_metrics['symmetry_loss']

            # Estimation Update
            if cur_it % 3 == 0:
                est_metrics = self.update_estimation(cur_it, batch)
                mean_vel_est_loss += est_metrics['vel_est_loss']
                mean_ot1_loss += est_metrics['ot1_loss']
                mean_recon_loss += est_metrics['recon_loss']
                mean_vel_kl_loss += est_metrics['vel_kl_loss']
                mean_z_kl_loss += est_metrics['z_kl_loss']
                mean_mu_vel += est_metrics['mu_vel']
                mean_mu_z += est_metrics['mu_z']
                mean_std_vel += est_metrics['std_vel']
                mean_std_z += est_metrics['std_z']
                mean_kl_coef_vel += est_metrics['kl_coef_vel']
                mean_kl_coef_z += est_metrics['kl_coef_z']

            num_updates += 1

        # Average metrics
        mean_value_loss_default /= num_updates
        mean_value_loss_contact /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates
        mean_symmetry_loss /= num_updates
        mean_vel_est_loss /= num_updates
        mean_ot1_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_vel_kl_loss /= num_updates
        mean_z_kl_loss /= num_updates
        mean_mu_vel /= num_updates
        mean_mu_z /= num_updates
        mean_std_vel /= num_updates
        mean_std_z /= num_updates
        mean_kl_coef_vel /= num_updates
        mean_kl_coef_z /= num_updates

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
                'Loss/symmetry_loss': mean_symmetry_loss,
                'Loss/vel_est_loss': mean_vel_est_loss,
                'Loss/Ot+1_loss': mean_ot1_loss,
                'Loss/recon_loss': mean_recon_loss,
                'Loss/vel_kl_loss': mean_vel_kl_loss,
                'Loss/z_kl_loss': mean_z_kl_loss,
                'Loss/total_kl_loss': mean_vel_kl_loss + mean_z_kl_loss,
                'VAE/mu_vel': mean_mu_vel,
                'VAE/mu_z': mean_mu_z,
                'VAE/std_vel': mean_std_vel,
                'VAE/std_z': mean_std_z,
                'VAE/kl_coef_vel': mean_kl_coef_vel,
                'VAE/kl_coef_z': mean_kl_coef_z,
            })

        return metrics

    def update_ppo(self, cur_it, batch):
        """Update PPO policy-related components (surrogate, value, entropy losses)"""
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs = batch['observations']
            critic_obs = batch['critic_observations']
            mixer_hidden_states = batch['mixer_hidden_states'] if self.actor.is_recurrent else None
            actor_hidden_states = batch['actor_hidden_states'] if self.actor.is_recurrent else None
            mask = batch['masks'] if self.actor.is_recurrent else slice(None)
            actions = batch['actions']
            default_values = batch['values']
            contact_values = batch['values_contact']
            advantages = batch['advantages']
            default_returns = batch['returns']
            contact_returns = batch['returns_contact']
            old_actions_log_prob = batch['actions_log_prob']

            # Forward pass
            self.actor.act(
                proprio=obs.proprio,
                prop_his=obs.prop_his,
                depth=obs.depth,
                mixer_hidden_states=mixer_hidden_states,
                actor_hidden_states=actor_hidden_states,
            )

            # KL divergence calculation
            with torch.no_grad():
                kl_mean = kl_divergence(
                    Normal(batch['action_mean'], batch['action_sigma']),
                    Normal(self.actor.action_mean, self.actor.action_std)
                ).sum(dim=2, keepdim=True)
                kl_mean = masked_mean(kl_mean, mask)

            if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
                # Adaptive learning rate
                if kl_mean > self.cfg.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                    self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                for param_group in self.ppo_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            # Policy loss
            actions_log_prob = self.actor.get_actions_log_prob(actions)
            evaluation_default = self.critic['default'](critic_obs.priv_his, critic_obs.scan, critic_obs.edge_mask)
            evaluation_contact = self.critic['contact'](critic_obs.priv_his, critic_obs.scan, critic_obs.edge_mask)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - old_actions_log_prob)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate_loss = masked_mean(torch.max(surrogate, surrogate_clipped), mask)

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped_default = default_values + (
                        evaluation_default - default_values).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_loss_default = (evaluation_default - default_returns).square()
                value_loss_clipped_default = (value_clipped_default - default_returns).square()
                value_loss_default = masked_mean(torch.maximum(value_loss_default, value_loss_clipped_default), mask)

                value_clipped_contact = contact_values + (
                        evaluation_contact - contact_values).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_loss_contact = (evaluation_contact - contact_returns).square()
                value_loss_clipped_contact = (value_clipped_contact - contact_returns).square()
                value_loss_contact = masked_mean(torch.max(value_loss_contact, value_loss_clipped_contact), mask)
            else:
                value_loss_default = masked_MSE(evaluation_default, default_returns, mask)
                value_loss_contact = masked_MSE(evaluation_contact, contact_returns, mask)

            value_loss = value_loss_default + value_loss_contact

            # Entropy loss
            entropy_loss = -masked_mean(self.actor.entropy, mask)

            # Total PPO loss
            total_loss = surrogate_loss + self.cfg.value_loss_coef * value_loss + self.cfg.entropy_coef * entropy_loss

        # Gradient step
        self.ppo_optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.ppo_optimizer)
        self.scaler.update()

        self.actor.clip_std(self.cfg.noise_range[0], self.cfg.noise_range[1])

        return {
            'kl_mean': kl_mean,
            'value_loss_default': value_loss_default.item(),
            'value_loss_contact': value_loss_contact.item(),
            'surrogate_loss': surrogate_loss.item(),
            'entropy_loss': -entropy_loss.item(),
        }

    def update_symmetry(self, cur_it, batch):
        """Update symmetry-related components"""
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            mixer_hidden_states = batch['mixer_hidden_states'] if self.actor.is_recurrent else None
            actor_hidden_states = batch['actor_hidden_states'] if self.actor.is_recurrent else None
            mask_batch = batch['masks'].squeeze() if self.actor.is_recurrent else slice(None)
            n_steps = mask_batch.size(0)

            # Forward pass to get original action mean
            self.actor.train_act(obs_batch, hidden_states=hidden_states_batch)
            action_mean_original = self.actor.action_mean.clone().detach()

            # Symmetry loss computation
            obs_mirrored_batch = obs_batch.flatten(0, 1).mirror().unflatten(0, (n_steps, -1))
            self.actor.train_act(obs_mirrored_batch, hidden_states=hidden_states_batch)

            mu_batch = obs_batch.mirror_dof_prop_by_x(action_mean_original.flatten(0, 1)).unflatten(0, (n_steps, -1))
            symmetry_loss = self.mse_loss(mu_batch[mask_batch], self.actor.action_mean[mask_batch])

        # Gradient step
        self.actor_optimizer.zero_grad()
        self.scaler.scale(symmetry_loss).backward()
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()

        return {
            'symmetry_loss': symmetry_loss.item(),
        }

    def update_estimation(
            self,
            cur_it,
            batch,
            target_std_vel=0.1,
            target_std_z=0.01,
    ):
        """Update estimation-related components (estimation, prediction, VAE, recon losses)"""
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs = batch['observations']
            critic_obs = batch['critic_observations']
            mixer_hidden_states = batch['mixer_hidden_states'] if self.actor.is_recurrent else None
            mask = batch['masks'] if self.actor.is_recurrent else slice(None)
            obs_next = batch['observations_next']

            # Forward pass
            vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, hmap = self.actor.estimate(
                obs.prop_his, obs.depth, mixer_hidden_states=mixer_hidden_states)

            # Estimation loss
            vel_est_loss = masked_MSE(vel, critic_obs.est_gt, mask)
            ot1_loss = masked_MSE(ot1, obs_next.proprio, mask)
            recon_loss = masked_L1(hmap, critic_obs.scan.flatten(2), mask)

            # KL loss
            vel_kl_loss = 1 + logvar_vel - mu_vel.pow(2) - logvar_vel.exp()
            vel_kl_loss = -0.5 * masked_mean(vel_kl_loss.sum(dim=2, keepdim=True), mask)
            z_kl_loss = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
            z_kl_loss = -0.5 * masked_mean(z_kl_loss.sum(dim=2, keepdim=True), mask)

            # Adaptive KL coefficient based on std_vel and std_z
            std_vel = logvar_vel.exp().sqrt().mean().item()
            std_z = logvar_z.exp().sqrt().mean().item()

            # Adaptive KL coefficient similar to learning rate adaptation
            if std_vel < target_std_vel:
                self.kl_coef_vel = min(1., self.kl_coef_vel * 1.1)
            else:
                self.kl_coef_vel = max(1e-7, self.kl_coef_vel / 1.1)

            if std_z < target_std_z:
                self.kl_coef_z = min(1., self.kl_coef_z * 1.1)
            else:
                self.kl_coef_z = max(1e-7, self.kl_coef_z / 1.1)

            kl_loss = self.kl_coef_vel * vel_kl_loss + self.kl_coef_z * z_kl_loss

            # Total estimation loss
            total_loss = vel_est_loss + ot1_loss + recon_loss + kl_loss

        # Gradient step
        self.actor_optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()

        return {
            'vel_est_loss': vel_est_loss.item(),
            'ot1_loss': ot1_loss.item(),
            'recon_loss': recon_loss.item(),
            'vel_kl_loss': vel_kl_loss.item(),
            'z_kl_loss': z_kl_loss.item(),
            'mu_vel': mu_vel.mean().item(),
            'mu_z': mu_z.mean().item(),
            'std_vel': std_vel,
            'std_z': std_z,
            'kl_coef_vel': self.kl_coef_vel,
            'kl_coef_z': self.kl_coef_z,
        }

    def reset(self, dones):
        if self.actor.is_recurrent and self.mixer_hidden_states is not None:
            self.mixer_hidden_states[:, dones] = 0.

    def play_act(self, obs, use_estimated_values=False, **kwargs):
        with torch.autocast(self.device.type, torch.float16, enabled=self.cfg.use_amp):

            if isinstance(use_estimated_values, bool):
                kwargs['use_estimated_values'] = use_estimated_values & torch.ones(self.task_cfg.env.num_envs, 1, dtype=torch.bool, device=self.device)
            else:
                kwargs['use_estimated_values'] = use_estimated_values

            actions, vel_est, self.mixer_hidden_states, self.actor_hidden_states = self.actor.act(
                proprio=obs.proprio.unsqueeze(0),
                prop_his=obs.prop_his.unsqueeze(0),
                depth=obs.depth.unsqueeze(0),
                mixer_hidden_states=self.mixer_hidden_states,
                actor_hidden_states=self.actor_hidden_states,
                **kwargs
            )
            return {"actions": actions, "vel_est": vel_est}

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        if load_optimizer:
            if 'ppo_optimizer_state_dict' in loaded_dict:
                self.ppo_optimizer.load_state_dict(loaded_dict['ppo_optimizer_state_dict'])
            if 'actor_optimizer_state_dict' in loaded_dict:
                self.actor_optimizer.load_state_dict(loaded_dict['actor_optimizer_state_dict'])

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, device=self.device)

        return loaded_dict['infos']

    def save(self):
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
        }
