import torch
import torch.optim as optim
from torch import nn
from torch.distributions import kl_divergence, Normal

from rsl_rl.algorithms.alg_base import BaseAlgorithm
from rsl_rl.algorithms.template_models import UniversalCritic
from rsl_rl.algorithms.utils import masked_mean, masked_MSE
from rsl_rl.storage import RolloutStoragePerception as RolloutStorage
from .networks import Actor, VAE

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class Transition:
    def __init__(self):
        self.observations = None
        self.critic_observations = None
        self.vae_hidden_states = None
        self.est_vel = None
        self.est_z = None

        self.observations_next = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.values = None
        self.actions_log_prob = None
        self.action_mean = None
        self.action_sigma = None
        self.use_estimated_values = None

    def get_items(self):
        return self.__dict__.items()

    def clear(self):
        for key in self.__dict__:
            setattr(self, key, None)


class PPODreamWaQ(BaseAlgorithm):
    def __init__(self, task_cfg, device=torch.device('cpu'), **kwargs):
        # PPO parameters
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        # PPO components
        self.vae = VAE(task_cfg.env, task_cfg.policy).to(self.device)
        self.actor = Actor(task_cfg.env, task_cfg.policy).to(self.device)
        self.actor.reset_std(self.cfg.init_noise_std, device=self.device)
        self.critic = UniversalCritic(task_cfg.env, task_cfg.policy).to(self.device)

        self.optimizer_ppo = optim.Adam([
            *self.actor.parameters(), *self.critic.parameters()
        ], lr=self.learning_rate)

        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        self.scaler_vae = GradScaler(enabled=self.cfg.use_amp)

        # Initialize adaptive KL coefficient
        self.kl_coef_vel = 0.01
        self.kl_coef_z = 0.01

        # Rollout Storage
        self.transition = Transition()
        self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, obs_critic, use_estimated_values=True, **kwargs):
        # store observations
        self.transition.observations = obs
        self.transition.critic_observations = obs_critic
        self.transition.vae_hidden_states = self.vae.get_hidden_states()

        vel, z = self.vae(obs.proprio.unsqueeze(0))[:2]
        vel, z = vel.squeeze(0), z.squeeze(0)

        actions = self.actor.act(obs.proprio, vel, z)

        if self.transition.vae_hidden_states is None:
            # only for the first step where hidden_state is None
            self.transition.vae_hidden_states = torch.zeros_like(self.vae.get_hidden_states())

        self.transition.est_vel = vel
        self.transition.est_z = z

        # store
        self.transition.actions = actions
        self.transition.values = self.critic.evaluate(obs_critic).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        self.transition.use_estimated_values = use_estimated_values
        return actions

    def process_env_step(self, rewards, dones, infos, *args):
        self.transition.observations_next = args[0].as_obs_next()
        self.transition.rewards = rewards.clone().unsqueeze(1)
        self.transition.dones = dones.unsqueeze(1)

        # Bootstrapping on time-outs
        if 'time_outs' in infos:
            if 'reach_goals' in infos:
                bootstrapping = (infos['time_outs'] | infos['reach_goals']).unsqueeze(1).to(self.device)
            else:
                bootstrapping = (infos['time_outs']).unsqueeze(1).to(self.device)

            self.transition.rewards += self.cfg.gamma * self.transition.values * bootstrapping

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.vae.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, cur_it, **kwargs):
        update_est = cur_it % 3 == 0

        # ---- PPO ----
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        # ---- VAE ----
        mean_vel_est_loss = 0
        mean_ot1_est_loss = 0
        # ---- VAE ----
        mean_vel_kl_loss = 0
        mean_z_kl_loss = 0
        mean_abs_vel = 0
        mean_abs_z = 0
        mean_std_vel = 0
        mean_std_z = 0
        mean_snr_vel = 0
        mean_snr_z = 0
        mean_kl_coef_vel = 0
        mean_kl_coef_z = 0

        kl_change = []
        num_updates = 0

        for batch in self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):
            num_updates += 1

            # ########################## policy loss ##########################
            ppo_metrics = self.update_ppo(batch)
            mean_kl += ppo_metrics['kl_mean']
            mean_value_loss += ppo_metrics['value_loss']
            mean_surrogate_loss += ppo_metrics['surrogate_loss']
            mean_entropy_loss += ppo_metrics['entropy_loss']
            kl_change.append(ppo_metrics['kl_mean'])

            # update estimation
            if update_est:
                est_metrics = self.update_estimation(batch)
                mean_vel_est_loss += est_metrics['vel_est_loss']
                mean_ot1_est_loss += est_metrics['ot1_loss']
                mean_vel_kl_loss += est_metrics['vel_kl_loss']
                mean_z_kl_loss += est_metrics['z_kl_loss']
                mean_abs_vel += est_metrics['abs_vel']
                mean_abs_z += est_metrics['abs_z']
                mean_std_vel += est_metrics['std_vel']
                mean_std_z += est_metrics['std_z']
                mean_snr_vel += est_metrics['snr_vel']
                mean_snr_z += est_metrics['snr_z']
                mean_kl_coef_vel += est_metrics['kl_coef_vel']
                mean_kl_coef_z += est_metrics['kl_coef_z']

        # ---- PPO ----
        mean_kl /= num_updates
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        # ---- VAE ----
        mean_vel_est_loss /= num_updates
        mean_ot1_est_loss /= num_updates
        mean_vel_kl_loss /= num_updates
        mean_z_kl_loss /= num_updates
        mean_abs_vel /= num_updates
        mean_abs_z /= num_updates
        mean_std_vel /= num_updates
        mean_std_z /= num_updates
        mean_snr_vel /= num_updates
        mean_snr_z /= num_updates
        mean_kl_coef_vel /= num_updates
        mean_kl_coef_z /= num_updates

        kl_str = 'kl: '
        for k in kl_change:
            kl_str += f'{k:.3f} | '
        print(kl_str)

        self.storage.clear()

        metrics = {
            'Loss/learning_rate': self.learning_rate,
            'Loss/value_loss': mean_value_loss,
            'Loss/kl_div': mean_kl,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Train/noise_std': self.actor.log_std.exp().mean().item(),
        }

        if update_est:
            metrics.update({
                'VAE/vel_est_loss': mean_vel_est_loss,
                'VAE/Ot+1_loss': mean_ot1_est_loss,
                'VAE_KL/vel_kl_loss': mean_vel_kl_loss,
                'VAE_KL/z_kl_loss': mean_z_kl_loss,
                'VAE_KL/abs_vel': mean_abs_vel,
                'VAE_KL/abs_z': mean_abs_z,
                'VAE_KL/std_vel': mean_std_vel,
                'VAE_KL/std_z': mean_std_z,
                'VAE_KL/SNR_vel': mean_snr_vel,
                'VAE_KL/SNR_z': mean_snr_z,
                'VAE_KL/kl_coef_vel': mean_kl_coef_vel,
                'VAE_KL/kl_coef_z': mean_kl_coef_z,
            })

        return metrics

    def update_ppo(self, batch: dict):
        obs = batch['observations']
        critic_obs = batch['critic_observations']
        est_vel = batch['est_vel']
        est_z = batch['est_z']
        mask = batch['masks']
        actions = batch['actions']
        values = batch['values']
        advantages = batch['advantages']
        returns = batch['returns']
        old_actions_log_prob = batch['actions_log_prob']

        # Forward pass
        self.actor.act(obs.proprio, est_vel, est_z)

        actions_log_prob = self.actor.get_actions_log_prob(actions)
        evaluations = self.critic.evaluate(critic_obs)

        # Surrogate loss
        ratio = torch.exp(actions_log_prob - old_actions_log_prob)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
        surrogate_loss = masked_mean(torch.maximum(surrogate, surrogate_clipped), mask)

        # Value function loss
        if self.cfg.use_clipped_value_loss:
            value_clipped = values + (
                    evaluations - values).clamp(-self.cfg.clip_param, self.cfg.clip_param)
            value_loss = (evaluations - returns).square()
            value_loss_clipped_default = (value_clipped - returns).square()
            value_loss = masked_mean(torch.maximum(value_loss, value_loss_clipped_default), mask)
        else:
            value_loss = masked_MSE(evaluations, returns, mask)

        # Entropy loss
        entropy_loss = -masked_mean(self.actor.entropy, mask)

        # Total PPO loss
        total_loss = surrogate_loss + self.cfg.value_loss_coef * value_loss + self.cfg.entropy_coef * entropy_loss

        # Use KL to adaptively update learning rate
        with torch.no_grad():
            kl_mean = kl_divergence(
                Normal(batch['action_mean'], batch['action_sigma']),
                Normal(self.actor.action_mean, self.actor.action_std)
            ).sum(dim=2, keepdim=True)
            kl_mean = masked_mean(kl_mean, mask).item()

        if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
            if kl_mean > self.cfg.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                self.learning_rate = min(1e-3, self.learning_rate * 1.5)

            for param_group in self.optimizer_ppo.param_groups:
                param_group['lr'] = self.learning_rate

        # Gradient step
        self.optimizer_ppo.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            [*self.actor.actor_backbone.parameters(), *self.critic.parameters()],
            self.cfg.max_grad_norm
        )
        self.optimizer_ppo.step()

        self.actor.clip_std(self.cfg.noise_range[0], self.cfg.noise_range[1])

        return {
            'kl_mean': kl_mean,
            'value_loss': value_loss.item(),
            'surrogate_loss': surrogate_loss.item(),
            'entropy_loss': -entropy_loss.item(),
        }

    def update_estimation(
            self,
            batch,
            target_snr_vel=2.0,  # Target SNR for velocity (mean/std ratio)
            target_snr_z=2.0,  # Target SNR for z (mean/std ratio)
    ):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs = batch['observations']
            critic_obs = batch['critic_observations']
            vae_hidden_states = batch['vae_hidden_states']
            masks = batch['masks']
            obs_next = batch['observations_next']

            vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, _ = self.vae(obs.proprio, vae_hidden_states)

            # Estimation loss
            vel_est_loss = masked_MSE(vel, critic_obs.est_gt, masks)
            ot1_loss = masked_MSE(ot1, obs_next.proprio, masks)

            # KL loss
            vel_kl_loss = 1 + logvar_vel - mu_vel.pow(2) - logvar_vel.exp()
            vel_kl_loss = -0.5 * masked_mean(vel_kl_loss.sum(dim=2, keepdim=True), masks)
            z_kl_loss = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
            z_kl_loss = -0.5 * masked_mean(z_kl_loss.sum(dim=2, keepdim=True), masks)

            # Adaptive KL coefficient based on SNR (Signal-to-Noise Ratio)
            std_vel = logvar_vel.exp().sqrt().mean().item()
            std_z = logvar_z.exp().sqrt().mean().item()
            mean_abs_vel = mu_vel.abs().mean().item()  # Absolute mean for SNR calculation
            mean_abs_z = mu_z.abs().mean().item()  # Absolute mean for SNR calculation

            # Calculate SNR = |mean| / std (avoid division by zero)
            snr_vel = mean_abs_vel / (std_vel + 1e-8)
            snr_z = mean_abs_z / (std_z + 1e-8)

        # Adaptive KL coefficient based on SNR
        if snr_vel < target_snr_vel:
            self.kl_coef_vel = max(1e-7, self.kl_coef_vel / 1.1)
        else:
            self.kl_coef_vel = max(1e-7, self.kl_coef_vel * 1.1)

        if snr_z < target_snr_z:
            self.kl_coef_z = max(1e-7, self.kl_coef_z / 1.1)
        else:
            self.kl_coef_z = max(1e-7, self.kl_coef_z * 1.1)

        kl_loss = self.kl_coef_vel * vel_kl_loss + self.kl_coef_z * z_kl_loss

        # Total estimation loss
        total_loss = vel_est_loss + ot1_loss + kl_loss

        # Gradient step
        self.optimizer_vae.zero_grad()
        self.scaler_vae.scale(total_loss).backward()
        self.scaler_vae.step(self.optimizer_vae)
        self.scaler_vae.update()

        return {
            'vel_est_loss': vel_est_loss.item(),
            'ot1_loss': ot1_loss.item(),
            'vel_kl_loss': vel_kl_loss.item(),
            'z_kl_loss': z_kl_loss.item(),
            'abs_vel': mean_abs_vel,
            'abs_z': mean_abs_z,
            'std_vel': std_vel,
            'std_z': std_z,
            'snr_vel': snr_vel,  # Signal-to-Noise Ratio for velocity
            'snr_z': snr_z,  # Signal-to-Noise Ratio for z
            'kl_coef_vel': self.kl_coef_vel,
            'kl_coef_z': self.kl_coef_z,
        }

    def play_act(self, obs, **kwargs):
        vel, z = self.vae(obs.proprio.unsqueeze(0), sample=False)[:2]
        vel, z = vel.squeeze(0), z.squeeze(0)

        return {'actions': self.actor.act(obs.proprio, vel, z, **kwargs)}

    def reset(self, dones):
        self.vae.reset(dones)

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.vae.load_state_dict(loaded_dict['vae'])
        self.actor.load_state_dict(loaded_dict['actor'])
        self.critic.load_state_dict(loaded_dict['critic'])

        if load_optimizer:
            self.optimizer_ppo.load_state_dict(loaded_dict['optimizer_ppo'])
            self.optimizer_vae.load_state_dict(loaded_dict['optimizer_vae'])

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, device=self.device)

        return loaded_dict['infos']

    def save(self):
        return {
            'vae': self.vae.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_ppo': self.optimizer_ppo.state_dict(),
            'optimizer_vae': self.optimizer_vae.state_dict(),
        }
