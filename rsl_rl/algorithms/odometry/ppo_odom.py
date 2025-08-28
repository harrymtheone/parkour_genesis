import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from rsl_rl.algorithms.alg_base import BaseAlgorithm
from rsl_rl.modules.odometer.actor import Actor
from rsl_rl.modules.utils import MixtureOfCritic
from rsl_rl.storage import RolloutStorageMixtureOfCritic as RolloutStorage

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


class PPO_Odom(BaseAlgorithm):
    def __init__(self, task_cfg, device=torch.device('cpu'), env=None, **kwargs):
        self.env = env

        # PPO parameters
        self.task_cfg = task_cfg
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        self.cur_it = 0

        # PPO components
        self.actor = Actor(task_cfg).to(self.device)
        self.actor.reset_std(self.cfg.init_noise_std)

        self.critic = MixtureOfCritic(task_cfg).to(self.device)
        self.optimizer = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)

        # Rollout Storage
        self.transition = Transition()
        self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)

        self.mse_loss = nn.MSELoss()

    def act(self,
            obs,
            obs_critic,
            use_estimated_values: torch.Tensor = None,
            recon: torch.Tensor = None,
            **kwargs):
        # act function should run within torch.inference_mode context
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            # store observations
            self.transition.observations = obs.no_depth()
            self.transition.critic_observations = obs_critic
            self.transition.actor_hidden_states = self.actor.get_hidden_states()

            self.transition.recon = recon
            actions = self.actor.act(obs, recon, use_estimated_values=use_estimated_values)

            if self.transition.actor_hidden_states is None:
                # only for the first step where hidden_state is None
                self.transition.actor_hidden_states = 0 * self.actor.get_hidden_states()

            # store
            self.transition.actions = actions
            self.transition.values = self.critic.evaluate(obs_critic)
            self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions)
            self.transition.action_mean = self.actor.action_mean
            self.transition.action_sigma = self.actor.action_std
            self.transition.use_estimated_values = use_estimated_values
            return actions

    def process_env_step(self, rewards, dones, infos, *args):
        self.transition.dones = dones.unsqueeze(1)
        self.transition.rewards = {rew_name: rew.clone().unsqueeze(1) for rew_name, rew in infos['step_rew'].items()}

        # Bootstrapping on time-outs
        if 'time_outs' in infos:
            if 'reach_goals' in infos:
                bootstrapping = (infos['time_outs'] | infos['reach_goals']).unsqueeze(1).to(self.device)
            else:
                bootstrapping = (infos['time_outs']).unsqueeze(1).to(self.device)

            for rew_name in self.transition.rewards:
                self.transition.rewards[rew_name] += self.cfg.gamma * self.transition.values[rew_name] * bootstrapping

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.critic.evaluate(last_critic_obs)
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, cur_it=0, **kwargs):
        self.cur_it = cur_it
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_symmetry_loss = 0

        kl_change = []
        num_updates = 0

        if self.actor.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)

        for batch in generator:
            # ########################## policy loss ##########################
            kl_mean, value_loss, surrogate_loss, entropy_loss, symmetry_loss = self._compute_policy_loss(batch)
            loss = surrogate_loss + self.cfg.value_loss_coef * value_loss - entropy_loss + symmetry_loss

            num_updates += 1
            # policy statistics
            kl_change.append(kl_mean.item())
            mean_kl += kl_mean.item()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_loss.item()
            mean_symmetry_loss += symmetry_loss.item()

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
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_symmetry_loss /= num_updates

        kl_str = 'kl: '
        for k in kl_change:
            kl_str += f'{k:.3f} | '
        print(kl_str)

        self.storage.clear()

        return {
            'Loss/learning_rate': self.learning_rate,
            'Loss/kl_div': mean_kl,
            'Loss/value_loss': mean_value_loss,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Loss/symmetry_loss': mean_symmetry_loss,
            'Train/noise_std': self.actor.log_std.exp().mean().item(),
        }

    def _compute_policy_loss(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            actor_hidden_states_batch = batch['actor_hidden_states'] if self.actor.is_recurrent else None
            recon_batch = batch['recon'] if 'recon' in batch else None
            mask_batch = batch['masks'] if self.actor.is_recurrent else slice(None)
            actions_batch = batch['actions']
            target_values_batch = batch['values']
            advantages_batch = batch['advantages']
            returns_batch = batch['returns']
            old_actions_log_prob_batch = batch['actions_log_prob']
            use_estimated_values_batch = batch['use_estimated_values']

            self.actor.train_act(
                obs_batch,
                recon_batch,
                hidden_states=actor_hidden_states_batch,
                use_estimated_values=use_estimated_values_batch
            )

            with torch.no_grad():
                kl_mean = kl_divergence(
                    Normal(batch['action_mean'], batch['action_sigma']),
                    Normal(self.actor.action_mean, self.actor.action_std)
                )
                if self.actor.is_recurrent:
                    kl_mean = kl_mean.sum(dim=2, keepdim=True)
                    kl_mean = masked_mean(kl_mean, mask_batch)
                else:
                    kl_mean = kl_mean.sum(dim=1).mean()

            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            evaluation = self.critic.evaluate(critic_obs_batch)
            evaluation = torch.cat(tuple(evaluation.values()), dim=2 if self.actor.is_recurrent else 1)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            if self.actor.is_recurrent:
                surrogate_loss = masked_mean(torch.maximum(surrogate, surrogate_clipped), mask_batch)
            else:
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped = target_values_batch + (evaluation - target_values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses = (evaluation - returns_batch).square()
                value_losses_clipped = (value_clipped - returns_batch).square()
                if self.actor.is_recurrent:
                    value_loss = masked_mean(torch.maximum(value_losses, value_losses_clipped), mask_batch)
                else:
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                if self.actor.is_recurrent:
                    value_loss = masked_MSE(evaluation, returns_batch, mask_batch)
                else:
                    value_loss = self.mse_loss(evaluation, returns_batch)

            # Entropy loss
            if self.actor.is_recurrent:
                entropy_loss = self.cfg.entropy_coef * masked_mean(self.actor.entropy, mask_batch)
            else:
                entropy_loss = self.cfg.entropy_coef * self.actor.entropy.mean()

            # Symmetry loss
            batch_size = 4
            action_mean_original = self.actor.action_mean[:batch_size].detach()

            if hasattr(obs_batch, 'mirror'):
                obs_mirrored_batch = obs_batch[:batch_size].flatten(0, 1).mirror().unflatten(0, (batch_size, -1))
                if recon_batch is not None:
                    recon_batch = recon_batch[:batch_size]
                    use_estimated_values_batch = use_estimated_values_batch[:batch_size]

                self.actor.train_act(
                    obs_mirrored_batch,
                    recon_batch,
                    hidden_states=actor_hidden_states_batch,
                    use_estimated_values=use_estimated_values_batch
                )

                mu_batch = obs_batch.mirror_dof_prop_by_x(action_mean_original.flatten(0, 1)).unflatten(0, (batch_size, -1))
                symmetry_loss = 0.1 * self.mse_loss(mu_batch, self.actor.action_mean)
            else:
                symmetry_loss = torch.zeros_like(surrogate_loss)

            return kl_mean, value_loss, surrogate_loss, entropy_loss, symmetry_loss

    def play_act(self, obs, use_estimated_values=True, recon=None, **kwargs):
        with torch.autocast(self.device.type, torch.float16, enabled=self.cfg.use_amp):

            if isinstance(use_estimated_values, bool):
                kwargs['use_estimated_values'] = use_estimated_values & torch.ones(self.task_cfg.env.num_envs, 1, dtype=torch.bool, device=self.device)
            else:
                kwargs['use_estimated_values'] = use_estimated_values

            return {'actions': self.actor.act(obs, recon, **kwargs)}

    def reset(self, dones):
        self.actor.reset(dones)

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.critic.load(loaded_dict['critic_state_dict'])

        if load_optimizer:
            try:
                self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            except Exception as e:
                print(f"Failed to load optimizer state_dict: {e}")
                print("Continuing with fresh optimizer state...")

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std)
        
        return loaded_dict.get('infos', {})

    def save(self):
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
