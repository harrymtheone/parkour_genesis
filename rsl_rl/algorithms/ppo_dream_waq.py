import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence, Normal

from rsl_rl.modules.model_dreamwaq import Actor, Critic
from rsl_rl.storage import RolloutStorage
from .alg_base import BaseAlgorithm

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class Transition:
    def __init__(self):
        self.observations = None
        self.critic_observations = None
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
        self.__init__()


class PPODreamWaQ(BaseAlgorithm):
    def __init__(self, env_cfg, train_cfg, device=torch.device('cpu')):
        # PPO parameters
        self.cfg = train_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        # PPO components
        self.actor = Actor(env_cfg, train_cfg.policy).to(self.device)
        self.critic = Critic(env_cfg, train_cfg.policy).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.scaler_actor = GradScaler(enabled=self.cfg.use_amp)
        self.scaler_critic = GradScaler(enabled=self.cfg.use_amp)

        # reconstructor
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Rollout Storage
        self.transition = Transition()
        self.storage = RolloutStorage(env_cfg.num_envs, train_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, obs_critic, use_estimated_values=None, **kwargs):
        # store observations
        self.transition.observations = obs
        self.transition.critic_observations = obs_critic

        actions = self.actor.act(obs, use_estimated_values=use_estimated_values).detach()

        # store
        self.transition.actions = actions
        self.transition.values = self.critic.evaluate(obs_critic).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        self.transition.use_estimated_values = use_estimated_values

        self.actor.detach_hidden_state()
        return actions

    def process_env_step(self, rewards, dones, infos, *args):
        self.transition.rewards = rewards.clone().unsqueeze(1)
        self.transition.dones = dones.unsqueeze(1)

        # Bootstrapping on time-outs
        if 'time_outs' in infos:
            bootstrapping = (infos['reach_goals'] | infos['time_outs']).unsqueeze(1).to(self.device)
            self.transition.rewards += self.cfg.gamma * self.transition.values * bootstrapping

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, update_est=True, **kwargs):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_symmetry_loss = 0

        kl_change = []
        num_updates = 0

        for batch in self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):
            kl_mean, surrogate_loss, value_loss, entropy_loss, symmetry_loss = self._compute_loss(batch)

            kl_change.append(kl_mean)

            # Use KL to adaptively update learning rate
            if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
                for param_group in self.optimizer_actor.param_groups:
                    param_group['lr'] = self.learning_rate

            # Gradient step
            self.optimizer_actor.zero_grad()
            self.scaler_actor.scale(surrogate_loss - entropy_loss).backward()
            self.scaler_actor.step(self.optimizer_actor)
            self.scaler_actor.update()

            self.optimizer_critic.zero_grad()
            self.scaler_critic.scale(self.cfg.value_loss_coef * value_loss).backward()
            self.scaler_critic.step(self.optimizer_critic)
            self.scaler_critic.update()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_loss.item()
            mean_kl += kl_mean
            if update_est:
                mean_symmetry_loss += symmetry_loss

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates
        mean_symmetry_loss /= num_updates

        kl_str = 'kl: '
        for k in kl_change:
            kl_str += f'{k:.3f} | '
        print(kl_str)

        self.storage.clear()

        return {
            'Loss/learning_rate': self.learning_rate,
            'Loss/value_loss': mean_value_loss,
            'Loss/kl_div': mean_kl,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Loss/symmetry_loss': mean_symmetry_loss,
            'Policy/noise_std': self.actor.log_std.exp().mean().item(),
        }

    # @torch.compile()
    def _compute_loss(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            actions_batch = batch['actions']
            values_batch = batch['values']
            advantages_batch = batch['advantages']
            returns_batch = batch['returns']
            old_mu_batch = batch['action_mean']
            old_sigma_batch = batch['action_sigma']
            old_actions_log_prob_batch = batch['actions_log_prob']
            use_estimated_values_batch = batch['use_estimated_values']

            self.actor.act(obs_batch, use_estimated_values=use_estimated_values_batch)

            # Use KL to adaptively update learning rate
            if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
                with torch.no_grad():
                    kl_mean = kl_divergence(
                        Normal(old_mu_batch, old_sigma_batch),
                        Normal(self.actor.action_mean, self.actor.action_std)
                    ).sum(dim=-1).mean().item()

                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            evaluation = self.critic.evaluate(critic_obs_batch)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_losses = (evaluation - returns_batch).pow(2)
                evaluation_clipped = values_batch + (evaluation - values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses_clipped = (evaluation_clipped - returns_batch).pow(2)
                value_loss = torch.maximum(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (evaluation - returns_batch).pow(2).mean()

            entropy_loss = self.cfg.entropy_coef * self.actor.entropy.mean()

            return kl_mean, surrogate_loss, value_loss, entropy_loss, 0.

    def play_act(self, obs, use_estimated_values=True):
        return self.actor.act(obs.float(), use_estimated_values=use_estimated_values, eval_=True)

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        if load_optimizer:
            self.optimizer_actor.load_state_dict(loaded_dict['optimizer_actor_state_dict'])
            self.optimizer_critic.load_state_dict(loaded_dict['optimizer_critic_state_dict'])

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, device=self.device)

        return loaded_dict['infos']

    def save(self):
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
        }
