import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence, Normal

from rsl_rl.algorithms import BaseAlgorithm
from rsl_rl.modules.model_scan import Actor, Critic
from rsl_rl.storage import RolloutStorage


class Transition:
    def __init__(self):
        self.observations = None
        self.critic_observations = None
        # self.hidden_states = None  # dynamically set by setattr()
        self.actions = None
        self.rewards = None
        self.dones = None
        self.values = None
        self.actions_log_prob = None
        self.action_mean = None
        self.action_sigma = None

    def get_items(self):
        return self.__dict__.items()

    def clear(self):
        self.__init__()


class PPO(BaseAlgorithm):
    def __init__(self, env_cfg, train_cfg, device=torch.device('cpu')):
        # PPO parameters
        self.cfg = train_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        # PPO components
        self.actor = Actor(env_cfg, train_cfg.policy).to(self.device)
        self.critic = Critic(env_cfg, train_cfg).to(self.device)
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.transition = Transition()
        self.storage = RolloutStorage(env_cfg.num_envs, train_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, critic_obs, **kwargs):
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.actions = self.actor.act(obs).detach()

        self.transition.values = self.critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, *args):
        rewards_total = rewards.clone()

        self.transition.rewards = rewards_total.clone().unsqueeze(1)
        self.transition.dones = dones.float().unsqueeze(1)

        # Bootstrapping on time-outs
        if 'time_outs' in infos:
            bootstrapping = torch.logical_or(infos['time_outs'], infos['reach_goals']).unsqueeze(1)
            self.transition.rewards += self.cfg.gamma * self.transition.values * bootstrapping.to(self.device)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor.reset(dones)

        return rewards_total

    def compute_returns(self, last_critic_obs):
        last_values = self.critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, **kwargs):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0

        kl_change = []
        num_updates = 0

        for batch in self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            actions_batch = batch['actions']
            target_values_batch = batch['values']
            advantages_batch = batch['advantages']
            returns_batch = batch['returns']
            old_actions_log_prob_batch = batch['actions_log_prob']
            old_mu_batch = batch['action_mean']
            old_sigma_batch = batch['action_sigma']

            self.actor.act(obs_batch)  # match distribution dimension

            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            value_batch = self.critic.evaluate(critic_obs_batch)

            # KL
            if self.cfg.desired_kl is not None and self.cfg.schedule == 'adaptive':
                with torch.no_grad():
                    kl_mean = kl_divergence(
                        Normal(old_mu_batch, old_sigma_batch),
                        Normal(self.actor.action_mean, self.actor.action_std)
                    ).sum(dim=1).mean().item()

                    kl_change.append(kl_mean)

                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Entropy loss
            entropy_loss = self.cfg.entropy_coef * self.actor.entropy.mean()

            loss = (surrogate_loss
                    + self.cfg.value_loss_coef * value_loss
                    - entropy_loss)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.cfg.max_grad_norm)
            self.optimizer.step()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_loss
            mean_kl += kl_mean

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates

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
            'Policy/noise_std': self.actor.log_std.exp().mean().item(),
        }

    def play_act(self, obs, **kwargs):
        return self.actor.act(obs.float(), eval_=True)

    def train(self):
        self.actor.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, device=self.device)

        return loaded_dict['infos']

    def save(self):
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
