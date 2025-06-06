import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal

from rsl_rl.modules.model_bbm import RSSM, Actor
from rsl_rl.modules.utils import UniversalCritic
from rsl_rl.storage import RolloutStoragePerception as RolloutStorage
from .alg_base import BaseAlgorithm
from .utils import SymlogMSELoss, SymexpTwoHotLoss

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class Transition:
    def __init__(self):
        # actor
        self.observations = None
        self.critic_observations = None
        self.wm_feature = None
        # rssm
        self.state_deter = None
        self.state_stoch = None
        self.is_first_step = None
        # PPO
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
        for key in self.__dict__:
            setattr(self, key, None)


class PPO_BBM(BaseAlgorithm):
    def __init__(self, task_cfg, device, **kwargs):
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.symlog_mse = SymlogMSELoss()
        self.symexp_two_hot_loss = SymexpTwoHotLoss(low=-5., high=5.)

        # world model
        self.rssm = RSSM(task_cfg.env, task_cfg).to(self.device)
        self.optimizer_rssm = torch.optim.Adam(self.rssm.parameters(), lr=1e-4)
        self.scaler_rssm = GradScaler(enabled=self.cfg.use_amp)
        self.is_first_step = torch.ones(task_cfg.env.num_envs, dtype=torch.bool, device=self.device)

        # PPO components
        self.actor = Actor(task_cfg).to(self.device)
        self.actor.reset_std(self.cfg.init_noise_std, self.device)
        self.critic = UniversalCritic(task_cfg.env, task_cfg.policy).to(self.device)

        # storage and optimizer
        self.optimizer = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)
        self.transition = Transition()
        self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, obs_critic, step_world_model=True, **kwargs):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            self.transition.observations = obs
            self.transition.critic_observations = obs_critic
            self.transition.state_deter = self.rssm.get_deter()
            self.transition.state_stoch = self.rssm.get_stoch()
            self.transition.is_first_step = self.is_first_step

            wm_feature = self.rssm.step(obs.proprio, obs.depth, self.is_first_step)  # prev_actions is contained in obs.proprio
            self.transition.wm_feature = wm_feature

            actions = self.actor.act(wm_feature)

            # store
            self.transition.actions = actions
            self.transition.values = self.critic.evaluate(obs_critic)
            self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions)
            self.transition.action_mean = self.actor.action_mean
            self.transition.action_sigma = self.actor.action_std

        return actions

    def process_env_step(self, rewards, dones, infos, *args):
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
        self.is_first_step[:] = dones

    def compute_returns(self, last_critic_obs):
        last_values = self.critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, cur_it, **kwargs):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_dyn_loss = 0
        mean_rep_loss = 0
        mean_prop_loss = 0
        mean_depth_loss = 0
        mean_rew_loss = 0

        kl_change = []
        num_updates = 0
        for batch in self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):
            # update policy
            value_loss, surrogate_loss, entropy_loss, kl_mean = self._update_policy(batch)
            kl_change.append(kl_mean)
            num_updates += 1
            mean_value_loss += value_loss
            mean_surrogate_loss += surrogate_loss
            mean_entropy_loss += entropy_loss
            mean_kl += kl_mean

            # update RSSM
            loss_prop, loss_depth, loss_rew = self._update_rssm(batch)
            # mean_dyn_loss += dyn_loss
            # mean_rep_loss += rep_loss
            mean_prop_loss += loss_prop
            mean_depth_loss += loss_depth
            mean_rew_loss += loss_rew

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates

        mean_dyn_loss /= num_updates
        mean_rep_loss /= num_updates
        mean_prop_loss /= num_updates
        mean_depth_loss /= num_updates
        mean_rew_loss /= num_updates

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
            'Train/noise_std': self.actor.log_std.exp().mean().item(),
            'Loss/dyn_loss': mean_dyn_loss,
            'Loss/rep_loss': mean_rep_loss,
            'Loss/prop_loss': mean_prop_loss,
            'Loss/depth_loss': mean_depth_loss,
            'Loss/rew_loss': mean_rew_loss,
        }

    def _update_policy(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):

            # ############################ Dynamics ############################
            obs_batch = batch['observations']
            state_deter_batch = batch['state_deter']
            state_stoch_batch = batch['state_stoch']
            is_first_step_batch = batch['is_first_step']

            prop = obs_batch.proprio
            depth = obs_batch.depth

            with torch.no_grad():
                wm_feature = self.rssm.train_step(prop, depth, state_deter_batch, state_stoch_batch, is_first_step_batch)

            # ############################ PPO ############################
            critic_obs_batch = batch['critic_observations']
            actions_batch = batch['actions']
            target_values_batch = batch['values']
            advantages_batch = batch['advantages']
            returns_batch = batch['returns']
            old_mu_batch = batch['action_mean']
            old_sigma_batch = batch['action_sigma']
            old_actions_log_prob_batch = batch['actions_log_prob']

            self.actor.train_act(wm_feature.detach())

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
                value_clipped = target_values_batch + (evaluation - target_values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses = (evaluation - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (evaluation - returns_batch).pow(2).mean()

            # Entropy Loss
            entropy_loss = self.cfg.entropy_coef * self.actor.entropy.mean()

            loss = (surrogate_loss + self.cfg.value_loss_coef * value_loss - entropy_loss)

            # Use KL to adaptively update learning rate
            if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            # Gradient step
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return value_loss.item(), surrogate_loss.item(), entropy_loss.item(), kl_mean

    def _update_rssm(self, batch: dict, batch_size=32):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations'][:, :batch_size]
            state_deter_batch = batch['state_deter'][:, :batch_size]
            state_stoch_batch = batch['state_stoch'][:, :batch_size]
            is_first_step_batch = batch['is_first_step'][:, :batch_size]
            rewards_batch = batch['rewards'][:, :batch_size]

            prop = obs_batch.proprio
            depth = obs_batch.depth

            prop_pred, depth_pred, rew_pred = self.rssm.predict(
                prop, depth, state_deter_batch, state_stoch_batch, is_first_step_batch)

            # KL Divergence
            # loss, dyn_loss, rep_loss = self.compute_dyn_rep_loss(prior_digits, post_digits)

            # maximum likelihood
            prop_no_cmd = prop[:, :, :-self.rssm.num_actions].clone()
            prop_no_cmd[:, :, 6: 6 + 5] = 0.

            loss_prop = self.symlog_mse(prop_pred, prop_no_cmd)
            loss_depth = self.mse_loss(depth_pred, depth)
            loss_rew = self.symexp_two_hot_loss(rew_pred, rewards_batch)

            loss = 1.0 * loss_prop + 1.0 * loss_depth + 0.01 * loss_rew

        # Gradient step
        self.optimizer_rssm.zero_grad()
        self.scaler_rssm.scale(loss).backward()
        self.scaler_rssm.step(self.optimizer_rssm)
        self.scaler_rssm.update()

        # return dyn_loss.item(), rep_loss.item(), loss_prop.item(), loss_depth.item(), loss_rew.item()
        return loss_prop.item(), loss_depth.item(), loss_rew.item()

    def compute_dyn_rep_loss(self, prior, post, free_bits=1.0, dyn_scale=0.5, rep_scale=0.1):
        dist = self.rssm.get_dist

        dyn_loss = torch.distributions.kl.kl_divergence(
            dist(post.detach()), dist(prior))

        rep_loss = torch.distributions.kl.kl_divergence(
            dist(post), dist(prior.detach()))

        # this is implemented using maximum at the original repo as the gradients are not back-propagated for the out of limits.
        dyn_loss = torch.clip(dyn_loss, min=free_bits).mean()
        rep_loss = torch.clip(rep_loss, min=free_bits).mean()
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, dyn_loss, rep_loss

    def play_act(self, obs, step_world_model=True, **kwargs):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            assert 'dones' in kwargs
            dones = kwargs['dones']

            feature, predictions = self.rssm.play_step(obs.proprio, obs.depth, dones, sample=False)
            actions = self.actor.act(feature, **kwargs)

            predictions['actions'] = actions
            return predictions

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.rssm.load(loaded_dict['rssm_state_dict'])
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, device=self.device)

        return loaded_dict['infos']

    def save(self):
        return {
            'rssm_state_dict': self.rssm.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
