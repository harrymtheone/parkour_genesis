import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence, Normal

from rsl_rl.modules.model_pie_priv import Policy
from rsl_rl.modules.utils import UniversalCritic
from rsl_rl.storage import RolloutStoragePerception as RolloutStorage
from .alg_base import BaseAlgorithm

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class Transition:
    def __init__(self, is_recurrent):
        self.observations = None
        self.critic_observations = None
        if is_recurrent:
            self.priv_hidden_states = None
            self.est_hidden_states = None
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


class PPO_PIE_Priv(BaseAlgorithm):
    def __init__(self, env_cfg, train_cfg, device=torch.device('cpu'), **kwargs):
        # PPO parameters
        self.cfg = train_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        # PPO components
        self.actor = Policy(env_cfg, train_cfg.policy).to(self.device)
        self.critic = UniversalCritic(env_cfg, train_cfg).to(self.device)
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)

        # reconstructor
        self.mse_loss = nn.MSELoss()

        # Rollout Storage
        self.transition = Transition(self.actor.is_recurrent)
        self.storage = RolloutStorage(env_cfg.num_envs, train_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, obs_critic, use_estimated_values=True, **kwargs):
        # store observations
        self.transition.observations = obs
        self.transition.critic_observations = obs_critic
        if self.actor.is_recurrent:
            self.transition.priv_hidden_states, self.transition.est_hidden_states = self.actor.get_hidden_states()

        actions = self.actor.act(obs, obs_critic, use_estimated_values=use_estimated_values)

        if self.actor.is_recurrent and self.transition.priv_hidden_states is None:
            # only for the first step where hidden_state is None
            hidden_states = self.actor.get_hidden_states()
            self.transition.priv_hidden_states = 0 * hidden_states[0]
            self.transition.est_hidden_states = 0 * hidden_states[1]

        # store
        self.transition.actions = actions
        self.transition.values = self.critic.evaluate(obs_critic).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        self.transition.use_estimated_values = use_estimated_values
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
        if self.actor.is_recurrent:
            self.actor.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, cur_it=0, **kwargs):
        update_est = cur_it % 5 == 0
        if update_est:
            print('updating estimation!')

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_dagger_loss = 0

        kl_change = []
        num_updates = 0

        if self.actor.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)

        for batch in generator:
            # update policy
            value_loss, surrogate_loss, entropy_loss, dagger_loss = self._update_policy(batch)
            loss = surrogate_loss + self.cfg.value_loss_coef * value_loss - entropy_loss + dagger_loss

            # Use KL to adaptively update learning rate
            if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
                with torch.no_grad():
                    kl_mean = kl_divergence(
                        Normal(batch['action_mean'], batch['action_sigma']),
                        Normal(self.actor.action_mean, self.actor.action_std)
                    )[batch['masks'].squeeze()].sum(dim=-1).mean().item()

                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            kl_change.append(kl_mean)
            num_updates += 1
            mean_value_loss += value_loss
            mean_surrogate_loss += surrogate_loss
            mean_entropy_loss += entropy_loss
            mean_dagger_loss += dagger_loss
            mean_kl += kl_mean

            loss_est = 0.
            if update_est:
                pass
                # self._update_estimation(batch)

            # Gradient step
            self.optimizer.zero_grad()
            self.scaler.scale(loss + loss_est).backward()
            # self.scaler.unscale_(self.optimizer)
            # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates
        mean_dagger_loss /= num_updates

        kl_str = 'kl: '
        for k in kl_change:
            kl_str += f'{k:.3f} | '
        print(kl_str)

        self.storage.clear()

        return {
            'Loss/learning_rate': self.learning_rate,
            'Loss/value_loss': mean_value_loss,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Loss/kl_div': mean_kl,
            'Loss/dagger_loss': mean_dagger_loss,
            'Train/noise_std': self.actor.log_std.exp().mean().item(),
        }

    # @torch.compile
    def _update_policy(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            priv_hidden_states_batch = batch['priv_hidden_states'] if self.actor.is_recurrent else None
            est_hidden_states_batch = batch['est_hidden_states'] if self.actor.is_recurrent else None
            mask_batch = batch['masks'].squeeze() if self.actor.is_recurrent else slice(None)
            actions_batch = batch['actions']
            target_values_batch = batch['values']
            advantages_batch = batch['advantages']
            returns_batch = batch['returns']
            old_actions_log_prob_batch = batch['actions_log_prob']
            use_estimated_values_batch = batch['use_estimated_values']

            priv_out, est_out = self.actor.train_act(
                obs_batch,
                critic_obs_batch,
                hidden_states=(priv_hidden_states_batch, est_hidden_states_batch),
                use_estimated_values=use_estimated_values_batch
            )

            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            evaluation = self.critic.evaluate(critic_obs_batch)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped)[mask_batch].mean()

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped = target_values_batch + (evaluation - target_values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses = (evaluation - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped)[mask_batch].mean()
            else:
                value_loss = (evaluation - returns_batch)[mask_batch].pow(2).mean()

            # Entropy Loss
            entropy_loss = self.cfg.entropy_coef * self.actor.entropy[mask_batch].mean()

            # DAgger loss
            priv_out = torch.where(use_estimated_values_batch, priv_out, priv_out.detach())
            est_out = torch.where(use_estimated_values_batch, est_out.detach(), est_out)
            dagger_loss = self.mse_loss(priv_out, est_out)

        return value_loss, surrogate_loss, entropy_loss, dagger_loss

    def _update_estimation(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            priv_hidden_states_batch = batch['priv_hidden_states'] if self.actor.is_recurrent else None
            est_hidden_states_batch = batch['est_hidden_states'] if self.actor.is_recurrent else None
            mask_batch = batch['masks'].squeeze() if self.actor.is_recurrent else slice(None)
            use_estimated_values_batch = batch['use_estimated_values']

            batch_size = 4
            mask_batch = mask_batch.clone()
            mask_batch[batch_size:] = False

            priv_out, est_out = self.actor.train_act(
                obs_batch[:batch_size],
                critic_obs_batch[:batch_size],
                hidden_states=(priv_hidden_states_batch, est_hidden_states_batch),
                use_estimated_values=use_estimated_values_batch[:batch_size]
            )

            # # ################################  Estimation Update ################################
            # # privileged information estimation loss
            # estimation_loss = self.mse_loss(est[mask_batch], obs_batch.priv_actor[mask_batch])
            #
            # # Ot+1 prediction and VAE loss
            # prediction_loss = self.mse_loss(ot1[mask_batch], obs_next_batch.proprio[mask_batch])
            # vae_loss = 1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp()
            # vae_loss = -0.5 * vae_loss[mask_batch].sum(dim=1).mean()
            #
            # # reconstructor loss
            # recon_loss = self.l1_loss(recon[mask_batch], obs_batch.scan.flatten(2)[mask_batch])
            #
            # # Symmetry loss
            # obs_mirrored_batch = obs_batch[:batch_size].flatten(0, 1).mirror().unflatten(0, (batch_size, -1))
            # self.actor.train_act(obs_mirrored_batch, hidden_states=hidden_states_batch)
            #
            # action_mean_original = action_mean_original[:batch_size]
            # mu_batch = obs_batch.mirror_dof_prop_by_x(action_mean_original.flatten(0, 1)).unflatten(0, (batch_size, -1))
            # symmetry_loss = 0.1 * self.mse_loss(mu_batch, self.actor.action_mean)

            return dagger_loss

    def play_act(self, obs, obs_critic=None, **kwargs):
        return self.actor.act(obs, obs_critic, **kwargs)

    def train(self):
        self.actor.train()
        self.critic.train()

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
