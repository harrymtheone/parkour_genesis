import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence, Normal

from rsl_rl.algorithms.dreamwaq.model_dreamwaq import ActorGRU, Critic
from rsl_rl.storage import RolloutStoragePerception as RolloutStorage
from .alg_base import BaseAlgorithm
from ..modules.model_phase import Modulator

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class Transition:
    def __init__(self):
        self.observations = None
        self.critic_observations = None
        self.hidden_states = None
        self.modulator_hidden_states = None
        self.observations_next = None
        self.action_clock = None
        self.rewards = None
        self.dones = None
        self.values = None
        self.action_clock_log_prob = None
        self.action_clock_mean = None
        self.action_clock_sigma = None
        self.use_estimated_values = None

    def get_items(self):
        return self.__dict__.items()

    def clear(self):
        for key in self.__dict__:
            setattr(self, key, None)


class PPO_Phase(BaseAlgorithm):
    def __init__(self, task_cfg, device=torch.device('cpu'), **kwargs):
        # PPO parameters
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        # PPO components
        blind_actor = ActorGRU(task_cfg.env, task_cfg.policy).to(self.device)
        blind_actor.reset_std(self.cfg.init_noise_std, device=self.device)
        self.actor = Modulator(task_cfg.env, task_cfg.policy, blind_actor).to(self.device)

        self.critic = Critic(task_cfg.env, task_cfg.policy).to(self.device)
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)

        # reconstructor
        self.mse_loss = nn.MSELoss()

        # Rollout Storage
        self.transition = Transition()
        self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, obs_critic, use_estimated_values=True, **kwargs):
        # store observations
        self.transition.observations = obs
        self.transition.critic_observations = obs_critic

        hidden = self.actor.get_hidden_states()
        self.transition.hidden_states = hidden[0]
        self.transition.modulator_hidden_states = hidden[1]

        actions, clock = self.actor.act(obs, use_estimated_values=use_estimated_values)
        action_clock = torch.cat([actions, clock], dim=1)

        if self.transition.hidden_states is None:
            # only for the first step where hidden_state is None
            hidden = self.actor.get_hidden_states()
            self.transition.hidden_states = 0 * hidden[0]
            self.transition.modulator_hidden_states = 0 * hidden[1]

        # store
        self.transition.action_clock = action_clock
        self.transition.values = self.critic.evaluate(obs_critic)
        self.transition.action_clock_log_prob = self.actor.get_log_prob(action_clock)
        self.transition.action_clock_mean = self.actor.action_mean
        self.transition.action_clock_sigma = self.actor.action_std
        self.transition.use_estimated_values = use_estimated_values
        return action_clock

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
        if self.actor.is_recurrent:
            self.actor.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, update_est=True, **kwargs):
        update_est = False

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        # mean_symmetry_loss = 0
        mean_estimation_loss = 0
        mean_ot1_prediction_loss = 0
        mean_vae_loss = 0

        kl_change = []
        num_updates = 0

        if self.actor.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)

        for batch in generator:
            # update policy
            value_loss, surrogate_loss, entropy_loss, kl_mean = self._update_policy(batch)
            kl_change.append(kl_mean)
            num_updates += 1
            mean_value_loss += value_loss
            mean_surrogate_loss += surrogate_loss
            mean_entropy_loss += entropy_loss
            mean_kl += kl_mean

            # update estimation
            if update_est:
                estimation_loss, ot1_prediction_loss, vae_loss = self._update_estimation(batch)
                mean_estimation_loss += estimation_loss
                mean_ot1_prediction_loss += ot1_prediction_loss
                mean_vae_loss += vae_loss

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates
        mean_estimation_loss /= num_updates
        mean_ot1_prediction_loss /= num_updates
        mean_vae_loss /= num_updates
        # mean_symmetry_loss /= num_updates

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
            # 'Loss/symmetry_loss': mean_symmetry_loss,
            'Loss/estimation_loss': mean_estimation_loss,
            'Loss/ot1_prediction_loss': mean_ot1_prediction_loss,
            'Loss/vae_loss': mean_vae_loss,
            'Train/noise_std_action': self.actor.log_std_action.exp().mean().item(),
            'Train/noise_std_clock': self.actor.log_std_clock.exp().mean().item(),
        }

    def _update_policy(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            hidden_states_batch = batch['hidden_states']
            modulator_hidden_states_batch = batch['modulator_hidden_states']
            mask_batch = batch['masks'].squeeze() if self.actor.is_recurrent else slice(None)
            actions_batch = batch['action_clock']
            target_values_batch = batch['values']
            advantages_batch = batch['advantages']
            returns_batch = batch['returns']
            old_mu_batch = batch['action_clock_mean']
            old_sigma_batch = batch['action_clock_sigma']
            old_actions_log_prob_batch = batch['action_clock_log_prob']
            use_estimated_values_batch = batch['use_estimated_values']

            self.actor.train_act(
                obs_batch,
                (hidden_states_batch, modulator_hidden_states_batch),
                use_estimated_values=use_estimated_values_batch
            )

            # Use KL to adaptively update learning rate
            if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
                with torch.no_grad():
                    kl_mean = kl_divergence(
                        Normal(old_mu_batch, old_sigma_batch),
                        Normal(self.actor.action_mean, self.actor.action_std)
                    )[mask_batch].sum(dim=-1).mean().item()

                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

            actions_log_prob_batch = self.actor.get_log_prob(actions_batch)
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
            entropy_actor, entropy_clock = self.actor.entropy
            entropy_loss = (self.cfg.entropy_coef * entropy_actor[mask_batch] + self.cfg.clock_entropy_coef * entropy_clock[mask_batch]).mean()

            # Symmetry loss
            batch_size = 4
            action_mean_original = self.actor.action_mean[:batch_size].detach()

            obs_mirrored_batch = obs_batch[:batch_size].flatten(0, 1).mirror().unflatten(0, (batch_size, -1))
            self.actor.train_act(
                obs_mirrored_batch,
                (hidden_states_batch, modulator_hidden_states_batch),
                use_estimated_values=use_estimated_values_batch[:batch_size]
            )

            mu_batch = obs_batch.mirror_dof_prop_by_x(action_mean_original.flatten(0, 1)).unflatten(0, (batch_size, -1))
            symmetry_loss = 0.1 * self.mse_loss(mu_batch, self.actor.action_mean)

            loss = (surrogate_loss + self.cfg.value_loss_coef * value_loss - entropy_loss + symmetry_loss)

        # Use KL to adaptively update learning rate
        if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        # Gradient step
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # self.scaler.unscale_(self.optimizer)
        # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.cfg.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return value_loss.item(), surrogate_loss.item(), entropy_loss.item(), kl_mean

    # @torch.compile()
    def _update_estimation(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            hidden_states_batch = batch['hidden_states'] if self.actor.is_recurrent else None
            mask_batch = batch['masks'].squeeze() if self.actor.is_recurrent else slice(None)
            obs_next_batch = batch['observations_next']

            ot1, est_vel, est_mu, est_logvar = self.actor.estimate(obs_batch, hidden_states=hidden_states_batch)

            # privileged information estimation loss
            estimation_loss = self.mse_loss(est_vel[mask_batch], obs_batch.priv_actor[mask_batch])

            # Ot+1 prediction and VAE loss
            prediction_loss = self.mse_loss(ot1[mask_batch], obs_next_batch.proprio[mask_batch])
            vae_loss = 1 + est_logvar - est_mu.pow(2) - est_logvar.exp()
            vae_loss = -0.5 * vae_loss[mask_batch].sum(dim=1).mean()

            loss = estimation_loss + prediction_loss + vae_loss

        # Gradient step
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # self.scaler.unscale_(self.optimizer)
        # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.cfg.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return estimation_loss.item(), prediction_loss.item(), vae_loss.item()

    def play_act(self, obs, **kwargs):
        return self.actor.act(obs, **kwargs)

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):

        if 'modulator_state_dict' in loaded_dict:
            self.actor.load_state_dict(loaded_dict['modulator_state_dict'])
        else:
            print("Modulator state dict not found, training from scratch!")
            self.actor.blind_actor.load_state_dict(loaded_dict['actor_state_dict'])
            # self.actor.load_state_dict(loaded_dict['actor_state_dict'])

        self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        # if load_optimizer:
        #     self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

        self.actor.reset_std(self.cfg.init_noise_std, device=self.device)

        return loaded_dict['infos']

    def save(self):
        return {
            'modulator_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
