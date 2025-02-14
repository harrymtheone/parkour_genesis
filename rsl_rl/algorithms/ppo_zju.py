import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

from rsl_rl.algorithms import BaseAlgorithm
from rsl_rl.modules.model_zju import Estimator, Critic
from rsl_rl.storage import RolloutStoragePerception as RolloutStorage

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class Transition:
    def __init__(self):
        self.observations = None
        self.critic_observations = None
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
        self.__init__()


class PPO_ZJU(BaseAlgorithm):
    def __init__(self, env_cfg, train_cfg, device=torch.device('cpu')):

        # PPO parameters
        self.cfg = train_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device

        # PPO component
        self.actor = Estimator(env_cfg, train_cfg).to(self.device)
        self.critic = Critic(env_cfg, train_cfg).to(self.device)
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)

        # reconstructor
        self.recon_prev = torch.zeros(env_cfg.num_envs, *env_cfg.scan_shape, device=device)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss(reduction='none')

        # Rollout Storage
        self.transition = Transition()
        self.storage = RolloutStorage(env_cfg.num_envs, train_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, obs_critic, use_estimated_values=True):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            # store observations and previous construction
            obs.recon_prev = self.recon_prev.clone()
            self.transition.observations = obs
            self.transition.critic_observations = obs_critic

            actions, _, recon_refine, _ = self.actor.act(obs, use_estimated_values=use_estimated_values)

            self.transition.actions = actions.detach()
            self.transition.values = self.critic.evaluate(obs_critic).detach()
            self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
            self.transition.action_mean = self.actor.action_mean.detach()
            self.transition.action_sigma = self.actor.action_std.detach()
            self.transition.use_estimated_values = use_estimated_values

            self.recon_prev[:] = recon_refine.detach()
            return self.transition.actions

    def process_env_step(self, rewards, dones, infos, *args):
        self.transition.observations_next = args[0].as_obs_next()
        self.transition.rewards = rewards.clone().unsqueeze(1)
        self.transition.dones = dones.unsqueeze(1)

        # Bootstrapping on time-outs
        if 'time_outs' in infos:
            bootstrapping = torch.logical_or(infos['time_outs'], infos['reach_goals']).unsqueeze(1)
            self.transition.rewards += self.cfg.gamma * self.transition.values * bootstrapping.to(self.device)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor.reset(dones)

        return rewards

    def reset(self, dones):
        pass

    def compute_returns(self, last_critic_obs):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            last_values = self.critic.evaluate(last_critic_obs).detach()
            self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, update_est=True, **kwargs):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_recon_rough_loss = 0
        mean_recon_refine_loss = 0
        mean_estimation_loss = 0
        mean_prediction_loss = 0
        mean_vae_loss = 0
        mean_symmetry_loss = 0

        kl_change = []
        num_updates = 0

        for batch in self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):

            rtn = self._compute_loss(batch, update_est)
            loss, kl_mean, value_loss, surrogate_loss, entropy_loss, estimation_loss, prediction_loss, vae_loss, recon_rough_loss, recon_refine_loss, symmetry_loss = rtn

            # Use KL to adaptively update learning rate
            if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
                kl_change.append(kl_mean)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            # Gradient step
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            # self.scaler.unscale_(self.optimizer)
            # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            num_updates += 1
            mean_value_loss += value_loss
            mean_surrogate_loss += surrogate_loss
            mean_entropy_loss += entropy_loss
            mean_kl += kl_mean

            if update_est:
                mean_estimation_loss += estimation_loss
                mean_prediction_loss += prediction_loss
                mean_vae_loss += vae_loss
                mean_recon_rough_loss += recon_rough_loss
                mean_recon_refine_loss += recon_refine_loss
                mean_symmetry_loss += symmetry_loss

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates
        mean_estimation_loss /= num_updates
        mean_prediction_loss /= num_updates
        mean_vae_loss /= num_updates
        mean_recon_rough_loss /= num_updates
        mean_recon_refine_loss /= num_updates
        mean_symmetry_loss /= num_updates

        kl_str = 'kl: '
        for k in kl_change:
            kl_str += f'{k:.3f} | '
        print(kl_str)

        self.storage.clear()
        return_dict = {
            'Loss/learning_rate': self.learning_rate,
            'Loss/value_loss': mean_value_loss,
            'Loss/kl_div': mean_kl,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Policy/noise_std': self.actor.log_std.exp().mean().item(),
        }
        if update_est:
            return_dict.update({
                'Loss/estimation_loss': mean_estimation_loss,
                'Loss/Ot+1 prediction_loss': mean_prediction_loss,
                'Loss/VAE_loss': mean_vae_loss,
                'Loss/recon_rough_loss': mean_recon_rough_loss,
                'Loss/recon_refine_loss': mean_recon_refine_loss,
                'Loss/symmetry_loss': mean_symmetry_loss,
            })
        return return_dict

    @torch.compile(mode='default')
    def _compute_loss(self, batch: dict, update_est: bool):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            obs_next_batch = batch['observations_next']
            actions_batch = batch['actions']
            target_values_batch = batch['values']
            advantages_batch = batch['advantages']
            returns_batch = batch['returns']
            old_mu_batch = batch['action_mean']
            old_sigma_batch = batch['action_sigma']
            old_actions_log_prob_batch = batch['actions_log_prob']
            use_estimated_values_batch = batch['use_estimated_values']

            est_mu = self.actor.train_act(obs_batch, use_estimated_values=use_estimated_values_batch)  # match distribution dimension

            # Use KL to adaptively update learning rate
            if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
                with torch.no_grad():
                    kl_mean = kl_divergence(
                        Normal(old_mu_batch, old_sigma_batch),
                        Normal(self.actor.action_mean, self.actor.action_std)
                    ).sum(dim=1).mean().item()

                    if kl_mean > self.cfg.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            value_batch = self.critic.evaluate(critic_obs_batch)
            mu_batch = self.actor.action_mean
            sigma_batch = self.actor.action_std

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = advantages_batch * ratio
            surrogate_clipped = advantages_batch * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

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

            if update_est:
                # privileged information estimation loss
                estimation_loss = self.mse_loss(est_mu[:, 16:], obs_batch.priv[:, :3 + 16 + 16])

                # # Ot+1 prediction and VAE loss
                # prediction_loss = self.mse_loss(ot1, obs_next_batch.proprio)
                # vae_loss = -0.5 * torch.sum(1 + est_logvar - est_mu[:, :16].pow(2) - est_logvar.exp(), dim=1).mean()

                batch_size = 256
                indices = torch.randperm(len(actions_batch))[:batch_size]

                # reconstructor loss
                recon_rough, recon_refine = self.actor.reconstruct(obs_batch.prop_his[indices],
                                                                   obs_batch.depth[indices],
                                                                   obs_batch.recon_prev[indices])  # match distribution dimension
                recon_rough_loss = self.mse_loss(recon_rough, obs_batch.scan[indices])
                recon_refine_loss = self.l1_loss(recon_refine, obs_batch.scan[indices]).mean()
                recon_loss = recon_rough_loss + recon_refine_loss

                # # Symmetry loss
                # obs_mirrored_batch = obs_batch.slice(indices).data_augmentation(only_mirrored=True)
                # self.actor.act(obs_mirrored_batch, use_estimated_values=use_estimated_values_batch[indices])
                #
                # mu_batch = obs_batch.mirror_dof_prop_by_x(mu_batch[indices].detach())
                # sigma_batch = torch.abs(obs_batch.mirror_dof_prop_by_x(sigma_batch[indices].detach()))

                # # symmetry_loss = 0.1 * self.mse_loss(mu_batch[indices], self.actor.action_mean)
                # symmetry_loss = 0.1 * kl_divergence(
                #     Normal(self.actor.action_mean, self.actor.action_std),
                #     Normal(mu_batch, sigma_batch)
                # ).mean()

                # Total loss
                loss = (
                        loss
                        + estimation_loss
                        # + prediction_loss
                        # + vae_loss
                        + recon_loss
                    # + symmetry_loss
                )

                return (
                    loss,
                    kl_mean,
                    value_loss.item(),
                    surrogate_loss.item(),
                    entropy_loss.item(),
                    estimation_loss.item(),
                    0.,  # prediction_loss.item(),
                    0.,  # vae_loss.item(),
                    recon_rough_loss.item(),
                    recon_refine_loss.item(),
                    0.,  # symmetry_loss.item()
                )

            return loss, kl_mean, value_loss.item(), surrogate_loss.item(), entropy_loss.item(), 0., 0., 0., 0., 0., 0.,

    def play_act(self, obs, use_estimated_values=True):
        obs.recon_prev = self.recon_prev.clone()
        actions, recon_rough, recon_refine, latent_est = self.actor.act(obs.float(), use_estimated_values=use_estimated_values, eval_=True)
        self.recon_prev[:] = recon_refine.detach()
        return actions, recon_rough, recon_refine, latent_est

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
