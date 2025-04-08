import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

# from rsl_rl.modules.model_zju_gru import EstimatorNoRecon, EstimatorGRU
from rsl_rl.modules.model_zju_exp import EstimatorNoRecon, EstimatorGRU
from rsl_rl.modules.utils import UniversalCritic
from rsl_rl.storage import RolloutStorage
# from rsl_rl.storage import RolloutStorageMultiCritic as RolloutStorage
from .alg_base import BaseAlgorithm

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class Transition:
    def __init__(self, enable_reconstructor):
        self.observations = None
        self.critic_observations = None
        self.obs_enc_hidden_states = None
        if enable_reconstructor:
            self.recon_hidden_states = None
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


class PPO_ZJU_Multi_Critic(BaseAlgorithm):
    def __init__(self, task_cfg, device=torch.device('cpu'), env=None, **kwargs):
        self.env = env

        # PPO parameters
        self.task_cfg = task_cfg
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device
        self.enable_reconstructor = task_cfg.policy.enable_reconstructor

        # PPO component
        if self.enable_reconstructor:
            self.actor = EstimatorGRU(task_cfg.env, task_cfg.policy).to(self.device)
        else:
            self.actor = EstimatorNoRecon(task_cfg.env, task_cfg.policy).to(self.device)

        self.critic = nn.ModuleDict({
            'default': UniversalCritic(task_cfg.env, task_cfg.policy),
            'contact': UniversalCritic(task_cfg.env, task_cfg.policy),
        }).to(self.device)
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)

        # reconstructor
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Rollout Storage
        self.transition = Transition(self.enable_reconstructor)
        # self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)
        self.storage = RolloutStorage(task_cfg, self.device)

        self.storage.register_storage('rewards_contact', 0, (1,))
        self.storage.register_storage('values_contact', 0, (1,))
        self.storage.register_storage('returns_contact', 0, (1,))

        self.storage.register_storage('obs_enc_hidden_states', 1, (1, task_cfg.policy.obs_gru_hidden_size))
        if self.enable_reconstructor:
            self.storage.register_storage('recon_hidden_states', 1, (2, task_cfg.policy.recon_gru_hidden_size))
            self.storage.register_storage('observations_next', 2, cfg=task_cfg.env.obs_next)

        self.storage.compose_storage()

    def act(self, obs, obs_critic, use_estimated_values: torch.Tensor = None, **kwargs):
        # act function should run within torch.inference_mode context
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            # store observations
            self.transition.observations = obs
            self.transition.critic_observations = obs_critic
            if self.actor.is_recurrent:
                hidden = self.actor.get_hidden_state()
                self.transition.obs_enc_hidden_states = hidden[0]
                if self.enable_reconstructor:
                    self.transition.recon_hidden_states = hidden[1]

            actions = self.actor.act(obs, use_estimated_values=use_estimated_values)

            if self.actor.is_recurrent and self.transition.obs_enc_hidden_states is None:
                # only for the first step where hidden_state is None
                hidden = self.actor.get_hidden_state()
                self.transition.obs_enc_hidden_states = 0 * hidden[0]
                if self.enable_reconstructor:
                    self.transition.recon_hidden_states = 0 * hidden[1]

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
        if self.enable_reconstructor:
            self.transition.observations_next = args[0].as_obs_next()
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
        if self.actor.is_recurrent:
            self.actor.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values_default = self.critic['default'].evaluate(last_critic_obs).detach()
        last_values_contact = self.critic['contact'].evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values_default, last_values_contact, self.cfg.gamma, self.cfg.lam)

    def update(self, cur_it=0, **kwargs):
        if cur_it > 20000:
            update_est = cur_it % 5 == 0
        else:
            update_est = True

        update_est &= self.enable_reconstructor
        mean_value_loss = 0
        mean_default_value_loss = 0
        mean_contact_value_loss = 0
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

        if self.actor.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)

        for batch in generator:
            # ########################## policy loss ##########################
            kl_mean, value_loss_default, value_loss_contact, surrogate_loss, entropy_loss, symmetry_loss = self._compute_policy_loss(batch)
            loss = surrogate_loss + self.cfg.value_loss_coef * (value_loss_default + value_loss_contact) - entropy_loss + symmetry_loss

            num_updates += 1
            # policy statistics
            kl_change.append(kl_mean.item())
            mean_kl += kl_mean.item()
            mean_default_value_loss += value_loss_default
            mean_contact_value_loss += value_loss_contact
            mean_value_loss = mean_default_value_loss + mean_contact_value_loss
            mean_surrogate_loss += surrogate_loss
            mean_entropy_loss += entropy_loss
            mean_symmetry_loss += symmetry_loss

            # ########################## estimation loss ##########################
            loss_est = 0
            if update_est:
                estimation_loss, prediction_loss, vae_loss, recon_rough_loss, recon_refine_loss = self._compute_estimation_loss(batch)
                loss_est = estimation_loss + prediction_loss + vae_loss + recon_rough_loss + recon_refine_loss

                # estimation statistics
                mean_estimation_loss += estimation_loss.item()
                mean_prediction_loss += prediction_loss.item()
                mean_vae_loss += vae_loss.item()
                mean_recon_rough_loss += recon_rough_loss.item()
                mean_recon_refine_loss += recon_refine_loss.item()

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
            self.scaler.scale(loss + loss_est).backward()
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
        if update_est:
            mean_estimation_loss /= num_updates
            mean_prediction_loss /= num_updates
            mean_vae_loss /= num_updates
            mean_recon_rough_loss /= num_updates
            mean_recon_refine_loss /= num_updates

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

        if update_est:
            return_dict.update({
                'Loss/estimation_loss': mean_estimation_loss,
                'Loss/Ot+1 prediction_loss': mean_prediction_loss,
                'Loss/VAE_loss': mean_vae_loss,
                'Loss/recon_rough_loss': mean_recon_rough_loss,
                'Loss/recon_refine_loss': mean_recon_refine_loss,
            })

        return return_dict

    # @torch.compile
    def _compute_policy_loss(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            obs_enc_hidden_states_batch = batch['obs_enc_hidden_states']
            recon_hidden_states_batch = batch['recon_hidden_states'] if self.enable_reconstructor else None
            mask_batch = batch['masks'].squeeze()
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
                hidden_states=(obs_enc_hidden_states_batch, recon_hidden_states_batch),
                use_estimated_values=use_estimated_values_batch
            )

            with torch.no_grad():
                kl_mean = kl_divergence(
                    Normal(batch['action_mean'], batch['action_sigma']),
                    Normal(self.actor.action_mean, self.actor.action_std)
                )[mask_batch].sum(dim=-1).mean()

            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            evaluation_default = self.critic['default'].evaluate(critic_obs_batch)
            evaluation_contact = self.critic['contact'].evaluate(critic_obs_batch)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped)[mask_batch].mean()

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped_default = default_values_batch + (
                        evaluation_default - default_values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses_default = (evaluation_default - default_returns_batch).pow(2)
                value_losses_clipped_default = (value_clipped_default - default_returns_batch).pow(2)
                value_losses_default = torch.max(value_losses_default, value_losses_clipped_default)[mask_batch].mean()

                value_clipped_contact = contact_values_batch + (
                        evaluation_contact - contact_values_batch).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses_contact = (evaluation_contact - contact_returns_batch).pow(2)
                value_losses_clipped_contact = (value_clipped_contact - contact_returns_batch).pow(2)
                value_losses_contact = torch.max(value_losses_contact, value_losses_clipped_contact)[mask_batch].mean()
            else:
                value_losses_default = (evaluation_default - default_returns_batch)[mask_batch].pow(2).mean()
                value_losses_contact = (evaluation_contact - contact_returns_batch)[mask_batch].pow(2).mean()

            # Entropy loss
            entropy_loss = self.cfg.entropy_coef * self.actor.entropy[mask_batch].mean()

            # Symmetry loss
            batch_size = 4
            action_mean_original = self.actor.action_mean[:batch_size].detach()

            obs_mirrored_batch = obs_batch[:batch_size].flatten(0, 1).mirror().unflatten(0, (batch_size, -1))
            self.actor.train_act(
                obs_mirrored_batch,
                hidden_states=(obs_enc_hidden_states_batch, recon_hidden_states_batch),
                use_estimated_values=use_estimated_values_batch[:batch_size]
            )

            mu_batch = obs_batch.mirror_dof_prop_by_x(action_mean_original.flatten(0, 1)).unflatten(0, (batch_size, -1))
            symmetry_loss = 0.1 * self.mse_loss(mu_batch, self.actor.action_mean)

            return kl_mean, value_losses_default, value_losses_contact, surrogate_loss, entropy_loss, symmetry_loss

    # @torch.compile
    def _compute_estimation_loss(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            batch_size = 4

            obs_batch = batch['observations'][:batch_size]
            obs_enc_hidden_states_batch = batch['obs_enc_hidden_states'][:batch_size].contiguous()
            recon_hidden_states_batch = batch['recon_hidden_states'][:batch_size].contiguous()
            obs_next_batch = batch['observations_next'][:batch_size]
            mask_batch = batch['masks'][:batch_size, :, 0]

            recon_rough, recon_refine, est_latent, est, est_logvar, ot1 = self.actor.reconstruct(
                obs_batch, obs_enc_hidden_states_batch, recon_hidden_states_batch)

            # privileged information estimation loss
            estimation_loss = self.mse_loss(est[mask_batch], obs_batch.priv_actor[mask_batch])

            # Ot+1 prediction and VAE loss
            prediction_loss = self.mse_loss(ot1[mask_batch], obs_next_batch.proprio[mask_batch])
            vae_loss = 1 + est_logvar - est_latent.pow(2) - est_logvar.exp()
            vae_loss = -0.5 * vae_loss[mask_batch].sum(dim=1).mean()

            # reconstructor loss
            scan = obs_batch.scan[mask_batch]
            recon_rough_loss = self.mse_loss(recon_rough[mask_batch], scan)
            recon_refine_loss = self.l1_loss(recon_refine[mask_batch], scan)

        return estimation_loss, prediction_loss, vae_loss, recon_rough_loss, recon_refine_loss

    def play_act(self, obs, **kwargs):
        if 'use_estimated_values' in kwargs:
            use_estimated_values_bool = kwargs['use_estimated_values']
            kwargs['use_estimated_values'] = use_estimated_values_bool & torch.ones(
                self.task_cfg.env.num_envs, 1, dtype=torch.bool, device=self.device)

        return self.actor.act(obs, **kwargs)

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        # self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, device=self.device)

    def save(self):
        return {
            'actor_state_dict': self.actor.state_dict(),
            # 'critic_state_dict': self.critic.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
        }
