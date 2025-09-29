import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from legged_gym.utils.helpers import class_to_dict
from rsl_rl.algorithms.alg_base import BaseAlgorithm
from rsl_rl.algorithms.template_models import UniversalCritic, AMPDiscriminator
from rsl_rl.datasets.amp_motion_loader import AMPMotionLoader
from rsl_rl.storage import RolloutStorageMultiCritic as RolloutStorage
from rsl_rl.storage.amp_replay_buffer import AMPReplayBuffer
from . import EmpiricalNormalization
from .networks import Policy

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
        self.mixer_hidden_states = None
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

    def get_items(self):
        return self.__dict__.items()

    def clear(self):
        for key in self.__dict__:
            setattr(self, key, None)


class PPO_PIE_AMP(BaseAlgorithm):
    def __init__(self, task_cfg, device=torch.device('cpu'), env=None, **kwargs):
        self.env = env

        # PPO parameters
        self.task_cfg = task_cfg
        self.amp_cfg = class_to_dict(task_cfg.amp)
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.amp_lr = self.cfg.amp_lr
        self.device = device

        # Initialize adaptive KL coefficient
        self.kl_coef_vel = 0.01
        self.kl_coef_z = 0.01

        self.amp_obs = torch.zeros(self.task_cfg.env.num_envs, 26, 3, device=self.device)

        self.cur_it = 0

        # PPO components
        self.actor = Policy(task_cfg.env, task_cfg.policy).to(self.device)
        self.actor.reset_std(self.cfg.init_noise_std, self.device)
        self.mixer_hidden_states = None

        self.critic = nn.ModuleDict({
            'default': UniversalCritic(task_cfg.env, task_cfg.policy),
            'contact': UniversalCritic(task_cfg.env, task_cfg.policy),
        }).to(self.device)
        params_ac = [
            {"params": self.actor.parameters(), "name": "actor"},
            {"params": self.critic.parameters(), "name": "critic"},
        ]
        self._build_amp_discriminator()

        self.optimizer_ppo = torch.optim.Adam(params_ac, lr=self.learning_rate)
        self.scaler_ppo = GradScaler(enabled=self.cfg.use_amp)

        self.optimizer_sym = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.scaler_sym = GradScaler(enabled=self.cfg.use_amp)

        self.optimizer_vae = torch.optim.Adam(
            [*self.actor.mixer.parameters(), *self.actor.vae.parameters()],
            lr=self.learning_rate
        )
        self.scaler_vae = GradScaler(enabled=self.cfg.use_amp)

        # Rollout Storage
        self.transition = Transition()
        self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)
        self.mse_loss = nn.MSELoss()

    def _build_amp_discriminator(self):
        amp_disc_cfg = self.amp_cfg["amp_disc_cfg"]
        policy_dt = self.task_cfg.control.decimation * self.task_cfg.sim.dt
        rew_norm_factor = self.task_cfg.rewards.rew_norm_factor
        amp_disc_cfg["amp_reward_coef"] = amp_disc_cfg["amp_reward_coef"] * policy_dt * rew_norm_factor
        amp_disc_cfg["task_rew_schedule_dict"]["traking_lin_vel_max"] = self.task_cfg.rewards.scales.tracking_lin_vel * rew_norm_factor
        self.amp_disc = AMPDiscriminator(device=self.device, **amp_disc_cfg)

        self.amp_motion_loader = AMPMotionLoader(motion_cfg=self.amp_cfg,
                                                 env_dt=policy_dt,
                                                 amp_num_frames=self.amp_cfg["amp_obs_hist_steps"],
                                                 device=self.device)

        self.norm_amp_obs = self.amp_cfg.get("amp_empirical_normalization", False)
        if self.norm_amp_obs:
            amp_norm_until = self.amp_cfg.get("amp_normal_update_until", None)
            self.amp_obs_normalizer = EmpiricalNormalization(shape=[self.amp_cfg["num_single_amp_obs"]],
                                                             num_repeats=self.amp_cfg["amp_obs_hist_steps"],
                                                             until=amp_norm_until).to(self.device)
        else:
            self.amp_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        amp_optim_cfg = self.amp_cfg["amp_optim_cfg"]
        params_dis = [
            {"params": self.amp_disc.trunk.parameters(), "weight_decay": amp_optim_cfg["amp_trunk_weight_decay"], "name": "amp_trunk",
             "lr": amp_optim_cfg["amp_disc_lr"]},
            {"params": self.amp_disc.amp_linear.parameters(), "weight_decay": amp_optim_cfg["amp_head_weight_decay"], "name": "amp_head",
             "lr": amp_optim_cfg["amp_disc_lr"]},
        ]
        self.optimizer_amp = torch.optim.Adam(params_dis, lr=amp_optim_cfg["amp_disc_lr"])
        self.scaler_amp = GradScaler(enabled=self.cfg.use_amp)
        self.amp_replay_buffer = AMPReplayBuffer(self.amp_disc.num_input, amp_optim_cfg["amp_replay_buffer_size"], self.device)
        self.amploss_coef = amp_optim_cfg["amp_loss_coef"]
        self.max_amp_disc_grad_norm = amp_optim_cfg["max_amp_disc_grad_norm"]
        self.amp_update_interval = amp_optim_cfg["amp_update_interval"]

    def act(self,
            obs,
            obs_critic,
            amp_obs: torch.Tensor = None,
            **kwargs):
        # act function should run within torch.inference_mode context
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            # store observations
            self.transition.observations = obs
            self.transition.critic_observations = obs_critic

            actions, self.mixer_hidden_states = self.actor.act(
                proprio=obs.proprio.unsqueeze(0),
                prop_his=obs.prop_his.unsqueeze(0),
                depth=obs.depth.unsqueeze(0),
                mixer_hidden_states=self.mixer_hidden_states,
            )

            if self.transition.mixer_hidden_states is None:
                # only for the first step where hidden_state is None
                self.transition.mixer_hidden_states = torch.zeros_like(self.mixer_hidden_states)

            # store
            self.transition.actions = actions
            self.transition.values = self.critic['default'].evaluate(obs_critic)
            self.transition.values_contact = self.critic['contact'].evaluate(obs_critic)
            self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions)
            self.transition.action_mean = self.actor.action_mean
            self.transition.action_sigma = self.actor.action_std

            self.amp_replay_buffer.insert(amp_obs.detach())

            return actions

    def process_env_step(self, rewards, dones, infos, *args):
        self.transition.observations_next = args[0].as_obs_next()
        self.transition.dones = dones.unsqueeze(1)

        # from Logan
        step_rew: dict = infos['step_rew']

        rew_contact = step_rew.get('feet_edge', 0.)
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
        last_values_default = self.critic['default'].evaluate(last_critic_obs).detach()
        last_values_contact = self.critic['contact'].evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values_default, last_values_contact, self.cfg.gamma, self.cfg.lam)

    def update(self, cur_it=0, **kwargs):
        self.cur_it = cur_it
        mean_value_loss_default = 0
        mean_value_loss_contact = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        mean_symmetry_loss = 0
        mean_vel_est_loss = 0
        mean_ot1_loss = 0
        mean_recon_loss = 0
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

        amp_policy_generator = self.amp_replay_buffer.feed_forward_generator(
            self.cfg.num_learning_epochs * self.cfg.num_mini_batches,
            self.task_cfg.env.num_envs * self.storage.num_transitions_per_env // self.cfg.num_mini_batches,
        )
        amp_expert_generator = self.amp_motion_loader.feed_forward_generator(
            self.cfg.num_learning_epochs * self.cfg.num_mini_batches,
            self.task_cfg.env.num_envs * self.storage.num_transitions_per_env // self.cfg.num_mini_batches,
        )

        generator = self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)

        for batch, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            num_updates += 1

            # ########################## policy loss ##########################
            ppo_metrics = self.update_ppo(batch)
            mean_kl += ppo_metrics['kl_mean']
            mean_value_loss_default += ppo_metrics['value_loss_default']
            mean_value_loss_contact += ppo_metrics['value_loss_contact']
            mean_surrogate_loss += ppo_metrics['surrogate_loss']
            mean_entropy_loss += ppo_metrics['entropy_loss']
            kl_change.append(ppo_metrics['kl_mean'])

            # ########################## discriminator loss ##########################
            if cur_it % self.amp_update_interval == 0:
                amp_metrics = self.update_amp(sample_amp_policy, sample_amp_expert)
                mean_amp_loss += amp_metrics['amp_loss']
                mean_grad_pen_loss += amp_metrics['grad_pen_loss']
                mean_policy_pred += amp_metrics['policy_pred']
                mean_expert_pred += amp_metrics['expert_pred']

            # # Symmetry Update
            # sym_metrics = self.update_symmetry(cur_it, batch)
            # mean_symmetry_loss += sym_metrics['symmetry_loss']

            if cur_it > 500 and cur_it % 3 == 0:
                est_metrics = self.update_estimation(batch)
                mean_vel_est_loss += est_metrics['vel_est_loss']
                mean_ot1_loss += est_metrics['ot1_loss']
                mean_recon_loss += est_metrics['recon_loss']
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
        mean_value_loss_default /= num_updates
        mean_value_loss_contact /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        # ---- AMP ----
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        # ---- SYM ----
        mean_symmetry_loss /= num_updates
        # ---- VAE ----
        mean_vel_est_loss /= num_updates
        mean_ot1_loss /= num_updates
        mean_recon_loss /= num_updates
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

        self.amp_disc.update_lambda(mean_policy_pred)

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
            'Loss/symmetry_loss': mean_symmetry_loss,
        }

        if cur_it % self.amp_update_interval == 0:
            metrics.update({
                'AMP/amp_loss': mean_amp_loss,
                'AMP/grad_pen_loss': mean_grad_pen_loss,
                'AMP/policy_pred': mean_policy_pred,
                'AMP/expert_pred': mean_expert_pred,
            })

        if cur_it > 500 and cur_it % 3 == 0:
            metrics.update({
                'VAE/vel_est_loss': mean_vel_est_loss,
                'VAE/Ot+1_loss': mean_ot1_loss,
                'VAE/recon_loss': mean_recon_loss,
                'VAE/vel_kl_loss': mean_vel_kl_loss,
                'VAE/z_kl_loss': mean_z_kl_loss,
                'VAE/total_kl_loss': mean_vel_kl_loss + mean_z_kl_loss,
                'VAE/abs_vel': mean_abs_vel,
                'VAE/abs_z': mean_abs_z,
                'VAE/std_vel': mean_std_vel,
                'VAE/std_z': mean_std_z,
                'VAE/SNR_vel': mean_snr_vel,
                'VAE/SNR_z': mean_snr_z,
                'VAE/kl_coef_vel': mean_kl_coef_vel,
                'VAE/kl_coef_z': mean_kl_coef_z,
            })

        return metrics

    def update_ppo(self, batch: dict):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs = batch['observations']
            critic_obs = batch['critic_observations']
            mixer_hidden_states = batch['mixer_hidden_states'] if self.actor.is_recurrent else None
            mask = batch['masks']
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
            )

            with torch.no_grad():
                kl_mean = kl_divergence(
                    Normal(batch['action_mean'], batch['action_sigma']),
                    Normal(self.actor.action_mean, self.actor.action_std)
                ).sum(dim=2, keepdim=True)
                kl_mean = masked_mean(kl_mean, mask)

            actions_log_prob = self.actor.get_actions_log_prob(actions)
            evaluation_default = self.critic['default'].evaluate(critic_obs)
            evaluation_contact = self.critic['contact'].evaluate(critic_obs)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - old_actions_log_prob)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate_loss = masked_mean(torch.maximum(surrogate, surrogate_clipped), mask)

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
                value_loss_contact = masked_mean(torch.maximum(value_loss_contact, value_loss_clipped_contact), mask)
            else:
                value_loss_default = masked_MSE(evaluation_default, default_returns, mask)
                value_loss_contact = masked_MSE(evaluation_contact, contact_returns, mask)

            value_loss = value_loss_default + value_loss_contact

            # Entropy loss
            entropy_loss = -masked_mean(self.actor.entropy, mask)

            # Total PPO loss
            total_loss = surrogate_loss + self.cfg.value_loss_coef * value_loss + self.cfg.entropy_coef * entropy_loss

        # Use KL to adaptively update learning rate
        if self.cfg.schedule == 'adaptive' and self.cfg.desired_kl is not None:
            if kl_mean > self.cfg.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                self.learning_rate = min(1e-3, self.learning_rate * 1.5)

            for param_group in self.optimizer_ppo.param_groups:
                param_group['lr'] = self.learning_rate

        # Gradient step
        self.optimizer_ppo.zero_grad()
        self.scaler_ppo.scale(total_loss).backward()
        self.scaler_ppo.step(self.optimizer_ppo)
        self.scaler_ppo.update()

        self.actor.clip_std(self.cfg.noise_range[0], self.cfg.noise_range[1])

        return {
            'kl_mean': kl_mean,
            'value_loss_default': value_loss_default.item(),
            'value_loss_contact': value_loss_contact.item(),
            'surrogate_loss': surrogate_loss.item(),
            'entropy_loss': -entropy_loss.item(),
        }

    def update_amp(self, sample_amp_policy, sample_amp_expert):
        with torch.no_grad():
            sample_amp_expert = self.amp_obs_normalizer(sample_amp_expert)
        amp_loss, grad_pen_loss, gen_d, ref_d = self.amp_disc.compute_amp_loss(sample_amp_expert, sample_amp_policy)

        loss_dis = self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

        self.optimizer_amp.zero_grad()
        self.scaler_amp.scale(loss_dis).backward()
        # 先反缩放梯度，然后进行梯度裁剪
        self.scaler_amp.unscale_(self.optimizer_amp)
        nn.utils.clip_grad_norm_(self.amp_disc.parameters(), self.max_amp_disc_grad_norm)
        self.scaler_amp.step(self.optimizer_amp)
        self.scaler_amp.update()

        return {
            'amp_loss': amp_loss.item(),
            'grad_pen_loss': grad_pen_loss.item(),
            'policy_pred': gen_d.item(),
            'expert_pred': ref_d.item(),
        }

    def update_symmetry(self, cur_it, batch):
        """Update symmetry-related components"""
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            obs = batch['observations']
            mixer_hidden_states = batch['mixer_hidden_states'] if self.actor.is_recurrent else None
            mask = batch['masks'] if self.actor.is_recurrent else slice(None)
            n_steps = mask.size(0)

            self.actor.act(
                proprio=obs.proprio,
                prop_his=obs.prop_his,
                depth=obs.depth,
                mixer_hidden_states=mixer_hidden_states,
            )
            action_mean_original = self.actor.action_mean.clone().detach()

            # Symmetry loss computation
            obs_mirrored = obs.flatten(0, 1).mirror().unflatten(0, (n_steps, -1))
            self.actor.act(
                proprio=obs_mirrored.proprio,
                prop_his=obs_mirrored.prop_his,
                depth=obs_mirrored.depth,
                mixer_hidden_states=mixer_hidden_states,
            )

            mu_batch = obs.mirror_dof_prop_by_x(action_mean_original.flatten(0, 1)).unflatten(0, (n_steps, -1))
            symmetry_loss = masked_MSE(mu_batch, self.actor.action_mean, mask)

        # Gradient step
        self.optimizer_sym.zero_grad()
        self.scaler_sym.scale(symmetry_loss).backward()
        self.scaler_sym.step(self.optimizer_sym)
        self.scaler_sym.update()

        return {
            'symmetry_loss': symmetry_loss.item(),
        }


    def update_estimation(
            self,
            batch,
            target_snr_vel=5.0,  # Target SNR for velocity (mean/std ratio)
            target_snr_z=5.0,  # Target SNR for z (mean/std ratio)
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
        total_loss = vel_est_loss + ot1_loss + recon_loss + kl_loss

        # Gradient step
        self.optimizer_vae.zero_grad()
        self.scaler_vae.scale(total_loss).backward()
        self.scaler_vae.step(self.optimizer_vae)
        self.scaler_vae.update()

        return {
            'vel_est_loss': vel_est_loss.item(),
            'ot1_loss': ot1_loss.item(),
            'recon_loss': recon_loss.item(),
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

    def reset(self, dones):
        if self.mixer_hidden_states is not None:
            self.mixer_hidden_states[:, dones] = 0.

    def play_act(self, obs, **kwargs):
        with torch.autocast(self.device.type, torch.float16, enabled=self.cfg.use_amp):
            actions, vel_est, self.mixer_hidden_states = self.actor.act(
                proprio=obs.proprio.unsqueeze(0),
                prop_his=obs.prop_his.unsqueeze(0),
                depth=obs.depth.unsqueeze(0),
                mixer_hidden_states=self.mixer_hidden_states,
                **kwargs
            )

            return {'actions': actions, 'vel_est': vel_est}

    def train(self):
        self.actor.train()
        self.critic.train()
        self.amp_disc.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.critic.load_state_dict(loaded_dict['critic_state_dict'])
        self.amp_disc.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.amp_obs_normalizer.load_state_dict(loaded_dict['amp_obs_normalizer_state_dict'])

        if load_optimizer:
            self.optimizer_ppo.load_state_dict(loaded_dict['optimizer_ac_state_dict'])
            self.optimizer_amp.load_state_dict(loaded_dict['optimizer_dis_state_dict'])

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std)

    def save(self):
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'discriminator_state_dict': self.amp_disc.state_dict(),
            'amp_obs_normalizer_state_dict': self.amp_obs_normalizer.state_dict(),
            'optimizer_ac_state_dict': self.optimizer_ppo.state_dict(),
            'optimizer_dis_state_dict': self.optimizer_amp.state_dict(),
        }
