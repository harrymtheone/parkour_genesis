import math
import random

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal

from legged_gym.envs.base.utils import CircularBuffer
from rsl_rl.modules.wmp import RSSM, ActorWMP, CriticWMP
from rsl_rl.storage import RolloutStoragePerception as RolloutStorage
from .alg_base import BaseAlgorithm

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


def symlog(x):
    # This one is for label transformation
    if isinstance(x, float):
        if abs(x) < 1e-8:
            return 0.
        else:
            return math.copysign(1., x) * math.log(abs(x) + 1.0)
    else:
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)  # (batch_size, 1)


def symexp(x):
    # Inverse of symlog
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class SymexpTwoHotLoss(nn.Module):
    def __init__(self, low, high, num_buckets=255, device='cuda'):
        super().__init__()
        self.low = low
        self.high = high
        self.num_buckets = num_buckets
        self.device = device

        # Buckets evenly spaced in symlog space
        self.buckets = torch.linspace(symlog(low), symlog(high), steps=num_buckets, device=device)

    def forward(self, data, target):
        """
        logits: (batch_size, num_buckets) raw outputs from network (before softmax)
        target: (batch_size,) continuous ground truth values (not transformed)
        """

        # 1. Transform target with symlog and convert to soft one-hot vector over buckets
        x = symlog(target).unsqueeze(-1)

        # Find bucket indices just below and above target values
        below = torch.sum(self.buckets <= x, dim=-1, dtype=torch.long) - 1
        above = torch.sum(self.buckets > x, dim=-1, dtype=torch.long)

        # Clamp indices to valid range
        below = torch.clamp(below, 0, self.num_buckets - 1)
        above = torch.clamp(above, 0, self.num_buckets - 1)

        equal = (below == above)

        dist_to_below = torch.where(equal, torch.tensor(1.0, device=self.device), torch.abs(self.buckets[below] - x.squeeze(-1)))
        dist_to_above = torch.where(equal, torch.tensor(1.0, device=self.device), torch.abs(self.buckets[above] - x.squeeze(-1)))
        total_dist = dist_to_below + dist_to_above

        weight_below = dist_to_above / total_dist
        weight_above = dist_to_below / total_dist

        target_soft = (
                nn.functional.one_hot(below, num_classes=self.num_buckets) * weight_below.unsqueeze(-1) +
                nn.functional.one_hot(above, num_classes=self.num_buckets) * weight_above.unsqueeze(-1)
        ).float().squeeze(2)

        # 2. Compute log probabilities from logits using log_softmax for numerical stability
        log_probs = nn.functional.log_softmax(data, dim=-1)

        # 3. Compute soft negative log likelihood loss
        loss = -torch.sum(target_soft * log_probs, dim=-1)  # (batch_size,)

        return loss.mean()


class SymlogMSELoss(nn.Module):
    def __init__(self, tolerance=1e-8, reduction='mean'):
        super().__init__()
        self.tolerance = tolerance
        self.reduction = reduction

    def forward(self, data, target):
        loss = nn.functional.mse_loss(data, symlog(target), reduction=self.reduction)
        return loss.clip(min=self.tolerance)


class WorldModelDataset:
    def __init__(self, task_cfg, device, batch_size=64, buffer_size=100, dataset_size=1000, dtype=torch.half, buf_device=torch.device('cuda')):
        self.device = device
        self.buffer_size = buffer_size
        self.dataset_size = dataset_size

        if batch_size <= task_cfg.env.num_envs:
            batch_size = task_cfg.env.num_envs

        self.envs_selected = torch.randperm(task_cfg.env.num_envs)[:batch_size]

        def create_buf(*data_shape):
            return CircularBuffer(buffer_size, batch_size, data_shape, device, buf_device, dtype)

        self.buffers = {
            "prop": create_buf(task_cfg.env.n_proprio - 12),
            "depth": create_buf(1, 64, 64),
            "action_his": create_buf(task_cfg.world_model.step_interval, task_cfg.env.num_actions),
            "state_deter": create_buf(512),
            "state_stoch": create_buf(32, 32),
            "reward": create_buf(1),
            "is_first_step": CircularBuffer(buffer_size, batch_size, (1,), device, buf_device, torch.bool),
        }

        self.datasets = {k: torch.empty((dataset_size, buffer_size, *buf.shape[2:]), dtype=buf.dtype, device=buf_device)
                         for k, buf in self.buffers.items()}
        self._cur_size = self._cur_idx = 0
        self.valid_len = torch.zeros(dataset_size, dtype=torch.long, device=device)

    def append(self, **kwargs):
        assert len(self.datasets) == len(kwargs)

        for k, v in kwargs.items():
            assert k in self.buffers
            self.buffers[k].append(v[self.envs_selected])

    def transfer_dones(self, dones, transfer_ids=None):
        if transfer_ids is None:
            transfer_ids = (dones[self.envs_selected] | (self.buffers['prop'].get_valid_len() == self.buffer_size)).nonzero(as_tuple=False).flatten()

        num_transfer = len(transfer_ids)

        if num_transfer == 0:
            return

        # normal transfer
        num_transfer_now = min(num_transfer, self.dataset_size - self._cur_idx)
        transfer_ids_now = transfer_ids[:num_transfer_now]

        valid_len_to_transfer = self.buffers["prop"].get_valid_len()[transfer_ids_now]
        self.valid_len[self._cur_idx: self._cur_idx + num_transfer_now] = valid_len_to_transfer

        for n, buf in self.buffers.items():
            data_to_transfer = buf.get_all()[:, transfer_ids_now].transpose(0, 1)
            self.datasets[n][self._cur_idx: self._cur_idx + num_transfer_now] = data_to_transfer
            buf.reset(transfer_ids_now)

        self._cur_idx = (self._cur_idx + num_transfer) % self.dataset_size
        self._cur_size = min(self._cur_size + num_transfer, self.dataset_size)

        # transfer remaining
        if num_transfer > num_transfer_now:
            self.transfer_dones(None, transfer_ids=transfer_ids[num_transfer_now:])

    def __len__(self):
        return self._cur_size

    def sample(self, batch_size=16, batch_length=64):
        prob = self.valid_len / len(self)
        batch_idx = torch.multinomial(prob, batch_size, replacement=False)

        batch_length = min(self.valid_len[batch_idx].min().item(), batch_length)
        if batch_length <= 1:
            return None

        sampled_data = {}
        batch_end_ids = [random.randint(batch_length, self.valid_len[idx].item()) for idx in batch_idx]

        for k in self.datasets:
            data_k = []

            for idx, end_idx in zip(batch_idx, batch_end_ids):
                data_k.append(self.datasets[k][idx, end_idx - batch_length: end_idx])

            sampled_data[k] = torch.stack(data_k, dim=0).transpose(0, 1).to(self.device)

        return sampled_data


class Transition:
    def __init__(self):
        self.observations = None
        self.critic_observations = None
        self.wm_feature = None
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


class PPO_WMP(BaseAlgorithm):
    def __init__(self, task_cfg, device, **kwargs):
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.symlog_mse = SymlogMSELoss()
        self.symexp_two_hot_loss = SymexpTwoHotLoss(low=-5., high=5.)

        # world model
        self.world_model = RSSM(task_cfg.env, task_cfg).to(self.device)
        self.optimizer_wm = torch.optim.Adam(self.world_model.parameters(), lr=1e-4)
        self.scaler_wm = GradScaler(enabled=self.cfg.use_amp)

        self.wm_dones = torch.ones(task_cfg.env.num_envs, dtype=torch.bool, device=self.device)
        """ env done in last {wm_step_interval} steps """
        self.wm_rew_sum = torch.zeros(task_cfg.env.num_envs, dtype=torch.float, device=self.device)
        """ reward sum in last steps """

        self.wm_action_his = CircularBuffer(task_cfg.world_model.step_interval, task_cfg.env.num_envs, (task_cfg.env.num_actions,), device=device)
        self.wm_dataset = WorldModelDataset(task_cfg, device=self.device)

        # PPO components
        self.actor = ActorWMP(task_cfg.env).to(self.device)
        self.critic = CriticWMP(task_cfg.env).to(self.device)

        # # AMP components
        # self.amp_loader = AMPLoader(
        #     device,
        #     time_between_frames=task_cfg.control.decimation * task_cfg.sim.dt,
        #     preload_transitions=True,
        #     num_preload_transitions=2000000,
        #     motion_files=['/home/harry/projects/parkour_genesis/legged_gym/robots/a1/mocap_motions/hop1.txt',
        #                   '/home/harry/projects/parkour_genesis/legged_gym/robots/a1/mocap_motions/hop2.txt',
        #                   '/home/harry/projects/parkour_genesis/legged_gym/robots/a1/mocap_motions/trot1.txt',
        #                   '/home/harry/projects/parkour_genesis/legged_gym/robots/a1/mocap_motions/trot2.txt', ]
        # )
        # self.amp_normalizer = Normalizer(30)
        # self.amp_discriminator = AMPDiscriminator(task_cfg.env, task_cfg).to(self.device)

        # storage and optimizer
        params = [
            {'params': [*self.actor.parameters(), *self.critic.parameters()], 'name': 'actor_critic'},
            # {'params': self.amp_discriminator.trunk.parameters(), 'weight_decay': 10e-4, 'name': 'amp_trunk'},
            # {'params': self.amp_discriminator.linear.parameters(), 'weight_decay': 10e-2, 'name': 'amp_head'}
        ]

        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.cfg.use_amp)
        self.transition = Transition()
        # self.transition_amp = Transition()
        self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, obs_critic, step_world_model=True, **kwargs):
        actor_obs, wm_obs = obs

        if step_world_model:
            state_deter = self.world_model.get_deter()
            state_stoch = self.world_model.get_stoch()

            act_his = self.wm_action_his.get_all().transpose(0, 1)
            self.world_model.step(wm_obs.proprio, wm_obs.depth, act_his, self.wm_dones)

            self.wm_dataset.append(
                prop=wm_obs.proprio,
                depth=wm_obs.depth,
                action_his=act_his,
                state_deter=state_deter,
                state_stoch=state_stoch,
                reward=self.wm_rew_sum.unsqueeze(1),
                is_first_step=self.wm_dones.unsqueeze(1),
            )
            self.wm_rew_sum[:] = 0.
            self.wm_dones[:] = False

        # store observations
        self.transition.observations = actor_obs
        self.transition.critic_observations = obs_critic
        wm_feature = self.world_model.get_feature()
        self.transition.wm_feature = wm_feature

        actions = self.actor.act(actor_obs, wm_feature=wm_feature)

        # store
        self.transition.actions = actions
        self.transition.values = self.critic.evaluate(obs_critic, wm_feature=wm_feature)
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions)
        self.transition.action_mean = self.actor.action_mean
        self.transition.action_sigma = self.actor.action_std

        self.wm_action_his.append(actions)

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

        self.wm_dataset.transfer_dones(dones)
        self.wm_action_his.reset(dones)
        self.wm_dones[dones] = True
        self.wm_rew_sum[:] += rewards
        self.wm_rew_sum[dones] = 0.

    def compute_returns(self, last_critic_obs):
        last_values = self.critic.evaluate(last_critic_obs, wm_feature=self.world_model.get_feature()).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, train_steps_per_iter=10, **kwargs):
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

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates

        # update estimation
        if len(self.wm_dataset) > 100:
            for _ in range(train_steps_per_iter):
                dyn_loss, rep_loss, loss_prop, loss_depth, loss_rew = self._update_world_model()

                mean_dyn_loss += dyn_loss
                mean_rep_loss += rep_loss
                mean_prop_loss += loss_prop
                mean_depth_loss += loss_depth
                mean_rew_loss += loss_rew

        mean_dyn_loss /= train_steps_per_iter
        mean_rep_loss /= train_steps_per_iter
        mean_prop_loss /= train_steps_per_iter
        mean_depth_loss /= train_steps_per_iter
        mean_rew_loss /= train_steps_per_iter

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
            obs_batch = batch['observations']
            critic_obs_batch = batch['critic_observations']
            wm_feature_batch = batch['wm_feature']
            actions_batch = batch['actions']
            target_values_batch = batch['values']
            advantages_batch = batch['advantages']
            returns_batch = batch['returns']
            old_mu_batch = batch['action_mean']
            old_sigma_batch = batch['action_sigma']
            old_actions_log_prob_batch = batch['actions_log_prob']

            self.actor.train_act(obs_batch, wm_feature=wm_feature_batch)

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
            evaluation = self.critic.evaluate(critic_obs_batch, wm_feature=wm_feature_batch)

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
        # self.scaler.unscale_(self.optimizer)
        # nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.cfg.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return value_loss.item(), surrogate_loss.item(), entropy_loss.item(), kl_mean

    def _update_world_model(self):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            data = self.wm_dataset.sample()

            if data is None:
                return 0., 0., 0., 0., 0.

            prop, depth, action_his, state_deter, state_stoch, reward, is_first_step = data.values()

            prior_digits, post_digits, prop_pred, depth_pred, rew_pred = self.world_model.train_step(
                prop, depth, action_his, state_deter, state_stoch, is_first_step)

            # KL Divergence
            loss, dyn_loss, rep_loss = self.compute_dyn_rep_loss(prior_digits, post_digits)

            # maximum likelihood
            loss_prop = self.symlog_mse(prop_pred, prop)
            loss_depth = self.mse_loss(depth_pred, depth)
            loss_rew = self.symexp_two_hot_loss(rew_pred, reward)

            loss += 1.0 * loss_prop + 1.0 * loss_depth + 1.0 * loss_rew

        # Gradient step
        self.optimizer_wm.zero_grad()
        self.scaler_wm.scale(loss).backward()
        self.scaler_wm.step(self.optimizer_wm)
        self.scaler_wm.update()

        return dyn_loss.item(), rep_loss.item(), loss_prop.item(), loss_depth.item(), loss_rew.item()

    def compute_dyn_rep_loss(self, prior, post, free_bits=1.0, dyn_scale=0.5, rep_scale=0.1):
        dist = self.world_model.get_dist

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
        assert 'dones' in kwargs
        dones = kwargs['dones']
        self.wm_action_his.reset(dones)
        self.wm_dones[dones] = True

        actor_obs, wm_obs = obs

        if step_world_model:
            act_his = self.wm_action_his.get_all().transpose(0, 1)
            predictions = self.world_model.play_step(wm_obs.proprio, wm_obs.depth, act_his, self.wm_dones, sample=False)
            self.wm_dones[:] = False

        actions = self.actor.act(actor_obs, wm_feature=self.world_model.get_feature(), **kwargs)
        self.wm_action_his.append(actions)

        if step_world_model:
            predictions['actions'] = actions
            return predictions
        else:
            return {'actions': actions}

    def train(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.world_model.load(loaded_dict['world_model_state_dict'])
        self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, device=self.device)

        return loaded_dict['infos']

    def save(self):
        return {
            'world_model_state_dict': self.world_model.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
