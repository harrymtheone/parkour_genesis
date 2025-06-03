import math

import torch
import torch.nn as nn

from legged_gym.envs.base.utils import CircularBuffer
from rsl_rl.modules.dreamer import Dreamer
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


class DreamerTransition:
    def __init__(self):
        self.prop = None
        self.depth = None
        self.prev_action = None
        self.state_deter = None
        self.state_stoch = None
        self.rewards = None
        self.dones = None

    def get_items(self):
        return self.__dict__.items()

    def clear(self):
        for key in self.__dict__:
            setattr(self, key, None)


class DreamerDataset:
    def __init__(self, task_cfg, device, buffer_len=32, dataset_size=10_000, dtype=torch.half, buf_device=torch.device('cuda')):
        self.device = device
        self.buffer_len = buffer_len
        self.dataset_size = dataset_size
        batch_size = task_cfg.env.num_envs

        def create_buf(*data_shape):
            return CircularBuffer(buffer_len, batch_size, data_shape, device, buf_device, dtype)

        self.buffers = {
            "prop": create_buf(task_cfg.env.n_proprio - 12),
            "depth": create_buf(1, 64, 64),
            "prev_action": create_buf(task_cfg.env.num_actions),
            "state_deter": create_buf(512),
            "state_stoch": create_buf(32, 32),
            "rewards": create_buf(1),
            "dones": CircularBuffer(buffer_len, batch_size, (1,), device, buf_device, torch.bool),
        }

        self.datasets = {k: torch.empty((dataset_size, buffer_len, *buf.shape[2:]), dtype=buf.dtype, device=buf_device)
                         for k, buf in self.buffers.items()}
        self._cur_size = self._cur_idx = 0
        self.valid_len = torch.zeros(dataset_size, dtype=torch.long, device=device)

    def add_transitions(self, transition: DreamerTransition):
        for k, v in transition.get_items():
            assert k in self.buffers, f"Key {k} not in buffers!"
            self.buffers[k].append(v)

    def transfer_dones(self, dones, transfer_ids=None):
        if transfer_ids is None:
            transfer_ids = (dones | (self.buffers['prop'].get_valid_len() == self.buffer_len)).nonzero(as_tuple=False).flatten()

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

    def sample(self, batch_size=16):
        batch_idx = torch.randperm(self._cur_size)[:batch_size]

        sampled_data = {k: self.datasets[k][batch_idx].transpose(0, 1)
                        for k in self.datasets}

        mask = torch.arange(self.buffer_len, device=self.device).unsqueeze(1) < self.valid_len[batch_idx]
        sampled_data['mask'] = mask

        return sampled_data


class PPO_Dreamer(BaseAlgorithm):
    def __init__(self, task_cfg, device, **kwargs):
        self.cfg = task_cfg.algorithm
        self.learning_rate = self.cfg.learning_rate
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.symlog_mse = SymlogMSELoss()
        self.symexp_two_hot_loss = SymexpTwoHotLoss(low=-5., high=5.)

        # world model
        self.dreamer = Dreamer(task_cfg.env, task_cfg).to(self.device)

        self.optimizer_dreamer = torch.optim.Adam(
            [*self.dreamer.rssm.parameters(), *self.dreamer.decoder.parameters()],
            lr=1e-4
        )
        self.scaler_dreamer = GradScaler(enabled=self.cfg.use_amp)
        self.optimizer_ac = torch.optim.Adam(
            [*self.dreamer.actor.parameters(), *self.dreamer.critic.parameters()],
            lr=1e-4
        )
        self.scaler_ac = GradScaler(enabled=self.cfg.use_amp)

        self.transition_dreamer = DreamerTransition()
        self.dataset_dreamer = DreamerDataset(task_cfg, device=self.device)

        # PPO components
        self.prev_actions = torch.zeros(task_cfg.env.num_envs, task_cfg.env.num_actions,
                                        dtype=torch.float, device=device, requires_grad=False)

        # # storage and optimizer
        # self.optimizer = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        # self.scaler = GradScaler(enabled=self.cfg.use_amp)
        # self.transition = Transition()
        # self.storage = RolloutStorage(task_cfg.env.num_envs, task_cfg.runner.num_steps_per_env, self.device)

    def act(self, obs, obs_critic, **kwargs):
        actor_obs, wm_obs = obs

        self.transition_dreamer.prop = wm_obs.proprio
        self.transition_dreamer.depth = wm_obs.depth
        self.transition_dreamer.prev_action = self.prev_actions
        self.transition_dreamer.state_deter = self.dreamer.get_deter()
        self.transition_dreamer.state_stoch = self.dreamer.get_stoch()

        actions = self.dreamer.step(wm_obs.proprio, wm_obs.depth, self.prev_actions)
        self.prev_actions[:] = actions
        return actions

    def process_env_step(self, rewards, dones, infos, *args):
        self.transition_dreamer.rewards = rewards.clone().unsqueeze(1)
        self.transition_dreamer.dones = dones.unsqueeze(1)

        # Record the transition
        self.dataset_dreamer.add_transitions(self.transition_dreamer)
        self.dataset_dreamer.transfer_dones(dones)
        self.transition_dreamer.clear()

        self.dreamer.reset(dones)

    def compute_returns(self, last_critic_obs):
        pass

    def update(self, train_steps_per_iter=10, **kwargs):
        if len(self.dataset_dreamer) < 1000:
            return {}

        mean_dyn_loss = 0
        mean_rep_loss = 0
        mean_prop_loss = 0
        mean_depth_loss = 0
        mean_rew_loss = 0
        mean_term_loss = 0

        for _ in range(train_steps_per_iter):
            dyn_loss, rep_loss, loss_prop, loss_depth, loss_rew, loss_term = self._update_world_model()

            mean_dyn_loss += dyn_loss
            mean_rep_loss += rep_loss
            mean_prop_loss += loss_prop
            mean_depth_loss += loss_depth
            mean_rew_loss += loss_rew
            mean_term_loss += loss_term

        mean_dyn_loss /= train_steps_per_iter
        mean_rep_loss /= train_steps_per_iter
        mean_prop_loss /= train_steps_per_iter
        mean_depth_loss /= train_steps_per_iter
        mean_rew_loss /= train_steps_per_iter
        mean_term_loss /= train_steps_per_iter

        self._update_actor_critic()

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
            'Loss/term_loss': mean_term_loss,
        }

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0

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
        if len(self.dataset_dreamer) > 1000:
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

    def _update_world_model(self):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            data = self.dataset_dreamer.sample()

            if data is None:
                return 0., 0., 0., 0., 0.

            prop, depth, prev_action, state_deter, state_stoch, reward, dones, mask = data.values()
            is_first_step = torch.zeros_like(dones)
            is_first_step[1:] = dones[:-1]

            prior_digits, post_digits, prop_pred, depth_pred, rew_pred, term_pred = self.dreamer.train_step(
                prop, depth, prev_action, state_deter, state_stoch, is_first_step)

            # KL Divergence
            loss, dyn_loss, rep_loss = self.compute_dyn_rep_loss(prior_digits[mask], post_digits[mask])

            # maximum likelihood
            loss_prop = self.symlog_mse(prop_pred[mask], prop[mask])
            loss_depth = self.mse_loss(depth_pred[mask], depth[mask])
            loss_rew = self.symexp_two_hot_loss(rew_pred[mask], reward[mask])
            loss_term = self.ce_loss(term_pred[mask], dones[mask].squeeze(1).long())  # TODO: do we need softmax first?

            loss += 1.0 * loss_prop + 1.0 * loss_depth + 1.0 * loss_rew + 1.0 * loss_term

        # Gradient step
        self.optimizer_dreamer.zero_grad()
        self.scaler_dreamer.scale(loss).backward()
        self.scaler_dreamer.step(self.optimizer_dreamer)
        self.scaler_dreamer.update()

        return dyn_loss.item(), rep_loss.item(), loss_prop.item(), loss_depth.item(), loss_rew.item()

    def compute_dyn_rep_loss(self, prior, post, free_bits=1.0, dyn_scale=0.5, rep_scale=0.1):
        dist = self.dreamer.rssm.get_dist

        dyn_loss = torch.distributions.kl.kl_divergence(
            dist(post.detach()), dist(prior))

        rep_loss = torch.distributions.kl.kl_divergence(
            dist(post), dist(prior.detach()))

        # this is implemented using maximum at the original repo as the gradients are not back-propagated for the out of limits.
        dyn_loss = torch.clip(dyn_loss, min=free_bits).mean()
        rep_loss = torch.clip(rep_loss, min=free_bits).mean()
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, dyn_loss, rep_loss

    def _update_actor_critic(self):
        # ######################## Rollout ########################
        pass

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
        self.dreamer.train()

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
            'dreamer_state_dict': self.dreamer.state_dict(),
            'optimizer_dreamer_state_dict': self.optimizer_dreamer.state_dict(),
            'optimizer_ac_state_dict': self.optimizer_ac.state_dict(),
        }
