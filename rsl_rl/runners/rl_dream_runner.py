import collections
import os
import statistics
import time
from pathlib import Path
import torch
import wandb

from legged_gym.utils.terrain import Terrain
from rsl_rl.algorithms import BaseAlgorithm, algorithm_dict


class RunnerLogger:
    collection_time = -1
    learn_time = -1
    episode_rew = []
    episode_terrain_level = []
    episode_rew_sum = collections.deque(maxlen=100)
    episode_length = collections.deque(maxlen=100)

    mean_base_height = None
    terrain_coefficient_variation = {}
    p_smpl = 1.


class RLDreamRunner(RunnerLogger):
    def __init__(self, task_cfg, model_dir, exptid, device=torch.device('cpu')):
        self.task_cfg = task_cfg
        self.cfg = task_cfg.runner
        self.exptid = exptid

        self.model_dir = model_dir
        if self.cfg.logger_backend == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_file = list(Path(self.model_dir).glob('events.*'))
            if tensorboard_file:
                raise FileExistsError("Logging directory not empty!")

            self.logger = SummaryWriter(log_dir=model_dir)  # NOQA

        self.device = torch.device(device) if type(device) is str else device

        # Create algorithm
        self.alg: BaseAlgorithm = algorithm_dict[task_cfg.runner.algorithm_name](self.task_cfg, device=self.device)

        self.num_steps_per_env = self.cfg.num_steps_per_env
        self.save_interval = self.cfg.save_interval

        # Log
        self.tot_steps = 0
        self.tot_time = 0
        self.start_it = 0
        self.cur_it = 0

    def learn(self, env, init_at_random_ep_len=True):
        if init_at_random_ep_len:
            env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))

        self.alg.train()  # switch to train mode (for dropout for example)
        obs, critic_obs = env.get_observations(), env.get_critic_observations()

        # statistics
        n_envs = self.task_cfg.env.num_envs
        cur_reward_sum = torch.zeros(n_envs, device=self.device)
        cur_episode_length = torch.zeros(n_envs, device=self.device)
        last_env_reward = torch.zeros(n_envs, device=self.device)
        self.mean_base_height = self.task_cfg.rewards.base_height_target + torch.zeros(n_envs, device=self.device)

        # AdaSmpl for each terrain type
        terrain_class, terrain_env_counts = torch.unique(env.env_class, return_counts=True)
        terrain_class_name = [Terrain.terrain_type(tc.item()).name for tc in terrain_class]
        coefficient_variation = torch.ones_like(terrain_class)
        terrain_coefficient_variation = {}

        # adaptive sampling probability (prob to use ground truth)
        use_estimated_values = torch.zeros(n_envs, dtype=torch.bool, device=self.device)

        for self.cur_it in range(self.start_it, self.start_it + self.cfg.max_iterations):
            start_time = time.time()
            self.episode_rew.clear()
            self.episode_terrain_level.clear()

            # Rollout
            with torch.inference_mode(mode=self.cfg.inference_enabled):
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, use_estimated_values=use_estimated_values.unsqueeze(1))
                    obs, critic_obs, rewards, dones, infos = env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    if 'episode_rew' in infos:
                        self.episode_rew.append(infos['episode_rew'])

                    if 'episode_terrain_level' in infos:
                        self.episode_terrain_level.append(infos['episode_terrain_level'])

                    cur_reward_sum[:] += rewards
                    cur_episode_length[:] += 1

                    self.mean_base_height[:] = 0.99 * self.mean_base_height + 0.01 * env.base_height
                    new_ids = torch.where(dones > 0)
                    if len(new_ids[0]) > 0:
                        last_env_reward[new_ids] = cur_reward_sum[new_ids]
                        self.episode_rew_sum.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                        self.episode_length.extend(cur_episode_length[new_ids].cpu().numpy().tolist())

                        use_estimated_values[new_ids] = torch.rand(len(new_ids[0]), device=self.device) > self.p_smpl

                        cur_reward_sum[new_ids] = 0.
                        cur_episode_length[new_ids] = 0.

                # update AdaSmpl coefficient
                if self.cur_it - self.start_it > 20:  # ensure there are enough samples
                    for i, (t, n) in enumerate(zip(terrain_class, terrain_class_name)):
                        rew_terrain = last_env_reward[env.env_class == t]
                        coefficient_variation[i] = rew_terrain.std() / (rew_terrain.mean().abs() + 1e-5)
                        terrain_coefficient_variation[f'Coefficient Variation/{n}'] = coefficient_variation[i].item()

                    # probability to use ground truth value
                    if self.cur_it > self.cfg.lock_smpl_until:
                        self.p_smpl = 0.999 * self.p_smpl + 0.001 * torch.tanh(
                            (coefficient_variation * terrain_env_counts).sum() / terrain_env_counts.sum()).item()

                # Learning step
                self.alg.compute_returns(critic_obs)

            torch.cuda.synchronize()
            self.collection_time = time.time() - start_time
            start_time = time.time()

            update_info = self.alg.update(cur_it=self.cur_it)
            torch.cuda.synchronize()
            self.learn_time = time.time() - start_time

            env.update_reward_curriculum(self.cur_it)

            self.log(update_info)

            if self.cur_it % self.save_interval == 0:
                self.save(os.path.join(self.model_dir, f'model_{self.cur_it}.pt'))
            self.save(os.path.join(self.model_dir, 'latest.pt'))

    def log(self, update_info, width=80, pad=35):
        self.tot_steps += self.num_steps_per_env * self.task_cfg.env.num_envs
        iteration_time = self.collection_time + self.learn_time
        self.tot_time += iteration_time

        # construct wandb logging dict
        logger_dict = {}

        # logging episode reward
        ep_rew = self.episode_rew
        for rew_name in ep_rew[0]:
            rew_tensor = [ep[rew_name] for ep in ep_rew]
            rew_tensor = torch.stack(rew_tensor, dim=0)
            logger_dict['Episode_rew/' + rew_name] = torch.mean(rew_tensor).item()

        # logging episode average terrain level
        ep_terrain_level = self.episode_terrain_level
        if len(ep_terrain_level) > 0:
            for terrain_name in ep_terrain_level[0]:
                level_tensor = [ep[terrain_name] for ep in ep_terrain_level]
                level_tensor = torch.stack(level_tensor, dim=0)
                logger_dict['Terrain Level/' + terrain_name] = torch.mean(level_tensor).item()

        logger_dict.update(self.terrain_coefficient_variation)

        # logging update information
        logger_dict.update(update_info)

        if len(self.episode_rew_sum) > 10:
            logger_dict['Train/mean_reward'] = statistics.mean(self.episode_rew_sum)  # use the latest 100 to compute
            logger_dict['Train/mean_episode_length'] = statistics.mean(self.episode_length)
        logger_dict['Train/base_height'] = self.mean_base_height.mean().item()
        logger_dict['Train/AdaSmpl'] = self.p_smpl

        if self.cfg.logger_backend == 'wandb':
            wandb.log(logger_dict, step=self.cur_it)
        elif self.cfg.logger_backend == 'tensorboard':
            for t, v in logger_dict.items():
                self.logger.add_scalar(t, v, global_step=self.cur_it)
            self.logger.flush()

        # logging string to print
        progress = f" \033[1m Learning iteration {self.cur_it}/{self.start_it + self.cfg.max_iterations} \033[0m "
        fps = int(self.num_steps_per_env * self.task_cfg.env.num_envs / iteration_time)
        curr_it = self.cur_it - self.start_it
        eta = self.tot_time / (curr_it + 1) * (self.cfg.max_iterations - curr_it)
        log_string = (
            f"""{'*' * width}\n"""
            f"""{progress.center(width, ' ')}\n\n"""
            f"""{'Experiment:':>{pad}} {self.exptid}\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_steps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s {self.collection_time:.2f}s {self.learn_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {eta // 60:.0f} mins {eta % 60:.1f} s\n"""
            f"""{'CUDA allocated:':>{pad}} {torch.cuda.memory_allocated() / 1024 / 1024:.2f}\n"""
            f"""{'CUDA reserved:':>{pad}} {torch.cuda.memory_reserved() / 1024 / 1024:.2f}\n"""
        )
        print(log_string)

    def play_act(self, obs, **kwargs):
        self.alg.actor.eval()
        return self.alg.play_act(obs, **kwargs)

    def save(self, path, infos=None):
        state_dict = self.alg.save()
        state_dict['iter'] = self.cur_it
        state_dict['infos'] = infos
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from", path)

        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.start_it = loaded_dict['iter']
        infos = self.alg.load(loaded_dict, load_optimizer)

        print("*" * 80)
        return infos
