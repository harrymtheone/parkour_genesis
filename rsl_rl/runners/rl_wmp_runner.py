import collections
import os
import statistics
import time

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


class RL_WMP_Runner(RunnerLogger):
    def __init__(self, task_cfg, log_dir=None, device=torch.device('cpu')):
        self.task_cfg = task_cfg
        self.cfg = task_cfg.runner
        self.log_dir = log_dir
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
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    step_world_model = self.tot_steps % 5 == 0
                    actions = self.alg.act(obs, critic_obs, step_world_model=step_world_model)
                    obs, critic_obs, rewards, dones, infos = env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    if self.log_dir is not None:
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

                            # # Do AdaSmpl for envs reset
                            if self.cur_it > 2000:
                                if self.cur_it == 2000:
                                    self.p_smpl = 1.0

                                use_estimated_values[new_ids] = torch.rand(len(new_ids[0]), device=self.device) > self.p_smpl

                            cur_reward_sum[new_ids] = 0.
                            cur_episode_length[new_ids] = 0.

                    self.tot_steps += 1

                # update AdaSmpl coefficient
                if self.cur_it - self.start_it > 20:  # ensure there are enough samples
                    for i, (t, n) in enumerate(zip(terrain_class, terrain_class_name)):
                        rew_terrain = last_env_reward[env.env_class == t]
                        coefficient_variation[i] = rew_terrain.std() / (rew_terrain.mean().abs() + 1e-5)
                        terrain_coefficient_variation[f'Coefficient Variation/{n}'] = coefficient_variation[i].item()

                    # probability to use ground truth value
                    self.p_smpl = 0.999 * self.p_smpl + 0.001 * torch.tanh((coefficient_variation * terrain_env_counts).sum() / terrain_env_counts.sum()).item()

                # Learning step
                self.alg.compute_returns(critic_obs)

            torch.cuda.synchronize()
            self.collection_time = time.time() - start_time
            start_time = time.time()

            update_info = self.alg.update(cur_it=self.cur_it)
            torch.cuda.synchronize()
            self.learn_time = time.time() - start_time

            env.update_reward_curriculum(self.cur_it)

            if self.log_dir is not None:
                self.log(update_info)

            if self.cur_it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{self.cur_it}.pt'))
            self.save(os.path.join(self.log_dir, 'latest.pt'))

    def log(self, update_info, width=80, pad=35):
        iteration_time = self.collection_time + self.learn_time
        self.tot_time += iteration_time

        # construct wandb logging dict
        wandb_dict = {}

        # logging episode reward
        ep_rew = self.episode_rew
        for rew_name in ep_rew[0]:
            rew_tensor = [ep[rew_name] for ep in ep_rew]
            rew_tensor = torch.stack(rew_tensor, dim=0)
            wandb_dict['Episode_rew/' + rew_name] = torch.mean(rew_tensor).item()

        # logging episode average terrain level
        ep_terrain_level = self.episode_terrain_level
        if len(ep_terrain_level) > 0:
            for terrain_name in ep_terrain_level[0]:
                level_tensor = [ep[terrain_name] for ep in ep_terrain_level]
                level_tensor = torch.stack(level_tensor, dim=0)
                wandb_dict['Terrain Level/' + terrain_name] = torch.mean(level_tensor).item()

        wandb_dict.update(self.terrain_coefficient_variation)

        # logging update information
        wandb_dict.update(update_info)

        if len(self.episode_rew_sum) > 10:
            wandb_dict['Train/mean_reward'] = statistics.mean(self.episode_rew_sum)  # use the latest 100 to compute
            wandb_dict['Train/mean_episode_length'] = statistics.mean(self.episode_length)
        wandb_dict['Train/base_height'] = self.mean_base_height.mean().item()
        wandb_dict['Train/AdaSmpl'] = self.p_smpl

        wandb.log(wandb_dict, step=self.cur_it)

        # logging string to print
        progress = f" \033[1m Learning iteration {self.cur_it}/{self.start_it + self.cfg.max_iterations} \033[0m "
        fps = int(self.num_steps_per_env * self.task_cfg.env.num_envs / iteration_time)
        curr_it = self.cur_it - self.start_it
        eta = self.tot_time / (curr_it + 1) * (self.cfg.max_iterations - curr_it)
        log_string = (
            f"""{'*' * width}\n"""
            f"""{progress.center(width, ' ')}\n\n"""
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
