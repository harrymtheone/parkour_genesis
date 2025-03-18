import collections
import os
import statistics
import time

import torch
import wandb

from rsl_rl.algorithms import BaseAlgorithm, algorithm_dict


def linear_change(start, end, span, start_it, cur_it):
    cur_value = start + (end - start) * (cur_it - start_it) / span
    cur_value = max(cur_value, min(start, end))
    cur_value = min(cur_value, max(start, end))
    return cur_value


class RLDreamRunner:
    from legged_gym.envs.base.humanoid_env import HumanoidEnv

    def __init__(self, env: HumanoidEnv, train_cfg, log_dir=None, device=torch.device('cpu')):
        self.cfg = train_cfg.runner
        self.log_dir = log_dir
        self.device = torch.device(device) if type(device) is str else device

        self.env = env
        self.env_cfg = self.env.cfg.env

        # Create algorithm
        self.alg: BaseAlgorithm = algorithm_dict[train_cfg.algorithm_name](self.env_cfg, train_cfg, device=self.device, env=env)

        self.num_steps_per_env = self.cfg.num_steps_per_env
        self.save_interval = self.cfg.save_interval

        # Log
        self.tot_steps = 0
        self.tot_time = 0
        self.start_it = 0
        self.cur_it = 0

    def learn(self, init_at_random_ep_len=True):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        self.alg.train()  # switch to train mode (for dropout for example)
        obs, critic_obs = self.env.get_observations(), self.env.get_critic_observations()

        # statistics
        infos = {}
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)
        last_env_reward = torch.zeros(self.env.num_envs, device=self.device)
        mean_base_height = self.env.cfg.rewards.base_height_target + torch.zeros(self.env.num_envs, device=self.device)
        episode_rew_sum = collections.deque(maxlen=100)
        episode_length = collections.deque(maxlen=100)

        # AdaSmpl for each terrain type
        terrain_class, terrain_env_counts = torch.unique(self.env.env_class, return_counts=True)
        coefficient_variation = torch.ones_like(terrain_class)

        # adaptive sampling probability (prob to use ground truth)
        p_smpl = 1.0
        use_estimated_values = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        for self.cur_it in range(self.start_it, self.start_it + self.cfg.max_iterations):
            start = time.time()
            ep_infos = {'episode_rew': [], 'episode_terrain_level': []}

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, use_estimated_values=use_estimated_values.unsqueeze(1))
                    obs, critic_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    if self.log_dir is not None:
                        if 'episode_rew' in infos:
                            ep_infos['episode_rew'].append(infos['episode_rew'])

                        if 'episode_terrain_level' in infos:
                            ep_infos['episode_terrain_level'].append(infos['episode_terrain_level'])

                        cur_reward_sum[:] += rewards
                        cur_episode_length[:] += 1

                        mean_base_height[:] = 0.99 * mean_base_height + 0.01 * self.env.base_height
                        new_ids = torch.where(dones > 0)
                        if len(new_ids[0]) > 0:
                            last_env_reward[new_ids] = cur_reward_sum[new_ids]
                            episode_rew_sum.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                            episode_length.extend(cur_episode_length[new_ids].cpu().numpy().tolist())

                            # Do AdaSmpl for envs reset
                            if self.cur_it > 10000:
                                use_estimated_values[new_ids] = torch.rand(len(new_ids[0]), device=self.device) > p_smpl

                            cur_reward_sum[new_ids] = 0.
                            cur_episode_length[new_ids] = 0.

                # update AdaSmpl coefficient
                if self.cur_it - self.start_it > 20:  # ensure there are enough samples
                    for i, t in enumerate(terrain_class):
                        rew_terrain = last_env_reward[self.env.env_class == t]
                        coefficient_variation[i] = rew_terrain.std() / (rew_terrain.mean().abs() + 1e-5)

                    # probability to use ground truth value
                    p_smpl = 0.999 * p_smpl + 0.001 * torch.tanh((coefficient_variation * terrain_env_counts).sum() / terrain_env_counts.sum()).item()

                # Learning step
                self.alg.compute_returns(critic_obs)

            torch.cuda.synchronize()
            collection_time = time.time() - start
            start = time.time()

            update_info = self.alg.update(cur_it=self.cur_it)
            torch.cuda.synchronize()
            learn_time = time.time() - start

            # self.env.update_phase_enabled(linear_change(1., 0., 2000, 0, self.cur_it))

            # self.env.reward_scales['orientation'] = linear_change(-1.0 * 3, -1.0, 1000, 500, self.cur_it)
            # self.env.reward_scales['base_height'] = linear_change(-1.0 * 3, -1.0, 1000, 500, self.cur_it)
            # self.env.reward_scales['dof_error'] = linear_change(-0.04 * 10, -0.04, 1000, 500, self.cur_it)

            self.env.update_reward_curriculum(self.cur_it)

            # if self.cur_it > 5000:
            #     self.env.cfg.domain_rand.push_robots = True

            if self.log_dir is not None:
                self.log(locals())

            if self.cur_it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{self.cur_it}.pt'))
            self.save(os.path.join(self.log_dir, 'latest.pt'))

    def log(self, locs, width=80, pad=35):
        self.tot_steps += self.num_steps_per_env * self.env.num_envs
        iteration_time = locs['collection_time'] + locs['learn_time']
        self.tot_time += iteration_time

        # construct wandb logging dict
        wandb_dict = {}

        # logging episode reward
        ep_rew = locs['ep_infos']['episode_rew']
        for rew_name in ep_rew[0]:
            rew_tensor = [ep[rew_name] for ep in ep_rew]
            rew_tensor = torch.stack(rew_tensor, dim=0)
            wandb_dict['Episode_rew/' + rew_name] = torch.mean(rew_tensor).item()

        # logging episode average terrain level
        ep_terrain_level = locs['ep_infos']['episode_terrain_level']
        if len(ep_terrain_level) > 0:
            for terrain_name in ep_terrain_level[0]:
                level_tensor = [ep[terrain_name] for ep in ep_terrain_level]
                level_tensor = torch.stack(level_tensor, dim=0)
                wandb_dict['Terrain Level/' + terrain_name] = torch.mean(level_tensor).item()

        # logging update information
        wandb_dict.update(locs['update_info'])

        if len(locs['episode_rew_sum']) > 10:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['episode_rew_sum'])  # use the latest 100 to compute
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['episode_length'])
        wandb_dict['Train/base_height'] = locs['mean_base_height'].mean().item()
        wandb_dict['Train/AdaSmpl'] = locs['p_smpl']

        wandb.log(wandb_dict, step=self.cur_it)

        # logging string to print
        progress = f" \033[1m Learning iteration {self.cur_it}/{self.start_it + self.cfg.max_iterations} \033[0m "
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
        curr_it = self.cur_it - self.start_it
        eta = self.tot_time / (curr_it + 1) * (self.cfg.max_iterations - curr_it)
        coefficient_variation = ", ".join([f"{x:.2f}" for x in locs['coefficient_variation'].tolist()])
        log_string = (
            f"""{'*' * width}\n"""
            f"""{progress.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_steps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s {locs['collection_time']:.2f}s {locs['learn_time']:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {eta // 60:.0f} mins {eta % 60:.1f} s\n"""
            f"""{'Coefficient Variance:':>{pad}} [{coefficient_variation}]\n"""
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
