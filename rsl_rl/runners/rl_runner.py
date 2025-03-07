import os
import statistics
import time
from collections import deque

import torch
import wandb

from rsl_rl.algorithms import algorithm_dict


def linear_change(start, end, span, start_it, cur_it):
    cur_value = start + (end - start) * (cur_it - start_it) / span
    cur_value = max(cur_value, min(start, end))
    cur_value = min(cur_value, max(start, end))
    return cur_value


class RLRunner:
    from legged_gym.envs.pdd.pdd_scan_environment import PddScanEnvironment

    def __init__(self, env: PddScanEnvironment, train_cfg, log_dir=None, device='cpu'):
        self.alg_cfg = train_cfg.algorithm
        self.cfg = train_cfg.runner
        self.log_dir = log_dir
        self.device = device

        self.env = env

        # Create algorithm
        self.alg = algorithm_dict[train_cfg.algorithm_name](self.env.cfg.env, train_cfg, device=self.device)

        self.num_steps_per_env = self.cfg.num_steps_per_env
        self.save_interval = self.cfg.save_interval

        # Log
        self.tot_timesteps = 0
        self.tot_time = 0
        self.start_it = 0
        self.cur_it = 0

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs, critic_obs = self.env.get_observations(), self.env.get_critic_observations()
        infos = {}
        self.alg.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        base_height_buffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        for self.cur_it in range(self.start_it, self.start_it + num_learning_iterations):
            start = time.time()

            ep_infos = {'episode_rew': [], 'episode_terrain_level': []}

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, critic_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    total_rew = self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if 'episode_rew' in infos:
                            ep_infos['episode_rew'].append(infos['episode_rew'])

                        if 'episode_terrain_level' in infos:
                            ep_infos['episode_terrain_level'].append(infos['episode_terrain_level'])

                        cur_reward_sum += total_rew
                        cur_episode_length += 1
                        base_height_buffer.append(torch.mean(self.env.base_height).cpu().numpy().tolist())

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, kl_div = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            # if self.env.cfg.rewards.only_positive_rewards:
            #     self.env.only_positive_rewards = self.cur_it < 3000

            if self.cur_it > 5000:
                self.env.cfg.domain_rand.push_robots = True

            # self.env.reward_scales['base_height'] = linear_change(-1.0 * 25, -1.0, 1000, 500, self.cur_it)
            # self.env.reward_scales['dof_error'] = linear_change(-0.04 * 10, -0.04, 1000, 500, self.cur_it)
            # self.env.reward_scales['feet_air_time'] = linear_change(0., 1.0, 500, 1000, self.cur_it)
            # self.env.reward_scales['feet_clearance'] = linear_change(0., 0.1, 500, 1000, self.cur_it)

            if self.log_dir is not None:
                self.log(locals())

            self.save(os.path.join(self.log_dir, 'latest.pt'))
            if self.cur_it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{self.cur_it}.pt'))

        self.save(os.path.join(self.log_dir, f'model_{self.start_it + num_learning_iterations}.pt'))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        wandb_dict = {}

        ep_rew = locs['ep_infos']['episode_rew']
        for rew_name in ep_rew[0]:
            rew_tensor = torch.tensor([], device=self.device)

            for ep in ep_rew:
                # handle scalar and zero dimensional tensor infos
                if not isinstance(ep[rew_name], torch.Tensor):
                    ep[rew_name] = torch.Tensor([ep[rew_name]])

                if len(ep[rew_name].shape) == 0:
                    ep[rew_name] = ep[rew_name].unsqueeze(0)

                rew_tensor = torch.cat((rew_tensor, ep[rew_name].to(self.device)))

            value = torch.mean(rew_tensor)
            wandb_dict['Episode_rew/' + rew_name] = value

        ep_terrain_level = locs['ep_infos']['episode_terrain_level']
        if self.env.curriculum:
            for terrain_name in ep_terrain_level[0]:
                level_tensor = torch.tensor([], device=self.device)

                for ep in ep_terrain_level:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep[terrain_name], torch.Tensor):
                        ep[terrain_name] = torch.Tensor([ep[terrain_name]])

                    if len(ep[terrain_name].shape) == 0:
                        ep[terrain_name] = ep[terrain_name].unsqueeze(0)

                    level_tensor = torch.cat((level_tensor, ep[terrain_name].to(self.device)))

                value = torch.mean(level_tensor)
                wandb_dict['Terrain Level/' + terrain_name] = value

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/kl_div'] = locs['kl_div']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate

        wandb_dict['Policy/mean_noise_std'] = self.alg.actor_critic.std.mean().item()

        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward']
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            wandb_dict['Train/base_height'] = statistics.mean(locs['base_height_buffer'])

        wandb.log(wandb_dict, step=self.cur_it)

        progress = f" \033[1m Learning iteration {self.cur_it}/{self.start_it + locs['num_learning_iterations']} \033[0m "
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        curr_it = self.cur_it - self.start_it
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string = (f"""{'*' * width}\n"""
                      f"""{progress.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                      f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                      f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                      f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def play_act(self, obs, **kwargs):
        return self.alg.actor_critic.act_inference(obs)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.cur_it,
            'infos': infos,
        }
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from", path)

        if type(path) is list:
            loaded_dict_list = [torch.load(pth, map_location=self.device, weights_only=True) for pth in path]
            self.start_it = loaded_dict_list[0]['iter']
            infos = self.alg.load(loaded_dict_list, load_optimizer)
        else:
            loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.start_it = loaded_dict['iter']
            infos = self.alg.load(loaded_dict, load_optimizer)

        print("*" * 80)
        return infos
