from tqdm import tqdm

try:
    import isaacgym, torch  # NOQA
except ImportError:
    import torch

import datetime
import os

from legged_gym.simulator import SimulatorType
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import TaskRegistry


def play(args):
    log_root = 'logs'
    args.simulator = SimulatorType.IsaacGym
    args.headless = True
    args.resume = True

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=args.task)

    # override some parameters for testing
    task_cfg.play.control = False
    task_cfg.env.num_envs = 1024
    task_cfg.terrain.num_rows = 5
    task_cfg.terrain.max_init_terrain_level = task_cfg.terrain.num_rows - 1
    task_cfg.terrain.curriculum = False
    # task_cfg.terrain.max_difficulty = True

    # task_cfg.depth.position_range = [(-0.01, 0.01), (-0., 0.), (-0.0, 0.01)]  # front camera
    # task_cfg.depth.position_range = [(-0., 0.), (-0, 0), (-0., 0.)]  # front camera
    # task_cfg.depth.angle_range = [-1, 1]
    task_cfg.domain_rand.push_robots = True
    task_cfg.domain_rand.push_interval_s = 6
    task_cfg.domain_rand.push_duration = [0.1]
    task_cfg.domain_rand.action_delay = True
    task_cfg.domain_rand.action_delay_range = [(2, 2)]
    task_cfg.domain_rand.add_dof_lag = True
    task_cfg.domain_rand.dof_lag_range = (3, 3)
    task_cfg.domain_rand.randomize_torques = False
    task_cfg.domain_rand.randomize_gains = False

    task_cfg.terrain.terrain_dict = {
        'smooth_slope': 1,
        'rough_slope': 1,
        'stairs_up': 1,
        'stairs_down': 1,
        'discrete': 0,
        'stepping_stone': 0,
        'gap': 0,
        'pit': 0,
        'parkour': 0,
        'parkour_gap': 0,
        'parkour_box': 0,
        'parkour_step': 0,
        'parkour_stair': 1,
        'parkour_mini_stair': 1,
        'parkour_flat': 0,
    }
    task_cfg.terrain.num_cols = sum(task_cfg.terrain.terrain_dict.values()) * 5

    # prepare environment
    args.n_rendered_envs = task_cfg.env.num_envs
    task_cfg = task_registry.get_cfg(name=args.task)
    env = task_registry.make_env(args=args, task_cfg=task_cfg)
    obs, obs_critic = env.get_observations(), env.get_critic_observations()
    dones = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)

    # load policy
    env.sim.clear_lines = True
    task_cfg.runner.resume = True
    task_cfg.runner.logger_backend = None
    runner = task_registry.make_alg_runner(task_cfg, args, log_root)

    # dataset for odometry estimation
    dataset_capacity = 1000
    data_length = 500

    buffer = {
        'prop': torch.zeros(task_cfg.env.num_envs, data_length, task_cfg.env.n_proprio, dtype=torch.half, device=env.device),
        'depth': torch.zeros(task_cfg.env.num_envs, data_length, 1, 64, 64, dtype=torch.half, device=env.device),
        'recon': torch.zeros(task_cfg.env.num_envs, data_length, 2, *task_cfg.env.scan_shape, dtype=torch.half, device=env.device),
        'priv': torch.zeros(task_cfg.env.num_envs, data_length, task_cfg.policy.estimator_output_dim, dtype=torch.half, device=env.device),
        'masks': torch.zeros(task_cfg.env.num_envs, data_length, dtype=torch.bool, device=env.device),
    }
    buffer_ptr = torch.zeros(task_cfg.env.num_envs, dtype=torch.long, device=env.device)

    dataset = {
        'prop': torch.empty(dataset_capacity, data_length, task_cfg.env.n_proprio, dtype=torch.half, device='cpu'),
        'depth': torch.empty(dataset_capacity, data_length, 1, 64, 64, dtype=torch.half, device='cpu'),
        'recon': torch.empty(dataset_capacity, data_length, 2, *task_cfg.env.scan_shape, dtype=torch.half, device='cpu'),
        'priv': torch.empty(dataset_capacity, data_length, task_cfg.policy.estimator_output_dim, dtype=torch.half, device='cpu'),
        'masks': torch.empty(dataset_capacity, data_length, dtype=torch.bool, device='cpu'),
    }
    dataset_idx = 0

    while True:
        # Store data in buffer at current pointer
        buffer['prop'][:, buffer_ptr] = obs.proprio.half()
        buffer['depth'][:, buffer_ptr] = obs.depth.half()
        buffer['recon'][:, buffer_ptr] = obs.scan.half()
        buffer['priv'][:, buffer_ptr] = obs.priv_actor.half()
        buffer['masks'][:, buffer_ptr] = True
        buffer_ptr[:] += 1

        rtn = runner.play_act(obs, use_estimated_values=False, eval_=True, dones=dones)
        obs, obs_critic, rewards, dones, _ = env.step(rtn['actions'])

        # Buffer logic: if any env is done, transfer it's buffer to dataset
        finished = dones | (buffer_ptr >= data_length)
        if torch.any(finished):
            num_finished = torch.sum(finished).item()

            if dataset_idx + num_finished > dataset_capacity:
                # Save dataset to disk
                save_dir = os.path.join('/home/harry/projects/parkour_genesis/odometry/dataset')
                os.makedirs(save_dir, exist_ok=True)
                now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                torch.save(dataset, os.path.join(save_dir, f'{now}.pt'))

                # clear the dataset
                for key in dataset:
                    dataset[key].zero_()
                dataset_idx = 0
            else:
                dataset['prop'][dataset_idx: dataset_idx + num_finished] = buffer['prop'][finished].cpu()
                dataset['depth'][dataset_idx: dataset_idx + num_finished] = buffer['depth'][finished].cpu()
                dataset['recon'][dataset_idx: dataset_idx + num_finished] = buffer['recon'][finished].cpu()
                dataset['priv'][dataset_idx: dataset_idx + num_finished] = buffer['priv'][finished].cpu()
                dataset['masks'][dataset_idx: dataset_idx + num_finished] = buffer['masks'][finished].cpu()
                dataset_idx += num_finished

            buffer_ptr[finished] = 0
            buffer['prop'][finished] = 0
            buffer['depth'][finished] = 0
            buffer['recon'][finished] = 0
            buffer['priv'][finished] = 0
            buffer['masks'][finished] = False

            print(f"Collecting data (dataset: {dataset_idx}/{dataset_capacity})")


if __name__ == '__main__':
    with torch.inference_mode():
        play(get_args())
