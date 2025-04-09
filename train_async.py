try:
    import isaacgym, torch
except ImportError:
    import torch

import os

import wandb

from legged_gym.simulator import SimulatorType
from legged_gym.utils.helpers import get_args, class_to_dict
from legged_gym.utils.task_registry import TaskRegistry


def worker_rollout(args, queue_dict):
    args.headless = True
    # args.simulator = SimulatorType.Genesis
    args.simulator = SimulatorType.IsaacGym

    # check if it is on AutoDL
    autodl_log_root = os.path.join(os.path.expanduser("~"), 'autodl-tmp')
    if os.path.isdir(autodl_log_root):
        log_root = os.path.join(autodl_log_root, 'logs')
    else:
        log_root = 'logs'

    print('-' * 10, 'log_root: ', log_root, '-' * 10)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=args.task)

    if args.debug:
        # args.headless = False
        task_cfg.terrain.num_rows = 2
        task_cfg.terrain.num_cols = 2
        task_cfg.env.num_envs = 2048

    ppo_runner = task_registry.make_alg_runner(task_cfg=task_cfg, args=args, log_root=log_root)
    env = task_registry.make_env(args=args, task_cfg=task_cfg)
    ppo_runner.rollout(env, queue_dict)


def worker_update(args, queue_dict):
    args.headless = True
    # args.simulator = SimulatorType.Genesis
    args.simulator = SimulatorType.IsaacGym

    # check if it is on AutoDL
    autodl_log_root = os.path.join(os.path.expanduser("~"), 'autodl-tmp')
    if os.path.isdir(autodl_log_root):
        log_root = os.path.join(autodl_log_root, 'logs')
    else:
        log_root = 'logs'

    print('-' * 10, 'log_root: ', log_root, '-' * 10)

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=args.task)

    if args.debug:
        mode = "disabled"
        task_cfg.env.num_envs = 2048
    else:
        mode = "online"

    # save training parameters
    wandb.init(project=args.proj_name,
               name=args.exptid,
               group=task_cfg.runner.algorithm_name,
               mode=mode,
               dir=log_root,
               config=class_to_dict(task_cfg))

    ppo_runner = task_registry.make_alg_runner(task_cfg=task_cfg, args=args, log_root=log_root)
    ppo_runner.update(queue_dict)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    args = get_args()

    queue_dict = {
        'queue_state_dict': torch.multiprocessing.Queue(maxsize=2),
        'queue_rollout_data': torch.multiprocessing.Queue(maxsize=2),
        'queue_info': torch.multiprocessing.Queue(),
    }

    update_process = torch.multiprocessing.Process(target=worker_update, args=(args, queue_dict))
    update_process.start()

    worker_rollout(args, queue_dict)

    update_process.join()
