try:
    import isaacgym, torch
except ImportError:
    import torch

import os

import wandb

from legged_gym.simulator import SimulatorType
from legged_gym.utils.helpers import get_args, class_to_dict
from legged_gym.utils.task_registry import TaskRegistry


def train(args):
    args.headless = True
    args.simulator = SimulatorType.IsaacGym
    # args.simulator = SimulatorType.Genesis
    # args.simulator = SimulatorType.IsaacSim

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
        # args.headless = False
        task_cfg.terrain.num_rows = 10
        task_cfg.terrain.num_cols = 2
        task_cfg.env.num_envs = 128
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
    env = task_registry.make_env(args=args, task_cfg=task_cfg)
    ppo_runner.learn(env)


if __name__ == '__main__':
    train(get_args())

    # from line_profiler import LineProfiler
    # from rsl_rl.runners.rl_dream_runner import RLDreamRunner
    # from legged_gym.envs.base.legged_robot import LeggedRobot
    #
    # lp = LineProfiler()
    # lp.add_function(LeggedRobot.step)
    # wrapper = lp(train)
    # wrapper(get_args())
    # lp.print_stats()
