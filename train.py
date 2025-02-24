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
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    if args.debug:
        mode = "disabled"
        env_cfg.terrain.num_rows = 10
        env_cfg.terrain.num_cols = 2
        env_cfg.env.num_envs = 256
    else:
        mode = "online"

    env, _ = task_registry.make_env(args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env, log_root, args=args, train_cfg=train_cfg)

    # save training parameters
    go1_cfg = class_to_dict(env_cfg)
    go1_cfg.update(class_to_dict(train_cfg))

    wandb.init(project=args.proj_name,
               name=args.exptid,
               group=args.exptid[:3],
               mode=mode,
               dir=log_root,
               config=go1_cfg)

    ppo_runner.learn()


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
