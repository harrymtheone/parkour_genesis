import os

from rsl_rl.runners import runner_list
from .helpers import get_load_path, set_seed


class TaskRegistry:
    def __init__(self, task_list=None):
        self.task_classes = {}
        self.task_cfgs = {}

        if task_list is None:
            from ..envs import task_list

        for t in task_list:
            name, task_class, task_cfg = t
            self.task_classes[name] = task_class
            self.task_cfgs[name] = task_cfg

    def get_cfg(self, name):
        return self.task_cfgs[name]

    def make_alg_runner(self, task_cfg, args, log_root):
        model_dir = os.path.join(log_root, args.proj_name, task_cfg.runner.algorithm_name, args.exptid)

        try:
            os.makedirs(model_dir)
        except FileExistsError:
            pass

        # make runners
        if task_cfg.runner.runner_name in runner_list:
            runner = runner_list[task_cfg.runner.runner_name]
            runner = runner(task_cfg, model_dir, args.exptid, device=args.device)
        else:
            raise ValueError(f'Runner not recognized! With train_cfg.runner_name={task_cfg.runner.runner_name}')

        if args.resumeid:
            if task_cfg.runner.resume_algorithm is None:
                resume_dir = os.path.join(log_root, args.proj_name, task_cfg.runner.algorithm_name, args.resumeid)
            else:
                resume_dir = os.path.join(log_root, args.proj_name, task_cfg.runner.resume_algorithm, args.resumeid)
        elif (args.resumeid is None) and task_cfg.runner.resume:
            resume_dir = model_dir
        else:
            resume_dir = None

        if resume_dir is not None:
            resume_path = get_load_path(resume_dir, checkpoint=args.checkpoint)
            runner.load(resume_path)

        return runner

    def make_env(self, args, task_cfg):
        # check if there is a registered env with that name
        if args.task in self.task_classes:
            task_class = self.task_classes[args.task]
        else:
            raise ValueError(f"Task with name: {args.task} was not registered")

        set_seed(task_cfg.seed)

        # initialize environment
        return task_class(cfg=task_cfg, args=args)
