import os

from rsl_rl.runners import runner_list
from .helpers import get_args, get_load_path, set_seed


class TaskRegistry:
    def __init__(self, task_list=None):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

        if task_list is None:
            from ..envs import task_list

        for t in task_list:
            name, task_class, env_cfg, train_cfg = t
            self.task_classes[name] = task_class
            self.env_cfgs[name] = env_cfg
            self.train_cfgs[name] = train_cfg

    def get_cfgs(self, name):
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, args=None, env_cfg=None):
        # if no args passed get command line arguments
        if args is None:
            args = get_args()

        # check if there is a registered env with that name
        if args.task in self.task_classes:
            task_class = self.task_classes[args.task]
        else:
            raise ValueError(f"Task with name: {args.task} was not registered")

        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(args.task)

        set_seed(env_cfg.seed)

        # initialize environment
        env = task_class(cfg=env_cfg, args=args)
        return env, env_cfg

    def make_alg_runner(self, env, log_root, args=None, train_cfg=None):
        # if no args passed get command line arguments
        if args is None:
            args = get_args()

        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if args.task is None:
                raise ValueError("Either 'name' or 'train_cfg' must not be None")

            # load config files
            _, train_cfg = self.get_cfgs(args.task)
        elif args.task is not None:
            print(f"'train_cfg' provided -> Ignoring 'name={args.task}'")

        model_dir = os.path.join(log_root, args.proj_name, args.exptid)

        try:
            os.makedirs(model_dir)
        except FileExistsError:
            pass

        # make runners
        if train_cfg.runner_name in runner_list:
            runner = runner_list[train_cfg.runner_name]
            runner = runner(env, train_cfg, log_dir=model_dir, device=args.device)
        else:
            raise ValueError(f'Runner not recognized! With train_cfg.runner_name={train_cfg.runner_name}')

        if args.resumeid:
            resume_dir = os.path.join(log_root, args.proj_name, args.resumeid)
            resume_path = get_load_path(resume_dir, checkpoint=args.checkpoint)
            runner.load(resume_path)

        return runner, train_cfg
