try:
    import isaacgym, torch  # NOQA
except ImportError:
    import torch

import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from legged_gym.simulator import SimulatorType
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.modules.model_odom import OdomTransformer


def play(args):
    log_root = 'logs'
    args.simulator = SimulatorType.IsaacGym
    args.headless = True
    args.resume = True

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=args.task)

    # override some parameters for testing
    task_cfg.play.control = False
    task_cfg.env.num_envs = 256
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

    use_amp = False
    transformer = OdomTransformer(50, 64, 128, 3).to(args.device)
    optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    l1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    if not args.debug:
        log_dir = os.path.join(log_root, 'odom_online', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        writer = SummaryWriter(log_dir)
        print(f"Logging to {log_dir}")
    global_step = 0

    while True:
        with torch.amp.autocast(enabled=use_amp, device_type=args.device):
            optim.zero_grad()

            recon_rough, recon_refine, priv = transformer.inference_forward(obs.proprio, obs.depth)

            loss_recon_rough = mse(recon_rough, obs.scan)
            loss_recon_refine = l1(recon_refine, obs.scan)
            loss_priv = mse(priv, obs.priv_actor)
            scaler.scale(loss_recon_rough + loss_recon_refine + loss_priv).backward()
            scaler.step(optim)
            scaler.update()

        if not args.headless:
            env.draw_recon(recon_refine.detach())

        if not args.debug:
            writer.add_scalar('Loss/recon_rough', loss_recon_rough.item(), global_step)
            writer.add_scalar('Loss/recon_refine', loss_recon_refine.item(), global_step)
            writer.add_scalar('Loss/priv', loss_priv.item(), global_step)
        global_step += 1

        with torch.inference_mode():
            rtn = runner.play_act(obs, use_estimated_values=False, eval_=True, dones=dones)
        obs, obs_critic, rewards, dones, _ = env.step(rtn['actions'])

        transformer.detach_hidden_states()
        if torch.any(dones):
            transformer.reset(dones)

        if not args.headless:
            env.refresh_graphics(clear_lines=True)

        if not args.debug and global_step % 1000 == 0:
            torch.save(transformer.state_dict(), os.path.join(log_dir, 'latest.pth'))


if __name__ == '__main__':
    play(get_args())
