try:
    import isaacgym, torch  # NOQA
except ImportError:
    import torch
import os
from datetime import datetime

import tqdm
from torch.utils.tensorboard import SummaryWriter

from legged_gym.simulator import SimulatorType
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import TaskRegistry


def play(args):
    # check if it is on AutoDL
    autodl_log_root = os.path.join(os.path.expanduser("~"), 'autodl-tmp')
    if os.path.isdir(autodl_log_root):
        log_root = os.path.join(autodl_log_root, 'logs')
    else:
        log_root = 'logs'

    print('-' * 10, 'log_root: ', log_root, '-' * 10)

    args.simulator = SimulatorType.IsaacGym
    args.headless = True
    args.resume = False

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=args.task)

    # override some parameters for testing
    task_cfg.play.control = False
    task_cfg.env.num_envs = 512
    task_cfg.terrain.num_rows = 10
    task_cfg.terrain.max_init_terrain_level = task_cfg.terrain.num_rows - 1
    task_cfg.terrain.curriculum = False
    # task_cfg.terrain.max_difficulty = True

    task_cfg.commands.resampling_time = 1

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
        'smooth_slope': 0,
        'rough_slope': 1,
        'stairs_up': 1,
        'stairs_down': 1,
        'huge_stair': 0,
        'discrete': 0,
        'stepping_stone': 0,
        'gap': 0,
        'pit': 0,
        'parkour_flat': 0,
        'parkour': 0,
        'parkour_gap': 0,
        'parkour_box': 0,
        'parkour_step': 0,
        'parkour_stair': 1,
        'parkour_stair_down': 1,
        'parkour_mini_stair': 1,
        'parkour_mini_stair_down': 1,
        'parkour_go_back_stair': 0,
    }
    task_cfg.terrain.num_cols = sum(task_cfg.terrain.terrain_dict.values())
    task_cfg.terrain.num_cols *= 1 if args.debug else 5

    # prepare environment
    args.n_rendered_envs = task_cfg.env.num_envs
    task_cfg = task_registry.get_cfg(name=args.task)
    env = task_registry.make_env(args=args, task_cfg=task_cfg)
    obs, obs_critic = env.get_observations(), env.get_critic_observations()

    # load policy
    env.sim.clear_lines = True
    task_cfg.runner.resume = True
    task_cfg.runner.logger_backend = None
    runner = task_registry.make_alg_runner(task_cfg, args, log_root)

    use_amp = True
    reconstructor = runner.odom.odom
    optim = torch.optim.Adam(reconstructor.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()
    bce = torch.nn.BCEWithLogitsLoss()

    # reconstructor.load_state_dict(torch.load('/home/harry/projects/parkour_genesis/logs/odom_online/best2/latest.pth', weights_only=True))

    if not args.debug:
        log_dir = os.path.join(log_root, 'odom_online', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        writer = SummaryWriter(log_dir)
        print(f"Logging to {log_dir}")

    loss_recon_rough = 0.
    loss_recon_refine = 0.
    loss_edge = 0.
    loss_priv = 0.
    accumulation_steps = 4
    num_epoch = 0

    use_estimated_values = torch.linspace(0, 1, task_cfg.env.num_envs, device=args.device) > 0.9

    for step_i in tqdm.trange(int(1e10)):

        with torch.amp.autocast(enabled=use_amp, device_type=args.device):
            # rollout - use the training hidden state
            recon_rough, recon_refine, priv_est = reconstructor.inference_forward(obs.proprio, obs.depth, eval_=False)

            # Accumulate losses
            loss_recon_rough += mse(recon_rough, obs.rough_scan.unsqueeze(1))
            loss_recon_refine += l1(recon_refine[:, 0], obs_critic.scan)
            loss_edge += bce(recon_refine[:, 1], obs_critic.edge_mask)
            loss_priv += mse(priv_est, obs_critic.priv_actor)

            # Perform an update every `accumulation_steps`
            if (step_i + 1) % accumulation_steps == 0:
                num_epoch += 1
                loss_recon_rough /= accumulation_steps
                loss_recon_refine /= accumulation_steps
                loss_edge /= accumulation_steps
                loss_priv /= accumulation_steps

                total_loss = loss_recon_rough + loss_recon_refine + loss_edge + loss_priv

                optim.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optim)
                scaler.update()

                if not args.debug:
                    # Log the averaged losses
                    writer.add_scalar('Loss/recon_rough', loss_recon_rough.item(), num_epoch)
                    writer.add_scalar('Loss/recon_refine', loss_recon_refine.item(), num_epoch)
                    writer.add_scalar('Loss/edge', loss_edge.item(), num_epoch)
                    writer.add_scalar('Loss/priv', loss_priv.item(), num_epoch)
                    writer.add_scalar('Loss/total', total_loss.item(), num_epoch)

                # Detach the training hidden state to start a new BPTT trajectory
                reconstructor.detach_hidden_states()
                loss_recon_rough = 0.
                loss_recon_refine = 0.
                loss_edge = 0.
                loss_priv = 0.

        if not args.headless:
            env.draw_recon(recon_refine[env.lookat_id].detach())

        with torch.inference_mode():
            rtn = runner.play_act(
                obs,
                use_estimated_values=use_estimated_values.unsqueeze(1),
                eval_=False,
                recon=recon_refine.detach(),
                est=priv_est.detach()
            )
        obs, obs_critic, rewards, dones, _ = env.step(rtn['actions'])

        if torch.any(dones):
            # Reset both hidden states when an episode ends
            reconstructor.reset(dones, eval_=False)

        if not args.headless:
            env.refresh_graphics(clear_lines=True)

        if not args.debug and num_epoch % 1000 == 0:
            torch.save(reconstructor.state_dict(), os.path.join(log_dir, 'latest.pth'))


if __name__ == '__main__':
    play(get_args())
