import tqdm

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
from rsl_rl.storage.odometer_storage import OdometerStorage


class Transition:
    def __init__(self):
        self.observations = None
        self.hidden_states = None
        self.dones = None

    def get_items(self):
        return self.__dict__.items()

    def clear(self):
        for key in self.__dict__:
            setattr(self, key, None)


def play(args):
    log_root = 'logs'
    args.simulator = SimulatorType.IsaacGym
    args.headless = True
    args.resume = True

    task_registry = TaskRegistry()
    task_cfg = task_registry.get_cfg(name=args.task)

    # override some parameters for testing
    task_cfg.play.control = False
    task_cfg.env.num_envs = 512
    task_cfg.terrain.num_rows = 10
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
        'smooth_slope': 0,
        'rough_slope': 0,
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
    task_cfg.terrain.num_cols = sum(task_cfg.terrain.terrain_dict.values())
    task_cfg.terrain.num_cols *= 1 if args.debug else 5

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

    use_amp = True
    odom = OdomTransformer(50, 64, 128, 3).to(args.device)
    optim = torch.optim.Adam(odom.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()

    transition = Transition()
    storage = OdometerStorage(task_cfg.env.num_envs, 10, args.device)

    if not args.debug:
        log_dir = os.path.join(log_root, 'odom_online', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        writer = SummaryWriter(log_dir)
        print(f"Logging to {log_dir}")

    num_epoch = 0

    for step_i in tqdm.trange(int(1e10)):
        with torch.amp.autocast(enabled=use_amp, device_type=args.device):

            # ################ data collection ################
            transition.observations = obs
            if odom.hidden_states is None:
                transition.hidden_states = torch.zeros(2, task_cfg.env.num_envs, 128)
            else:
                transition.hidden_states = odom.hidden_states

            with torch.no_grad():
                odom.inference_forward(obs.proprio, obs.depth)
                rtn = runner.play_act(obs, use_estimated_values=False, eval_=True)
            obs, obs_critic, rewards, dones, _ = env.step(rtn['actions'])

            transition.dones = dones

            storage.add_transitions(transition)

            if torch.any(dones):
                # Reset both hidden states when an episode ends
                odom.reset(dones, eval_=True)

            # ################ odometer update ################

            if storage.is_full():
                # Accumulate losses
                for batch in storage.recurrent_mini_batch_generator(num_epochs=1):
                    observations = batch['observations']
                    hidden_states = batch['hidden_states']

                    proprio = observations['proprio']
                    depth = observations['depth']
                    rough_scan = observations['rough_scan']
                    scan = observations['scan']
                    priv_actor = observations['priv_actor']

                    recon_rough, recon_refine, priv_est = odom(proprio, depth, hidden_states)

                    loss_recon_rough = mse(recon_rough.squeeze(2), rough_scan)  # TODO: maybe here we need to use masks
                    loss_recon_refine = mse(recon_refine[:, :, 0], scan[:, :, 0])
                    loss_edge = bce(recon_refine[:, :, 1], scan[:, :, 1])
                    loss_priv = mse(priv_est, priv_actor)

                    total_loss = loss_recon_rough + loss_recon_refine + loss_edge + loss_priv

                    optim.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.step(optim)
                    scaler.update()
                    num_epoch += 1

                    if not args.debug:
                        # Log the averaged losses
                        writer.add_scalar('Loss/recon_rough', loss_recon_rough.item(), num_epoch)
                        writer.add_scalar('Loss/recon_refine', loss_recon_refine.item(), num_epoch)
                        writer.add_scalar('Loss/edge', loss_edge.item(), num_epoch)
                        writer.add_scalar('Loss/priv', loss_priv.item(), num_epoch)
                        writer.add_scalar('Loss/total', total_loss.item(), num_epoch)

        if not args.headless:
            env.draw_recon(recon_refine[env.lookat_id].detach())
            env.refresh_graphics(clear_lines=True)

        if not args.debug and num_epoch % 100 == 0:
            torch.save(odom.state_dict(), os.path.join(log_dir, 'latest.pth'))


if __name__ == '__main__':
    play(get_args())
