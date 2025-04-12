import numpy as np
import torch
from rich.table import Table

real_vx_avg = real_vy_avg = real_vz_avg = real_yaw_avg = 0.
zero = 0.


def gen_info_panel(args, env):
    global real_vx_avg, real_vy_avg, real_vz_avg, real_yaw_avg
    cmd_vx, cmd_vy, cmd_yaw, _ = env.commands[env.lookat_id].cpu().numpy()
    real_vx, real_vy, real_vz = env.base_lin_vel[env.lookat_id].cpu().numpy()
    _, _, real_yaw = env.base_ang_vel[env.lookat_id].cpu().numpy()
    real_base_height = env.base_height[env.lookat_id].cpu().numpy()
    feet_height = env.feet_height[env.lookat_id].cpu().numpy()

    real_vx_avg = 0.9 * real_vx_avg + 0.1 * real_vx
    real_vy_avg = 0.9 * real_vy_avg + 0.1 * real_vy
    real_vz_avg = 0.9 * real_vz_avg + 0.1 * real_vz
    real_yaw_avg = 0.9 * real_yaw_avg + 0.1 * real_yaw

    phase = env.phase[env.lookat_id] if hasattr(env, 'phase') else 0.
    phase_increment_ratio = f'{env.phase_increment_ratio[env.lookat_id]: .2f}' if hasattr(env, 'phase_increment_ratio') else 'None'

    friction_ratio = env.sim.friction_coeffs[env.lookat_id].item()
    cmd_vx_correction = env.vel_correction[env.lookat_id, 0].cpu().numpy() if hasattr(env, 'vel_correction') else -100

    if args.headless:
        perc_contact_forces = torch.mean(env.contact_forces_avg, dim=0).cpu().numpy()
        perc_feet_air_time = torch.mean(env.feet_air_time_avg, dim=0).cpu().numpy()
    else:
        perc_contact_forces = env.contact_forces_avg[env.lookat_id].cpu().numpy()
        perc_feet_air_time = env.feet_air_time_avg[env.lookat_id].cpu().numpy()

    perc_feet_air_time = perc_feet_air_time / np.sum(perc_feet_air_time + 1e-5)
    # print(f'env level: {env.env_levels[env.lookat_id].cpu().numpy()}')

    table11 = Table()
    table11.add_column(f'time: {env.episode_length_buf[env.lookat_id].item() * env.dt:.2f}')
    table11.add_column("vx")
    table11.add_column("vy")
    table11.add_column("vz")
    table11.add_column("yaw")
    table11.add_row("cmd", f'{cmd_vx: .2f}', f'{cmd_vy: .2f}', f'{zero: .2f}', f'{cmd_yaw: .2f}')
    table11.add_row("real", f'{real_vx_avg: .2f}', f'{real_vy_avg: .2f}', f'{real_vz_avg: .2f}', f'{real_yaw_avg: .2f}')
    if hasattr(args, 'est'):
        table11.add_row("est_vel", f'{args.est[0]: .2f}', f'{args.est[1]: .2f}', f'{args.est[2]: .2f}', f'{zero: .2f}')

    table12 = Table()
    table12.add_column("target_yaw")
    table12.add_column("base_height")
    table12.add_row(f'{env.target_yaw[env.lookat_id]: .2f}', f'{real_base_height: .2f}')

    table21 = Table()
    table21.add_column("")
    table21.add_column("Left")
    table21.add_column("Right")
    table21.add_row("Feet height", f'{feet_height[0]: .2f}', f'{feet_height[1]: .2f}')
    table21.add_row("Contact forces", f'{perc_contact_forces[0]: .2f}', f'{perc_contact_forces[1]: .2f}')
    table21.add_row("Feet air time", f'{perc_feet_air_time[0]: .2f}', f'{perc_feet_air_time[1]: .2f}')

    table22 = Table()
    table22.add_column(f"phase: {phase: .2f}")
    table22.add_column(f"phase ratio: {phase_increment_ratio}")
    table22.add_row(f"friction: {friction_ratio: .2f}", f"{cmd_vx_correction: .2f}")

    grid = Table.grid()
    grid.add_row(table11, table12)
    grid.add_row(table21, table22)

    return grid
