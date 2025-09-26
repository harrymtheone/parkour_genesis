from collections import deque

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
    phase_bias = env.phase_bias[env.lookat_id].cpu().numpy() if hasattr(env, 'phase_bias') else ['None', 'None']

    friction_ratio = env.sim.friction_coeffs[env.lookat_id].item()

    if args.headless:
        perc_contact_forces = torch.mean(env.contact_forces_avg, dim=0).cpu().numpy()
        perc_feet_air_time = torch.mean(env.feet_air_time_avg, dim=0).cpu().numpy()
    else:
        perc_contact_forces = env.contact_forces_avg[env.lookat_id].cpu().numpy()
        perc_feet_air_time = env.feet_air_time_avg[env.lookat_id].cpu().numpy()

    perc_feet_air_time = perc_feet_air_time / np.sum(perc_feet_air_time + 1e-5)
    # print(f'env level: {env.env_levels[env.lookat_id].cpu().numpy()}')

    goal_timer = env.tracking_goal_timer[env.lookat_id] if hasattr(env, 'tracking_goal_timer') else 0.

    table11 = Table()
    table11.add_column(f'time: {env.episode_length_buf[env.lookat_id].item() * env.dt:.2f}')
    table11.add_column("vx")
    table11.add_column("vy")
    table11.add_column("vz")
    table11.add_column("yaw")
    table11.add_row("cmd", f'{cmd_vx: .2f}', f'{cmd_vy: .2f}', f'{zero: .2f}', f'{cmd_yaw: .2f}')
    table11.add_row("real", f'{real_vx_avg: .2f}', f'{real_vy_avg: .2f}', f'{real_vz_avg: .2f}', f'{real_yaw_avg: .2f}')

    table12 = Table()
    table12.add_column("target_yaw")
    table12.add_column("base_height")
    table12.add_row(f'{env.target_yaw[env.lookat_id]: .2f}', f'{real_base_height: .2f}')
    if hasattr(args, 'est_height'):
        table12.add_row(f'', f'{args.est_height: .2f}')

    table21 = Table()
    table21.add_column("Pushing" if env.pushing_robots else "")
    table21.add_column("Left")
    table21.add_column("Right")
    table21.add_row("Feet height", f'{feet_height[0]: .2f}', f'{feet_height[1]: .2f}')
    table21.add_row("Contact forces", f'{perc_contact_forces[0]: .2f}', f'{perc_contact_forces[1]: .2f}')
    table21.add_row("Feet air time", f'{perc_feet_air_time[0]: .2f}', f'{perc_feet_air_time[1]: .2f}')

    table22 = Table()
    table22.add_column(f"phase: {phase: .2f}")
    table22.add_column(f"phase ratio: {phase_increment_ratio}")
    table22.add_row(f"bias: {phase_bias[0]}", f"{phase_bias[1]}")
    table22.add_row(f"friction: {friction_ratio: .2f}", f"timeout: {goal_timer: .2f}")

    table22.add_row(f"recon_loss: {getattr(args, 'recon_loss', 0.): .2f}", f"")

    grid = Table.grid()
    grid.add_row(table11, table12)
    grid.add_row(table21, table22)

    return grid


import matplotlib

matplotlib.use('TkAgg')  # Use a faster interactive backend than default
import matplotlib.pyplot as plt


class BaseVisualizer:
    figsize: tuple
    subplot_shape: tuple
    subplot_props: dict
    his_length: int

    def __init__(self):
        # History buffers - support multiple lines per subplot
        self.his = {}
        for name, props in self.subplot_props.items():
            if 'lines' in props:
                # Multiple lines per subplot
                for line_name in props['lines']:
                    self.his[line_name] = deque(maxlen=self.his_length)
            else:
                # Single line per subplot (backward compatibility)
                self.his[name] = deque(maxlen=self.his_length)

        # Create figure and axes
        self.fig, axes = plt.subplots(*self.subplot_shape, figsize=self.figsize)
        self.fig.canvas.manager.set_window_title(self.__class__.__name__)

        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Prepare animated lines, static limit lines, and cache backgrounds
        self.lines = {}
        self.backgrounds = {}
        for (name, props), ax in zip(self.subplot_props.items(), axes_flat):
            ax.set_title(name)
            ax.set_xlim(0, self.his_length)
            ax.set_ylim(*props['range'])

            # Draw static horizontal limit lines if provided
            if 'lim_upper' in props:
                ax.axhline(props['lim_upper'], linestyle='--')
            if 'lim_lower' in props:
                ax.axhline(props['lim_lower'], linestyle='--')

            # Create animated lines for data
            if 'lines' in props:
                # Multiple lines per subplot
                colors = props.get('colors', ['blue', 'red', 'green', 'orange', 'purple'])
                for i, line_name in enumerate(props['lines']):
                    color = colors[i % len(colors)]
                    line, = ax.plot([], [], lw=1, animated=True, color=color, label=line_name)
                    self.lines[line_name] = line
                ax.legend()
            else:
                # Single line per subplot (backward compatibility)
                line, = ax.plot([], [], lw=1, animated=True)
                self.lines[name] = line

        # Disable any unused axes
        for ax in axes_flat[len(self.subplot_props):]:
            ax.axis('off')

        # Draw once and cache a clean background for each subplot
        self.fig.canvas.draw()
        for name, line in self.lines.items():
            ax = line.axes
            self.backgrounds[name] = self.fig.canvas.copy_from_bbox(ax.bbox)

        plt.show(block=False)

    def _plot(self, data: dict):
        # Handle both single values and dictionaries
        for name, y_val in data.items():
            if isinstance(y_val, dict):
                # Multiple lines for this subplot
                # First, update all line data
                lines_to_draw = []
                for line_name, values in y_val.items():
                    if line_name not in self.lines:
                        continue

                    buf = self.his[line_name]
                    buf.append(values)
                    y = np.array(buf)
                    x = np.arange(len(y))

                    line = self.lines[line_name]
                    line.set_data(x, y)
                    lines_to_draw.append(line)
                
                # Then restore background once and draw all lines
                if lines_to_draw:
                    ax = lines_to_draw[0].axes
                    # Use the first line's background (they should all be the same)
                    first_line_name = list(y_val.keys())[0]
                    self.fig.canvas.restore_region(self.backgrounds[first_line_name])
                    
                    # Draw all lines
                    for line in lines_to_draw:
                        ax.draw_artist(line)
                    
                    # Blit the updated region to the screen
                    self.fig.canvas.blit(ax.bbox)
            else:
                # Single line per subplot (original behavior)
                if name not in self.lines:
                    continue

                buf = self.his[name]
                buf.append(y_val)
                y = np.array(buf)
                x = np.arange(len(y))

                line = self.lines[name]
                ax = line.axes

                # Restore the clean background (with limit lines baked in)
                self.fig.canvas.restore_region(self.backgrounds[name])

                # Update line data
                line.set_data(x, y)

                # Redraw just this line
                ax.draw_artist(line)

                # Blit the updated region to the screen
                self.fig.canvas.blit(ax.bbox)

        # Flush GUI events
        self.fig.canvas.flush_events()

    def plot(self, env, args):
        raise NotImplementedError


class T1ActionsVisualizer(BaseVisualizer):
    figsize = (12, 12)
    subplot_shape = (6, 2)
    subplot_props = {
        # 'Waist': {'range': (-3, 3)},
        'Left_Hip_Pitch': {'range': (-3, 3)},
        'Right_Hip_Pitch': {'range': (-3, 3)},
        'Left_Hip_Roll': {'range': (-3, 3)},
        'Right_Hip_Roll': {'range': (-3, 3)},
        'Left_Hip_Yaw': {'range': (-3, 3)},
        'Right_Hip_Yaw': {'range': (-3, 3)},
        'Left_Knee_Pitch': {'range': (-3, 3)},
        'Right_Knee_Pitch': {'range': (-3, 3)},
        'Left_Ankle_Pitch': {'range': (-3, 3)},
        'Right_Ankle_Pitch': {'range': (-3, 3)},
        'Left_Ankle_Roll': {'range': (-3, 3)},
        'Right_Ankle_Roll': {'range': (-3, 3)},
    }
    his_length = 50

    def plot(self, env, args):
        actions = env.last_action_output.cpu().numpy()
        self._plot({
            'Waist': actions[env.lookat_id, 0],
            'Left_Hip_Pitch': actions[env.lookat_id, 1],
            'Left_Hip_Roll': actions[env.lookat_id, 2],
            'Left_Hip_Yaw': actions[env.lookat_id, 3],
            'Left_Knee_Pitch': actions[env.lookat_id, 4],
            'Left_Ankle_Pitch': actions[env.lookat_id, 5],
            'Left_Ankle_Roll': actions[env.lookat_id, 6],
            'Right_Hip_Pitch': actions[env.lookat_id, 7],
            'Right_Hip_Roll': actions[env.lookat_id, 8],
            'Right_Hip_Yaw': actions[env.lookat_id, 9],
            'Right_Knee_Pitch': actions[env.lookat_id, 10],
            'Right_Ankle_Pitch': actions[env.lookat_id, 11],
            'Right_Ankle_Roll': actions[env.lookat_id, 12],
        })


class T1DofPosVisualizer(BaseVisualizer):
    figsize = (12, 12)
    subplot_shape = (6, 2)
    subplot_props = {
        'Left_Hip_Pitch': {'range': (-2.00, 1.77), 'lim_lower': -1.80, 'lim_upper': 1.57},
        'Right_Hip_Pitch': {'range': (-2.00, 1.77), 'lim_lower': -1.80, 'lim_upper': 1.57},
        'Left_Hip_Roll': {'range': (-0.40, 1.77), 'lim_lower': -0.20, 'lim_upper': 1.57},
        'Right_Hip_Roll': {'range': (-1.77, 0.40), 'lim_lower': -1.57, 'lim_upper': 0.20},
        'Left_Hip_Yaw': {'range': (-1.20, 1.20), 'lim_lower': -1.00, 'lim_upper': 1.00},
        'Right_Hip_Yaw': {'range': (-1.20, 1.20), 'lim_lower': -1.00, 'lim_upper': 1.00},
        'Left_Knee_Pitch': {'range': (-0.20, 2.54), 'lim_lower': 0.00, 'lim_upper': 2.34},
        'Right_Knee_Pitch': {'range': (-0.20, 2.54), 'lim_lower': 0.00, 'lim_upper': 2.34},
        'Left_Ankle_Pitch': {'range': (-1.07, 0.55), 'lim_lower': -0.87, 'lim_upper': 0.35},
        'Right_Ankle_Pitch': {'range': (-1.07, 0.55), 'lim_lower': -0.87, 'lim_upper': 0.35},
        'Left_Ankle_Roll': {'range': (-0.64, 0.64), 'lim_lower': -0.44, 'lim_upper': 0.44},
        'Right_Ankle_Roll': {'range': (-0.64, 0.64), 'lim_lower': -0.44, 'lim_upper': 0.44},
    }
    his_length = 50

    def plot(self, env, args):
        dof_pos = env.sim.dof_pos.cpu().numpy()
        self._plot({
            'Left_Hip_Pitch': dof_pos[env.lookat_id, 11],
            'Left_Hip_Roll': dof_pos[env.lookat_id, 12],
            'Left_Hip_Yaw': dof_pos[env.lookat_id, 13],
            'Left_Knee_Pitch': dof_pos[env.lookat_id, 14],
            'Left_Ankle_Pitch': dof_pos[env.lookat_id, 15],
            'Left_Ankle_Roll': dof_pos[env.lookat_id, 16],
            'Right_Hip_Pitch': dof_pos[env.lookat_id, 17],
            'Right_Hip_Roll': dof_pos[env.lookat_id, 18],
            'Right_Hip_Yaw': dof_pos[env.lookat_id, 19],
            'Right_Knee_Pitch': dof_pos[env.lookat_id, 20],
            'Right_Ankle_Pitch': dof_pos[env.lookat_id, 21],
            'Right_Ankle_Roll': dof_pos[env.lookat_id, 22],
        })


class T1DofVelVisualizer(BaseVisualizer):
    figsize = (12, 12)
    subplot_shape = (6, 2)
    subplot_props = {
        # 'Waist': {'range': (-3, 3)},
        'Left_Hip_Pitch': {'range': (-3, 3)},
        'Right_Hip_Pitch': {'range': (-3, 3)},
        'Left_Hip_Roll': {'range': (-3, 3)},
        'Right_Hip_Roll': {'range': (-3, 3)},
        'Left_Hip_Yaw': {'range': (-3, 3)},
        'Right_Hip_Yaw': {'range': (-3, 3)},
        'Left_Knee_Pitch': {'range': (-3, 3)},
        'Right_Knee_Pitch': {'range': (-3, 3)},
        'Left_Ankle_Pitch': {'range': (-3, 3)},
        'Right_Ankle_Pitch': {'range': (-3, 3)},
        'Left_Ankle_Roll': {'range': (-3, 3)},
        'Right_Ankle_Roll': {'range': (-3, 3)},
    }
    his_length = 50

    def plot(self, env, args):
        dof_vel = env.sim.dof_vel.cpu().numpy()
        offset = 0
        self._plot({
            'Left_Hip_Pitch': dof_vel[env.lookat_id, offset + 0],
            'Left_Hip_Roll': dof_vel[env.lookat_id, offset + 1],
            'Left_Hip_Yaw': dof_vel[env.lookat_id, offset + 2],
            'Left_Knee_Pitch': dof_vel[env.lookat_id, offset + 3],
            'Left_Ankle_Pitch': dof_vel[env.lookat_id, offset + 4],
            'Left_Ankle_Roll': dof_vel[env.lookat_id, offset + 5],
            'Right_Hip_Pitch': dof_vel[env.lookat_id, offset + 6],
            'Right_Hip_Roll': dof_vel[env.lookat_id, offset + 7],
            'Right_Hip_Yaw': dof_vel[env.lookat_id, offset + 8],
            'Right_Knee_Pitch': dof_vel[env.lookat_id, offset + 9],
            'Right_Ankle_Pitch': dof_vel[env.lookat_id, offset + 10],
            'Right_Ankle_Roll': dof_vel[env.lookat_id, offset + 11],
        })


class T1TorqueVisualizer(BaseVisualizer):
    figsize = (12, 12)
    subplot_shape = (7, 2)
    subplot_props = {
        # 'Waist': {'range': (-3, 3)},
        'Left_Hip_Pitch': {'range': (-55, 55), 'lim_lower': -45, 'lim_upper': 45},
        'Right_Hip_Pitch': {'range': (-55, 55), 'lim_lower': -45, 'lim_upper': 45},
        'Left_Hip_Roll': {'range': (-40, 40), 'lim_lower': -30, 'lim_upper': 30},
        'Right_Hip_Roll': {'range': (-40, 40), 'lim_lower': -30, 'lim_upper': 30},
        'Left_Hip_Yaw': {'range': (-40, 40), 'lim_lower': -30, 'lim_upper': 30},
        'Right_Hip_Yaw': {'range': (-40, 40), 'lim_lower': -30, 'lim_upper': 30},
        'Left_Knee_Pitch': {'range': (-70, 70), 'lim_lower': -60, 'lim_upper': 60},
        'Right_Knee_Pitch': {'range': (-70, 70), 'lim_lower': -60, 'lim_upper': 60},
        'Left_Ankle_Pitch': {'range': (-30, 30), 'lim_lower': -20, 'lim_upper': 20},
        'Right_Ankle_Pitch': {'range': (-30, 30), 'lim_lower': -20, 'lim_upper': 20},
        'Left_Ankle_Roll': {'range': (-25, 25), 'lim_lower': -15, 'lim_upper': 15},
        'Right_Ankle_Roll': {'range': (-25, 25), 'lim_lower': -15, 'lim_upper': 15},
        'Left_Contact_Forces': {'range': (0, 700), 'lim_lower': 0, 'lim_upper': 300},
        'Right_Contact_Forces': {'range': (0, 700), 'lim_lower': 0, 'lim_upper': 300},
    }
    his_length = 50

    def plot(self, env, args):
        torques = env.torques.cpu().numpy()
        feet_contact_forces = torch.norm(env.sim.contact_forces[:, env.feet_indices], dim=-1).cpu().numpy()
        offset = 0
        self._plot({
            'Waist': torques[env.lookat_id, offset + 0],
            'Left_Hip_Pitch': torques[env.lookat_id, offset + 1],
            'Left_Hip_Roll': torques[env.lookat_id, offset + 2],
            'Left_Hip_Yaw': torques[env.lookat_id, offset + 3],
            'Left_Knee_Pitch': torques[env.lookat_id, offset + 4],
            'Left_Ankle_Pitch': torques[env.lookat_id, offset + 5],
            'Left_Ankle_Roll': torques[env.lookat_id, offset + 6],
            'Right_Hip_Pitch': torques[env.lookat_id, offset + 7],
            'Right_Hip_Roll': torques[env.lookat_id, offset + 8],
            'Right_Hip_Yaw': torques[env.lookat_id, offset + 9],
            'Right_Knee_Pitch': torques[env.lookat_id, offset + 10],
            'Right_Ankle_Pitch': torques[env.lookat_id, offset + 11],
            'Right_Ankle_Roll': torques[env.lookat_id, offset + 12],
            'Left_Contact_Forces': feet_contact_forces[env.lookat_id, 0],
            'Right_Contact_Forces': feet_contact_forces[env.lookat_id, 1],
        })


class RewVisualizer(BaseVisualizer):
    figsize = (12, 6)
    subplot_shape = (1, 1)
    subplot_props = {
        'Rew': {'range': (-0.2, 0.2), 'lim_lower': 0.},
    }
    his_length = 50

    def plot(self, env, *args):
        # rew = env.extras['step_rew']
        rew = env.rew_buf[env.lookat_id].item()
        self._plot({'Rew': rew})


class VelEstVisualizer(BaseVisualizer):
    figsize = (12, 6)
    subplot_shape = (3, 1)
    subplot_props = {
        'X': {'range': (-0.5, 0.5), 'lines': ['real_vx', 'est_x'], 'colors': ['blue', 'red']},
        'Y': {'range': (-0.5, 0.5), 'lines': ['real_vy', 'est_y'], 'colors': ['blue', 'red']},
        'Z': {'range': (-0.5, 0.5), 'lines': ['real_vz', 'est_z'], 'colors': ['blue', 'red']},
    }
    his_length = 50

    def plot(self, env, args):
        real_vx, real_vy, real_vz = env.base_lin_vel[env.lookat_id].cpu().numpy()
        est_x, est_y, est_z = args.vel_est[env.lookat_id].cpu().numpy()
        self._plot({
            'X': {'real_vx': real_vx, 'est_x': est_x},
            'Y': {'real_vy': real_vy, 'est_y': est_y},
            'Z': {'real_vz': real_vz, 'est_z': est_z},
        })
