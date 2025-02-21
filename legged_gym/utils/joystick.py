import sys

import pygame


class JoystickHandler:
    def __init__(self, sim):
        self.sim = sim

        self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd = 0., 0., 0.
        self.x_vel_cmd_scale, self.y_vel_cmd_scale, self.yaw_vel_cmd_scale = 0.99, 0.5, 1.0

        self.btn_listened = [6, 7, 10, 14]
        self.btn_prev_state = [0] * len(self.btn_listened)

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        else:
            print('JoyStick: failed to connected to a joystick!!!!')
            sys.exit()

    def on_press(self, btn_idx):
        if btn_idx == 6:
            self.sim.lookat(self.sim.lookat_id - 1)  # look at previous env
        elif btn_idx == 7:
            self.sim.lookat(self.sim.lookat_id + 1)  # look at next env
        elif btn_idx == 10:  # select
            sys.exit()
        elif btn_idx == 14:
            self.sim.free_cam = not self.sim.free_cam
        elif btn_idx == 4:
            self.sim.enable_viewer_sync = not self.sim.enable_viewer_sync

    def on_release(self, key):
        pass

    def get_control_input(self):
        return [self.x_vel_cmd * self.x_vel_cmd_scale,
                self.y_vel_cmd * self.y_vel_cmd_scale * 0,
                self.yaw_vel_cmd * self.yaw_vel_cmd_scale]

    def handle_device_input(self):
        pygame.event.get()
        self.x_vel_cmd = -self.joystick.get_axis(1)
        self.y_vel_cmd = -self.joystick.get_axis(0)
        self.yaw_vel_cmd = -self.joystick.get_axis(2)  # old joystick 2, new 3

        for i, btn_idx in enumerate(self.btn_listened):
            btn_pressed = self.joystick.get_button(btn_idx)

            if btn_pressed - self.btn_prev_state[i] > 0:
                self.on_press(btn_idx)
            elif btn_pressed - self.btn_prev_state[i] < 0:
                self.on_release(btn_idx)

            self.btn_prev_state[i] = btn_pressed


if __name__ == '__main__':
    import time

    js = JoystickHandler(None)

    while True:
        js.handle_device_input()
        time.sleep(0.1)
