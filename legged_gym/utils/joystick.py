import sys

import pygame


class BaseHandler:
    def __init__(self, legged_robot):
        self.legged_robot = legged_robot

        self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd = 0., 0., 0.
        self.x_vel_cmd_scale, self.y_vel_cmd_scale, self.yaw_vel_cmd_scale = 0.5, 0.4, 1.0

        pygame.init()
        # pygame.display.set_mode((500, 500))

    def get_control_input(self):
        return [self.x_vel_cmd * self.x_vel_cmd_scale,
                self.y_vel_cmd * self.y_vel_cmd_scale * 0,
                self.yaw_vel_cmd * self.yaw_vel_cmd_scale]

    def handle_device_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)

            elif event.type == pygame.KEYUP:
                self.handle_keyup(event.key)

    def handle_keydown(self, key):
        if key == pygame.K_ESCAPE:
            sys.exit()

        # if key == pygame.K_v:
        #     self.env.enable_viewer_sync = not self.env.enable_viewer_sync
        #     if self.env.free_cam:
        #         self.env.set_camera(self.env.cfg.viewer.pos, self.env.cfg.viewer.lookat)
        #     return

        # if key == pygame.K_SPACE:
        #     pause = True
        #     while pause:
        #         time.sleep(0.1)
        #         self.env.gym.draw_viewer(self.env.viewer, self.env.sim, True)
        #
        #         for evt in pygame.event.get():
        #             if evt.type == pygame.KEYDOWN and evt.key == pygame.K_SPACE:
        #                 pause = False
        #
        #         if self.env.gym.query_viewer_has_closed(self.env.viewer):
        #             sys.exit()
        #     return

        if not self.env.free_cam:
            if key == pygame.K_LEFTBRACKET:
                self.env.lookat_id = (self.env.lookat_id - 1) % self.env.num_envs
                self.env.lookat(self.env.lookat_id)
                return

            if key == pygame.K_RIGHTBRACKET:
                self.env.lookat_id = (self.env.lookat_id + 1) % self.env.num_envs
                self.env.lookat(self.env.lookat_id)
                return

    def handle_keyup(self, key):
        pass


class KeyboardHandler(BaseHandler):
    def handle_device_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)

            elif event.type == pygame.KEYUP:
                self.handle_keyup(event.key)

    def handle_keydown(self, key):
        super().handle_keydown(key)

        if self.legged_robot.cfg.play.control:
            if key == pygame.K_w:
                self.x_vel_cmd += 1
            elif key == pygame.K_s:
                self.x_vel_cmd -= 1
            elif key == pygame.K_a:
                self.y_vel_cmd += 1
            elif key == pygame.K_d:
                self.y_vel_cmd -= 1
            elif key == pygame.K_q:
                self.yaw_vel_cmd += 1
            elif key == pygame.K_e:
                self.yaw_vel_cmd -= 1

    def handle_keyup(self, key):
        if self.legged_robot.cfg.play.control:
            if key == pygame.K_w:
                self.x_vel_cmd -= 1
            if key == pygame.K_s:
                self.x_vel_cmd += 1
            if key == pygame.K_a:
                self.y_vel_cmd -= 1
            if key == pygame.K_d:
                self.y_vel_cmd += 1
            if key == pygame.K_q:
                self.yaw_vel_cmd -= 1
            if key == pygame.K_e:
                self.yaw_vel_cmd += 1


CAMERA_MODE = "free"  # 自由模式/跟随模式状态码
reset_pos = False
A_button_was_pressed = False
X_button_was_pressed = False


class JoystickHandler(BaseHandler):
    def __init__(self, legged_robot):
        super().__init__(legged_robot)
        self.x_vel_cmd_scale, self.y_vel_cmd_scale, self.yaw_vel_cmd_scale = 0.99, 0.5, 1.0
        pygame.joystick.init()

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        else:
            print('JoyStick: failed to connected to a joystick!!!!')
            sys.exit()

    def handle_device_input(self):
        global CAMERA_MODE, reset_pos, A_button_was_pressed, X_button_was_pressed

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)

        # 获取手柄按钮输入
        pygame.event.get()
        X_button_pressed = self.joystick.get_button(2)  # 老版遥控器是 B ，跟随视角
        A_button_pressed = self.joystick.get_button(0)  # 老版遥控器是 x ,重生
        # 按钮操作触发
        if X_button_pressed and not X_button_was_pressed:
            CAMERA_MODE = "follow" if CAMERA_MODE == "free" else "free"
            print(f"Camera mode switched to:{CAMERA_MODE}")
        X_button_was_pressed = X_button_pressed

        if A_button_pressed and not A_button_was_pressed:
            reset_pos = True
            print("press A")
        else:
            reset_pos = False
            A_button_was_pressed = A_button_pressed

        # 获取手柄命令输入
        self.x_vel_cmd = -self.joystick.get_axis(1)
        self.y_vel_cmd = -self.joystick.get_axis(0)
        self.yaw_vel_cmd = -self.joystick.get_axis(2)  # old joystick 2, new 3
