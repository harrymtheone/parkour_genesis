from enum import Enum
from typing import List

import torch
import warp as wp

from legged_gym.utils.math import torch_rand_float


class BaseWrapper:
    def __init__(self, cfg, args):
        self.cfg = cfg

        self.device = torch.device(args.device)
        self.headless = args.headless
        self.num_envs = self.cfg.env.num_envs

        self._body_names: List[str]
        self._dof_names: List[str]
        self.num_bodies: int
        self.num_dof: int

        self.dof_pos_limits: torch.Tensor
        self.torque_limits: torch.Tensor

        self.height_samples: torch.Tensor
        self.height_guidance: torch.Tensor
        self.edge_mask: torch.Tensor

        self.init_done = False

    # ------------------------------------------------- Simulator Interfaces -------------------------------------------------

    def get_trimesh(self):
        raise NotImplementedError

    def refresh_variable(self):
        raise NotImplementedError

    def set_root_state(self, env_ids, pos, quat, lin_vel, ang_vel):
        raise NotImplementedError

    def set_dof_state(self, env_ids, dof_pos, dof_vel):
        raise NotImplementedError

    def set_dof_kp(self, kp, env_ids=None):
        # Only used when you use pos_target drive mode
        raise NotImplementedError

    def set_dof_kv(self, kp, env_ids=None):
        # Only used when you use pos_target drive mode
        raise NotImplementedError

    def set_dof_stiffness(self, stiffness, env_ids=None):
        raise NotImplementedError

    def set_dof_damping_coef(self, damping_coef, env_ids=None):
        raise NotImplementedError

    def set_dof_friction(self, friction, env_ids=None):
        raise NotImplementedError

    def set_dof_armature(self, armature, env_ids=None):
        raise NotImplementedError

    def apply_perturbation(self, force, torque):
        raise NotImplementedError

    def step_environment(self):
        raise NotImplementedError

    def control_dof_torque(self, torques):
        raise NotImplementedError

    def get_full_names(self, names, is_link) -> list:
        full_names = []
        if type(names) is str:
            names = [names]

        for n in names:
            if is_link:
                full_names.extend([s for s in self._body_names if n in s])
            else:
                full_names.extend([s for s in self._dof_names if n in s])

        assert len(full_names) > 0, f"No names found! {names}, {is_link}"
        return full_names

    def create_indices(self, names, is_link):
        raise NotImplementedError

    @property
    def dof_names(self):
        return self._dof_names

    @property
    def root_pos(self):
        raise NotImplementedError

    @property
    def root_quat(self):
        # Quaternion: (x, y, z, w)
        raise NotImplementedError

    @property
    def root_lin_vel(self):
        raise NotImplementedError

    @property
    def root_ang_vel(self):
        raise NotImplementedError

    @property
    def dof_pos(self):
        raise NotImplementedError

    @property
    def dof_vel(self):
        raise NotImplementedError

    @property
    def link_pos(self):
        raise NotImplementedError

    @property
    def link_quat(self):  # (x, y, z, w)
        raise NotImplementedError

    @property
    def link_vel(self):
        raise NotImplementedError

    @property
    def link_COM(self):
        raise NotImplementedError

    @property
    def link_mass(self):
        raise NotImplementedError

    @property
    def contact_forces(self):
        raise NotImplementedError

    # ------------------------------------------------- Graphics Interface -------------------------------------------------
    def render(self):
        raise NotImplementedError

    def clear_debug_lines(self):
        raise NotImplementedError

    def lookat(self, i):
        raise NotImplementedError

    def draw_points(self, points, radius=0.02, color=(0, 1, 0), sphere_lines=4, z_shift=0.02):
        raise NotImplementedError

    # ------------------------------------------------- Utils -------------------------------------------------

    def _zero_tensor(self, *shape, dtype=torch.float, requires_grad=False):
        return torch.zeros(*shape, dtype=dtype, device=self.device, requires_grad=requires_grad)

    def _zero_wp_array(self, *shape, dtype=None, device=None):
        if dtype is None:
            raise ValueError('dtype cannot be None!')

        if device is None:
            device = wp.device_from_torch(self.device)

        return wp.zeros(shape, dtype=dtype, device=device)

    # ------------------------------------------------- Domain Randomization -------------------------------------------------

    def _randomize_rigid_body_props(self):
        """ Randomise some of the rigid body properties of the actor in the given environments, i.e.
            sample the mass, centre of mass position, friction and restitution."""

        self._link_mass = self._zero_tensor(self.num_envs, self.num_bodies)
        self._link_com_shift = self._zero_tensor(self.num_envs, self.num_bodies, 3)

        if not self.init_done:
            self.friction_coeffs = 1 + self._zero_tensor(self.num_envs, 1)
            self.payload_masses = self._zero_tensor(self.num_envs, 1)

        if self.cfg.domain_rand.randomize_base_mass:
            self.payload_masses = torch_rand_float(self.cfg.domain_rand.added_mass_range[0],
                                                   self.cfg.domain_rand.added_mass_range[1],
                                                   (self.num_envs, 1),
                                                   device=self.device)

        if self.cfg.domain_rand.randomize_link_mass:
            self.link_mass_multiplier = torch_rand_float(self.cfg.domain_rand.link_mass_multiplier_range[0],
                                                         self.cfg.domain_rand.link_mass_multiplier_range[1],
                                                         (self.num_envs, self.num_bodies - 1),
                                                         device=self.device)

        if self.cfg.domain_rand.randomize_com:
            self.com_displacements = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0],
                                                      self.cfg.domain_rand.com_displacement_range[1],
                                                      (self.num_envs, 3),
                                                      device=self.device)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[:] = torch_rand_float(self.cfg.domain_rand.friction_range[0],
                                                       self.cfg.domain_rand.friction_range[1],
                                                       (self.num_envs, 1),
                                                       device=self.device)
            self.compliance_coeffs = torch_rand_float(self.cfg.domain_rand.compliance_range[0],
                                                      self.cfg.domain_rand.compliance_range[1],
                                                      (self.num_envs, 1),
                                                      device=self.device)
            self.restitution_coeffs = torch_rand_float(self.cfg.domain_rand.restitution_range[0],
                                                       self.cfg.domain_rand.restitution_range[1],
                                                       (self.num_envs, 1),
                                                       device=self.device)
