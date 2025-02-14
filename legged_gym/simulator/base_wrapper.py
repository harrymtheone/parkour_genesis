import torch
import warp as wp

from legged_gym.utils.math import torch_rand_float


class BaseWrapper:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.suppress_warning = False

        self.device = torch.device(args.device)
        self.num_envs = self.cfg.env.num_envs

        self._body_names = None
        self._dof_names = None
        self.num_bodies = None
        self.num_dof = None

        self.dof_pos_limits = None
        self.torque_limits = None

        self.height_samples = None
        self.height_guidance = None
        self.edge_mask = None

    # ------------------------------------------------- Simulator Interfaces -------------------------------------------------

    def set_root_state(self, env_ids, pos, quat, lin_vel, ang_vel):
        raise NotImplementedError

    def set_dof_state(self, env_ids, dof_pos, dof_vel):
        raise NotImplementedError

    def refresh_variable(self):
        raise NotImplementedError

    def apply_perturbation(self, force, torque):
        raise NotImplementedError

    def step_environment(self):
        raise NotImplementedError

    def control_dof_torque(self, torques):
        raise NotImplementedError

    def get_full_names(self, names, is_link) -> list:
        if is_link:
            if type(names) is str:
                names = [names]

            full_names = []
            for n in names:
                full_names.extend([s for s in self._body_names if n in s])
            return full_names

        else:
            return [s for s in self._dof_names if names in s]

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
    def link_quat(self):
        raise NotImplementedError

    @property
    def link_vel(self):
        raise NotImplementedError

    @property
    def contact_forces(self):
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
            self.friction_coeffs = torch_rand_float(self.cfg.domain_rand.friction_range[0],
                                                    self.cfg.domain_rand.friction_range[1],
                                                    (self.num_envs, 1),
                                                    device=self.device).repeat(1, self.num_bodies)
            self.restitution_coeffs = torch_rand_float(self.cfg.domain_rand.restitution_range[0],
                                                       self.cfg.domain_rand.restitution_range[1],
                                                       (self.num_envs, 1),
                                                       device=self.device).repeat(1, self.num_bodies)

