from enum import Enum


class SimulatorType(Enum):
    IsaacGym = 0
    Genesis = 1


def get_simulator(simulator: SimulatorType):
    if simulator == SimulatorType.IsaacGym:
        from .isaacgym_wrapper import IsaacGymWrapper
        return IsaacGymWrapper

    elif simulator == SimulatorType.Genesis:
        from .genesis_wrapper import GenesisWrapper
        return GenesisWrapper
