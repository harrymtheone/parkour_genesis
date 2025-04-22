from enum import Enum

from .base_wrapper import DriveMode
from legged_gym.simulator.sensors.sensor_manager import SensorManager


class SimulatorType(Enum):
    IsaacGym = 0
    Genesis = 1
    IsaacSim = 2


def get_simulator(simulator: SimulatorType):
    if simulator == SimulatorType.IsaacGym:
        from .isaacgym_wrapper import IsaacGymWrapper
        return IsaacGymWrapper

    elif simulator == SimulatorType.Genesis:
        from .genesis_wrapper import GenesisWrapper
        return GenesisWrapper

    elif simulator == SimulatorType.IsaacSim:
        from .isaacsim_wrapper import IsaacSimWrapper
        return IsaacSimWrapper

    else:
        raise NotImplementedError
