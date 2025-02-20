from enum import Enum

from .base_wrapper import DriveMode
from legged_gym.simulator.sensors.sensor_manager import SensorManager


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
