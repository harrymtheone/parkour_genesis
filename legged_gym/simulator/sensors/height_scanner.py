from dataclasses import MISSING

from legged_gym.simulator.sensors.sensor_base import SensorBase


class HeightScanner(SensorBase):
    def __init__(self, cfg_dict, device, mesh_id, sim):
        super().__init__(cfg_dict, device, mesh_id, sim)


MISSING
