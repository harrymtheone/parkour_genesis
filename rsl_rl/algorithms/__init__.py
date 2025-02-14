from .alg_base import BaseAlgorithm

from .ppo import PPO
from .ppo_zju import PPO_ZJU
from .ppo_dream_waq import PPODreamWaQ

algorithm_dict = {
    'ppo_scan': PPO,
    'ppo_zju': PPO_ZJU,
    'ppo_dreamwaq': PPODreamWaQ,
}
