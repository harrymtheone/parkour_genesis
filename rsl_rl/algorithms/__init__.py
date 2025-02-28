from .alg_base import BaseAlgorithm

from .ppo import PPO
from .ppo_zju import PPO_ZJU
from .ppo_pie import PPO_PIE
from .ppo_vbbl import PPO_Vbbl
from .ppo_dream_waq import PPODreamWaQ

algorithm_dict = {
    'ppo_scan': PPO,
    'ppo_zju': PPO_ZJU,
    'ppo_pie': PPO_PIE,
    'ppo_vbbl': PPO_Vbbl,
    'ppo_dreamwaq': PPODreamWaQ,
}
