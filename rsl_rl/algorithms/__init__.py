from .alg_base import BaseAlgorithm

from .ppo import PPO
from .ppo_zju import PPO_ZJU
from .ppo_pie import PPO_PIE
from .ppo_priv import PPO_Priv
from .ppo_priv_vel import PPO_Priv_Vel
from .ppo_dream_waq import PPODreamWaQ

algorithm_dict = {
    'ppo_scan': PPO,
    'ppo_zju': PPO_ZJU,
    'ppo_pie': PPO_PIE,
    'ppo_priv': PPO_Priv,
    'ppo_priv_vel': PPO_Priv_Vel,
    'ppo_dreamwaq': PPODreamWaQ,
}
