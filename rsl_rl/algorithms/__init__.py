from .alg_base import BaseAlgorithm

from .ppo_zju import PPO_ZJU
from .ppo_pie import PPO_PIE
from .ppo_priv import PPO_Priv
from .ppo_dream_waq import PPODreamWaQ
from .ppo_dreamer import PPODreamer

algorithm_dict = {
    'ppo_priv': PPO_Priv,
    'ppo_dreamwaq': PPODreamWaQ,
    'ppo_pie': PPO_PIE,
    'ppo_zju': PPO_ZJU,
    'ppo_dreamer': PPODreamer,
}
