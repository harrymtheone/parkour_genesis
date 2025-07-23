from .alg_base import BaseAlgorithm
from .ppo_dream_waq import PPODreamWaQ
from .ppo_phase import PPO_Phase
from .ppo_pie import PPO_PIE
from .ppo_priv import PPO_Priv
from .ppo_wmp import PPO_WMP
from .ppo_dreamer import PPO_Dreamer
from .ppo_zju import PPO_ZJU
from .ppo_zju_multi_critic import PPO_ZJU_Multi_Critic
from .ppo_bbm import PPO_BBM
from rsl_rl.algorithms.odometry import PPO_Odom, PPO_Odom_ROA

from rsl_rl.algorithms.distiller import Distiller

algorithm_dict = {
    'distiller': Distiller,
    'ppo_odom': PPO_Odom,
    'ppo_odom_roa': PPO_Odom_ROA,

    'ppo_priv': PPO_Priv,
    'ppo_dreamwaq': PPODreamWaQ,
    'ppo_pie': PPO_PIE,
    'ppo_zju': PPO_ZJU,
    'ppo_zju_mc': PPO_ZJU_Multi_Critic,
    'ppo_wmp': PPO_WMP,
    'ppo_dreamer': PPO_Dreamer,

    'ppo_bbm': PPO_BBM,
    'ppo_phase': PPO_Phase,
}
