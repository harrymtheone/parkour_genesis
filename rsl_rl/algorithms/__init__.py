from .alg_base import BaseAlgorithm
from .ppo_dream_waq import PPODreamWaQ
from .ppo_phase import PPO_Phase
from .ppo_pie import PPO_PIE
from .ppo_priv import PPO_Priv
from .ppo_wmp import PPO_WMP
from .ppo_dreamer import PPO_Dreamer
from .ppo_zju import PPO_ZJU
from .ppo_zju_multi_critic import PPO_ZJU_Multi_Critic

algorithm_dict = {
    'ppo_priv': PPO_Priv,
    'ppo_dreamwaq': PPODreamWaQ,
    'ppo_pie': PPO_PIE,
    'ppo_zju': PPO_ZJU,
    'ppo_zju_mc': PPO_ZJU_Multi_Critic,
    'ppo_wmp': PPO_WMP,
    'ppo_dreamer': PPO_Dreamer,

    'ppo_phase': PPO_Phase,
}
