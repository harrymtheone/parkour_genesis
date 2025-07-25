from .config.g1_base_config import G1BaseCfg
from .config.g1_dreamwaq_config import G1DreamWaqCfg, T1DreamWaqPhase2Cfg
from .config.g1_multi_critic_config import G1_Multi_Critic_Cfg, T1_Multi_Critic_Stair_Cfg, T1_Multi_Critic_Parkour_Cfg
from .config.g1_odom_config import G1OdomCfg, G1OdomStairCfg, G1OdomFinetuneCfg

from .g1_base_env import G1BaseEnv
from .g1_dreamwaq_environment import G1DreamWaqEnvironment
from .g1_zju_environment import G1ZJUEnvironment
from .g1_odom_environment import G1OdomEnvironment


g1_tasks = {
    'g1_base': (G1BaseEnv, G1BaseCfg()),
    
    'g1_dreamwaq': (G1DreamWaqEnvironment, G1DreamWaqCfg()),
    'g1_dreamwaq_phase2': (G1DreamWaqEnvironment, T1DreamWaqPhase2Cfg()),
    
    'g1_odom': (G1OdomEnvironment, G1OdomCfg()),
    'g1_odom_stair': (G1OdomEnvironment, G1OdomStairCfg()),
    'g1_odom_finetune': (G1OdomEnvironment, G1OdomFinetuneCfg()),
    
    # 'g1_multi_critic': (G1ZJUEnvironment, G1_Multi_Critic_Cfg()),
    # 'g1_multi_critic_stair': (G1ZJUEnvironment, T1_Multi_Critic_Stair_Cfg()),
    # 'g1_multi_critic_parkour': (G1ZJUEnvironment, T1_Multi_Critic_Parkour_Cfg()),
} 