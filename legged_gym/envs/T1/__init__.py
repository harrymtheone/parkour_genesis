from .config.t1_dreamwaq_config import T1DreamWaqCfg, T1DreamWaqPhase2Cfg
from .config.t1_multi_critic_config import T1_Multi_Critic_Cfg
from .config.t1_multi_critic_config import T1_Multi_Critic_Stair_Cfg
from .config.t1_odom_amp_config import T1_Odom_AMP_Cfg
from .config.t1_odom_config import T1_Odom_Cfg, T1_Odom_Stair_Cfg, T1_Odom_Finetune_Cfg
from .config.t1_odom_neg_config import T1_Odom_Neg_Cfg, T1_Odom_Stair_Neg_Cfg, T1_Odom_Neg_Finetune_Cfg
from .config.t1_pie_config import T1_PIE_Cfg, T1_PIE_Stair_Cfg
from .config.t1_pie_amp_config import T1_PIE_AMP_Cfg
from .t1_dreamwaq_environment import T1DreamWaqEnvironment
from .t1_odom_amp_environment import T1OdomAmpEnv
from .t1_odom_environment import T1OdomEnvironment
from .t1_odom_neg_rew_environment import T1OdomNegEnvironment
from .t1_pie_environment import T1PIEEnvironment
from .t1_zju_environment import T1ZJUEnvironment
from .t1_pie_amp_environment import T1PIEAmpEnv

t1_tasks = {
    # 't1_priv': (T1PrivEnvironment, T1_Priv_Cfg()),
    # 't1_priv_stair': (T1PrivEnvironment, T1_Priv_Stair_Cfg()),
    # 't1_priv_dis': (T1PrivEnvironment, T1_Priv_Distil_Cfg()),

    't1_dreamwaq': (T1DreamWaqEnvironment, T1DreamWaqCfg()),
    't1_dreamwaq_p2': (T1DreamWaqEnvironment, T1DreamWaqPhase2Cfg()),

    't1_pie': (T1PIEEnvironment, T1_PIE_Cfg()),
    't1_pie_stair': (T1PIEEnvironment, T1_PIE_Stair_Cfg()),

    # 't1_zju': (T1ZJUEnvironment, T1_ZJU_Cfg()),
    # 't1_zju_stair': (T1ZJUEnvironment, T1_ZJU_Stair_Cfg()),
    # 't1_zju_parkour': (T1ZJUEnvironment, T1_ZJU_Parkour_Cfg()),

    't1_mc': (T1ZJUEnvironment, T1_Multi_Critic_Cfg()),
    't1_mc_stair': (T1ZJUEnvironment, T1_Multi_Critic_Stair_Cfg()),

    # 't1_bbm': (T1_BBM_Environment, T1_BBM_Cfg()),

    't1_odom': (T1OdomEnvironment, T1_Odom_Cfg()),
    't1_odom_stair': (T1OdomEnvironment, T1_Odom_Stair_Cfg()),
    't1_odom_finetune': (T1OdomEnvironment, T1_Odom_Finetune_Cfg()),

    't1_odom_neg': (T1OdomNegEnvironment, T1_Odom_Neg_Cfg()),
    't1_odom_neg_stair': (T1OdomNegEnvironment, T1_Odom_Stair_Neg_Cfg()),
    't1_odom_neg_finetune': (T1OdomNegEnvironment, T1_Odom_Neg_Finetune_Cfg()),

    't1_odom_amp': (T1OdomAmpEnv, T1_Odom_AMP_Cfg()),

    't1_pie_amp': (T1PIEAmpEnv, T1_PIE_AMP_Cfg()),

    # 't1_phase': (T1_Phase_Environment, T1_Phase_Cfg()),
    # 't1_phase_stair': (T1_Phase_Environment, T1_Phase_Stair_Cfg()),
}
