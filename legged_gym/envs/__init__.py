from .T1.t1_priv_environment import T1PrivEnvironment
from .T1.config.t1_priv_config import T1PrivCfg

from .T1.t1_dreamwaq_environment import T1DreamWaqEnvironment
from .T1.config.t1_dreamwaq_config import T1DreamWaqCfg

from .T1.t1_pie_environment import T1PIEEnvironment
from .T1.config.t1_pie_config import T1PIECfg

from .T1.t1_zju_environment import T1ZJUEnvironment
from .T1.config.t1_zju_config import T1_ZJU_Cfg, T1_ZJU_Stair_Cfg, T1_ZJU_Parkour_Cfg
from .T1.config.t1_multi_critic_config import T1_Multi_Critic_Cfg
from .T1.config.t1_multi_critic_config import T1_Multi_Critic_Stair_Cfg

from .pdd.pdd_dreamwaq_environment import PddDreamWaqEnvironment
from .pdd.config.pdd_dreamwaq_config import PddDreamWaqCfg, PddDreamWaqCfgPPO, PddDreamWaqGRUCfgPPO

from .pdd.pdd_scan_environment import PddScanEnvironment
from .pdd.config.pdd_scan_config import PddScanCfg, PddScanCfgPPO, PddScanStairCfg, PddScanStairCfgPPO

from .pdd.config.pdd_zju_config import PddZJUCfg, PddZJUCfgPPO
from .pdd.pdd_zju_environment import PddZJUEnvironment

from .go1.go1_zju_environment import Go1ZJUEnvironment
from .go1.config.go1_zju_config import Go1_ZJU_Cfg, Go1_ZJU_Pit_Cfg

from .A1.a1_dreamer_environment import A1DreamerEnvironment
from .A1.config.a1_dreamer_config import A1_Dreamer_Cfg

task_list = [
    # ['pdd_scan', PddScanEnvironment, PddScanCfg(), PddScanCfgPPO()],  # TODO: not finished yet
    # ['pdd_scan_stair', PddScanEnvironment, PddScanStairCfg(), PddScanStairCfgPPO()],  # TODO: not finished yet
    #
    # ['pdd_dreamwaq', PddDreamWaqEnvironment, PddDreamWaqCfg(), PddDreamWaqCfgPPO()],  # TODO: not finished yet
    #
    # ['pdd_zju', PddZJUEnvironment, PddZJUCfg(), PddZJUCfgPPO()],  # TODO: not finished yet

    ['t1_priv', T1PrivEnvironment, T1PrivCfg()],
    ['t1_dreamwaq', T1DreamWaqEnvironment, T1DreamWaqCfg()],
    ['t1_pie', T1PIEEnvironment, T1PIECfg()],

    ['t1_zju', T1ZJUEnvironment, T1_ZJU_Cfg()],
    ['t1_zju_stair', T1ZJUEnvironment, T1_ZJU_Stair_Cfg()],
    ['t1_zju_parkour', T1ZJUEnvironment, T1_ZJU_Parkour_Cfg()],

    ['t1_mc', T1ZJUEnvironment, T1_Multi_Critic_Cfg()],
    ['t1_mc_stair', T1ZJUEnvironment, T1_Multi_Critic_Stair_Cfg()],

    ['go1_zju', Go1ZJUEnvironment, Go1_ZJU_Cfg()],
    ['go1_zju_pit', Go1ZJUEnvironment, Go1_ZJU_Pit_Cfg()],

    # ['a1_zju', Go1ZJUEnvironment, Go1_ZJU_Cfg(), Go1_ZJU_CfgPPO()],  # TODO: not finished yet
    # ['a1_zju_pit', Go1ZJUEnvironment, Go1_ZJU_Pit_Cfg(), Go1_ZJU_VAE_Pit_CfgPPO()],  # TODO: not finished yet
    ['a1_wmp', Go1ZJUEnvironment, A1_Dreamer_Cfg()],  # TODO: not finished yet


]
