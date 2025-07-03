# from .A1.a1_wmp_environment import A1WMPEnvironment
# from .A1.config.a1_wmp_config import A1_WMP_Cfg
#
# from .go1 import Go1ZJUEnvironment, Go1WMPEnvironment
# from .go1 import Go1_ZJU_Cfg, Go1_ZJU_Pit_Cfg, Go1_WMP_Cfg, Go1_Dreamer_Cfg
# from .pdd.config.pdd_dreamwaq_config import PddDreamWaqCfg, PddDreamWaqCfgPPO, PddDreamWaqGRUCfgPPO
# from .pdd.config.pdd_scan_config import PddScanCfg, PddScanCfgPPO, PddScanStairCfg, PddScanStairCfgPPO
# from .pdd.config.pdd_zju_config import PddZJUCfg, PddZJUCfgPPO
# from .pdd.pdd_dreamwaq_environment import PddDreamWaqEnvironment
# from .pdd.pdd_scan_environment import PddScanEnvironment
# from .pdd.pdd_zju_environment import PddZJUEnvironment
#
# task_list = [
#     # ['pdd_scan', PddScanEnvironment, PddScanCfg(), PddScanCfgPPO()],  # TODO: not finished yet
#     # ['pdd_scan_stair', PddScanEnvironment, PddScanStairCfg(), PddScanStairCfgPPO()],  # TODO: not finished yet
#     #
#     # ['pdd_dreamwaq', PddDreamWaqEnvironment, PddDreamWaqCfg(), PddDreamWaqCfgPPO()],  # TODO: not finished yet
#     #
#     # ['pdd_zju', PddZJUEnvironment, PddZJUCfg(), PddZJUCfgPPO()],  # TODO: not finished yet
#
#     ['go1_zju', Go1ZJUEnvironment, Go1_ZJU_Cfg()],
#     ['go1_zju_pit', Go1ZJUEnvironment, Go1_ZJU_Pit_Cfg()],
#     ['go1_wmp', Go1WMPEnvironment, Go1_WMP_Cfg()],
#     ['go1_dreamer', Go1WMPEnvironment, Go1_Dreamer_Cfg()],
#
#     # ['a1_zju', Go1ZJUEnvironment, Go1_ZJU_Cfg(), Go1_ZJU_CfgPPO()],  # TODO: not finished yet
#     # ['a1_zju_pit', Go1ZJUEnvironment, Go1_ZJU_Pit_Cfg(), Go1_ZJU_VAE_Pit_CfgPPO()],  # TODO: not finished yet
#     ['a1_wmp', A1WMPEnvironment, A1_WMP_Cfg()],  # TODO: not finished yet
# ]

from .T1 import t1_tasks

tasks = {}
tasks.update(t1_tasks)
