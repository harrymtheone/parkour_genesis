from .pdd.config.pdd_dreamwaq_config import PddDreamWaqCfg, PddDreamWaqCfgPPO, PddDreamWaqDRCfg, PddDreamWaqDRCfgPPO
from .pdd.config.pdd_scan_config import PddScanCfg, PddScanCfgPPO, PddScanStairCfg, PddScanStairCfgPPO
from .pdd.config.pdd_zju_config import PddZJUCfg, PddZJUCfgPPO
from .pdd.pdd_dreamwaq_environment import PddDreamWaqEnvironment
from .pdd.pdd_scan_environment import PddScanEnvironment
from .pdd.pdd_zju_environment import PddZJUEnvironment


task_list = [
    ['pdd_scan', PddScanEnvironment, PddScanCfg(), PddScanCfgPPO()],
    ['pdd_scan_stair', PddScanEnvironment, PddScanStairCfg(), PddScanStairCfgPPO()],

    ['pdd_dreamwaq', PddDreamWaqEnvironment, PddDreamWaqCfg(), PddDreamWaqCfgPPO()],
    ['pdd_dreamwaq_dr', PddDreamWaqEnvironment, PddDreamWaqDRCfg(), PddDreamWaqDRCfgPPO()],

    ['pdd_zju', PddZJUEnvironment, PddZJUCfg(), PddZJUCfgPPO()],

]
