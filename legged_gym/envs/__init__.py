from .T1.t1_dreamwaq_environment import T1DreamWaqEnvironment
from .T1.t1_zju_environment import T1ZJUEnvironment
from .T1.t1_pie_environment import T1PIEEnvironment
from .T1.t1_gait_environment import T1GaitEnvironment
from .T1.config.t1_dreamwaq_config import T1DreamWaqCfg, T1DreamWaqCfgPPO
from .T1.config.t1_zju_config import T1ZJUCfg, T1ZJUCfgPPO
from .T1.config.t1_pie_config import T1PIECfg, T1PIECfgPPO
from .T1.config.t1_gait_config import T1GaitCfg, T1GaitCfgPPO

from .pdd.config.pdd_dreamwaq_config import PddDreamWaqCfg, PddDreamWaqCfgPPO, PddDreamWaqGRUCfgPPO
from .pdd.config.pdd_scan_config import PddScanCfg, PddScanCfgPPO, PddScanStairCfg, PddScanStairCfgPPO
from .pdd.config.pdd_zju_config import PddZJUCfg, PddZJUCfgPPO
from .pdd.pdd_dreamwaq_environment import PddDreamWaqEnvironment
from .pdd.pdd_scan_environment import PddScanEnvironment
from .pdd.pdd_zju_environment import PddZJUEnvironment

from .go1.go1_zju_environment import Go1ZJUEnvironment
from .go1.config.go1_zju_config import Go1ZJUCfg, Go1ZJUCfgPPO

task_list = [
    ['pdd_scan', PddScanEnvironment, PddScanCfg(), PddScanCfgPPO()],
    ['pdd_scan_stair', PddScanEnvironment, PddScanStairCfg(), PddScanStairCfgPPO()],

    ['pdd_dreamwaq', PddDreamWaqEnvironment, PddDreamWaqCfg(), PddDreamWaqCfgPPO()],
    ['pdd_dreamwaq_gru', PddDreamWaqEnvironment, PddDreamWaqCfg(), PddDreamWaqGRUCfgPPO()],

    ['pdd_zju', PddZJUEnvironment, PddZJUCfg(), PddZJUCfgPPO()],

    ['t1_dreamwaq', T1DreamWaqEnvironment, T1DreamWaqCfg(), T1DreamWaqCfgPPO()],
    ['t1_zju', T1ZJUEnvironment, T1ZJUCfg(), T1ZJUCfgPPO()],
    ['t1_pie', T1PIEEnvironment, T1PIECfg(), T1PIECfgPPO()],

    ['t1_gait', T1GaitEnvironment, T1GaitCfg(), T1GaitCfgPPO()],

    ['go1_zju', Go1ZJUEnvironment, Go1ZJUCfg(), Go1ZJUCfgPPO()]

]
