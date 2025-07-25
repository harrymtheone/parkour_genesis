from .config.pdd_dreamwaq_config import PddDreamWaqCfg
from .config.pdd_odom_config import PddOdomCfg, PddOdomStairCfg, PddOdomFinetuneCfg

from .pdd_dreamwaq_environment import PddDreamWaqEnvironment
from .pdd_odom_environment import PddOdomEnvironment


pdd_tasks = {
    'pdd_dreamwaq': (PddDreamWaqEnvironment, PddDreamWaqCfg()),

    'pdd_odom': (PddOdomEnvironment, PddOdomCfg()),
    'pdd_odom_stair': (PddOdomEnvironment, PddOdomStairCfg()),
    'pdd_odom_finetune': (PddOdomEnvironment, PddOdomFinetuneCfg()),
}