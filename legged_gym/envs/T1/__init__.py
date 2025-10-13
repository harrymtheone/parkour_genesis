from .t1_dreamwaq import *
from .t1_odom_amp import *
from .t1_odom_neg import *
from .t1_pie_amp import *

t1_tasks = {
    't1_dreamwaq': (T1DreamWaqEnv, T1DreamWaqCfg()),

    't1_odom_neg': (T1OdomNegEnvironment, T1_Odom_Neg_Cfg()),

    't1_odom_amp': (T1OdomAmpEnv, T1_Odom_AMP_Cfg()),

    't1_pie_amp': (T1PIEAmpEnv, T1_PIE_AMP_Cfg()),
}
