from .t1_dreamwaq import *
from .t1_odom_amp import *
from .t1_odom_neg import *
from .t1_pie_amp import *
from .t1_pie import *

t1_tasks = {
    't1_dreamwaq': (T1_DreamWaq_Env, T1DreamWaqCfg()),

    't1_odom_neg': (T1_Odom_Neg_Env, T1_Odom_Neg_Cfg()),

    't1_odom_amp': (T1_Odom_Amp_Env, T1_Odom_AMP_Cfg()),

    't1_pie': (T1_PIE_Env, T1_PIE_Cfg()),
    't1_pie_amp': (T1_PIE_Amp_Env, T1_PIE_AMP_Cfg()),
}
