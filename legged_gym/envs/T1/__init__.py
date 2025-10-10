from .t1_pie_amp_environment import T1PIEAmpEnv
from .config.t1_pie_amp_config import T1_PIE_AMP_Cfg

t1_tasks = {
    't1_pie_amp': (T1PIEAmpEnv, T1_PIE_AMP_Cfg()),
}
