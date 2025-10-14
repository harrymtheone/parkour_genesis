from .template_models import UniversalCritic, MixtureOfCritic, AMPType, AMPDiscriminator

from .alg_base import BaseAlgorithm

from .dreamwaq import *

from .odom_amp import *

from .pie import *
from .pie_amp import *
from .pie_amp_edge import *

algorithm_dict = {
    'ppo_dreamwaq': PPODreamWaQ,

    'ppo_odom_amp': PPO_Odom_AMP,

    'ppo_pie_amp': PPO_PIE_AMP,
    'ppo_pie_amp_edge': PPO_PIE_AMP_Edge,
}
