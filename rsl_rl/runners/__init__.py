from .rl_async_runner import RLAsyncRunner
from .rl_dream_runner import RLDreamRunner
from .rl_dreamer_runner import RL_Dreamer_Runner
from .rl_odom_runner import RLOdomRunner
from .rl_runner import RLRunner
from .rl_amp_runner import RLAmpRunner

runner_list = {
    'rl_scan': RLRunner,
    'rl_dream': RLDreamRunner,
    'rl_async': RLAsyncRunner,
    'rl_dreamer': RL_Dreamer_Runner,
    'rl_odom': RLOdomRunner,

    'rl_amp': RLAmpRunner,
}
