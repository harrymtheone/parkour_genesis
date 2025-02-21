class BaseAlgorithm:
    def act(self, obs, obs_critic, **kwargs):
        raise NotImplementedError

    def process_env_step(self, rewards, dones, infos, *args):
        raise NotImplementedError

    def reset(self, dones):
        raise NotImplementedError

    def compute_returns(self, last_critic_obs):
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError

    def play_act(self, obs, **kwargs):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def load(self, loaded_dict, load_optimizer=True):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    @staticmethod
    def recurrent_wrapper(func):
        def wrapper(alg_obj, *args):
            n_steps = args[0].size(0)
            rtn = func(alg_obj, *[arg.flatten(0, 1) for arg in args])
            return (r.unflatten(0, (n_steps, -1)) for r in rtn)

        return wrapper
