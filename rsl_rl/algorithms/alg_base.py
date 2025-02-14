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
