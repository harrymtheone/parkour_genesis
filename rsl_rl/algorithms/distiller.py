import torch

from rsl_rl.algorithms import BaseAlgorithm, PPO_Priv, PPO_PIE


class Distiller(BaseAlgorithm):
    def __init__(self, task_cfg, device, **kwargs):
        super().__init__()

        self.teacher = PPO_Priv(task_cfg, device, **kwargs)

        from legged_gym.envs.T1.config.t1_pie_config import T1_PIE_Stair_Cfg
        pie_cfg = T1_PIE_Stair_Cfg()
        self.student = PPO_PIE(pie_cfg, device, **kwargs)

        self.optimizer = torch.optim.Adam(self.student.actor.parameters(), lr=1e-4)
        self.mse_loss = torch.nn.MSELoss()

        self.actions_student = []
        self.actions_teacher = []

    def act(self, obs, obs_critic, **kwargs):
        with torch.enable_grad():
            self.actions_student.append(self.student.actor.act(obs, eval_=True))

        with torch.no_grad():
            self.actions_teacher.append(self.teacher.actor.act(obs_critic, eval_=True))

        return self.actions_student[-1].detach()
        # return self.actions_teacher[-1].detach()

    def process_env_step(self, rewards, dones, infos, *args):
        self.student.actor.reset(dones)

    def compute_returns(self, last_critic_obs):
        pass

    def update(self, **kwargs):
        actions_student = torch.cat(self.actions_student, dim=0)
        actions_teacher = torch.cat(self.actions_teacher, dim=0)

        loss = self.mse_loss(actions_student, actions_teacher)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.actions_student.clear()
        self.actions_teacher.clear()

        self.student.actor.detach_hidden_states()

        return {
            'Loss/behavior_cloning': loss.item(),
        }

    def train(self):
        self.teacher.train()
        self.student.train()

    def load(self, loaded_dict, load_optimizer=True):
        self.teacher.load(loaded_dict, load_optimizer)

    def save(self):
        return {'actor_state_dict': self.student.actor.state_dict()}
