import isaacgym, torch
import os

from torch import nn

from legged_gym.utils.task_registry import TaskRegistry
from rsl_rl.modules.model_zju import ObsEnc, Recon, Actor
from rsl_rl.modules.model_zju_vae import LocoTransformerVAE


class Estimator(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()

        self.obs_enc = ObsEnc(env_cfg.len_prop_his * env_cfg.n_proprio)
        self.reconstructor = Recon()
        self.transformer = LocoTransformerVAE()
        self.actor = Actor(env_cfg, train_cfg)

        self.std = nn.Parameter(train_cfg.estimator.init_noise_std * torch.ones(env_cfg.num_actions))

    def forward(self, proprio, prop_his, depth, recon_prev):
        # encode history proprio
        latent_obs = self.obs_enc(prop_his)

        # compute reconstruction
        recon_rough, recon_refine = self.reconstructor(depth, recon_prev.unsqueeze(1), latent_obs)

        # cross-model mixing using transformer
        est_mu, est_log_var, ot1 = self.transformer(recon_refine, latent_obs)

        # compute action
        mean = self.actor(proprio, est_mu)

        # output action
        return mean, recon_rough.squeeze(1), recon_refine.squeeze(1), est_mu


if __name__ == '__main__':
    exptid = 'zju_vae_pit03r'
    checkpoint = 39900

    task_registry = TaskRegistry()
    env_cfg, train_cfg = task_registry.get_cfg(name='go1_zju')

    # ----------------------------------------------------------------------------------------------------------
    prop = torch.randn(1, env_cfg.env.n_proprio)
    prop_his = torch.randn(1, env_cfg.env.len_prop_his, env_cfg.env.n_proprio)
    depth = torch.randn(1, env_cfg.env.len_depth_his, *reversed(env_cfg.depth.resized))
    recon_prev = torch.randn(1, *env_cfg.env.scan_shape)

    device = torch.device('cpu')
    load_path = os.path.join('../../logs/parkour_new/', exptid, f'model_{checkpoint}.pt')
    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=device, weights_only=True)

    model = Estimator(env_cfg.env, train_cfg)
    model.load_state_dict(state_dict['actor_state_dict'])
    model.eval()

    try:
        os.mkdir('onnx')
    except FileExistsError:
        pass

    torch.onnx.export(model,
                      (prop, prop_his, depth, recon_prev),  # 第一个 JIT 模型
                      "onnx/encoder.onnx",  # 第一个模型的输出文件名
                      export_params=True,  # 导出模型的参数
                      opset_version=17,  # ONNX opset 版本
                      do_constant_folding=True,  # 优化常量折叠
                      input_names=['prop', 'prop_his', 'depth', 'recon_prev'],  # 输入名
                      output_names=['action', 'recon_rough', 'recon_refine', 'est'],  # 输出名
                      )

