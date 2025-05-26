import torch
from torch import nn

from .decoder import WMDecoder
from .encoder import WMEncoder
from .recurrent_model import RecurrentModel
from ..utils import gru_wrapper


class RSSM(nn.Module):
    def __init__(self, env_cfg, train_cfg):
        super().__init__()

        self.cfg = train_cfg.world_model

        self.encoder = WMEncoder(env_cfg, train_cfg)
        self.decoder = WMDecoder(env_cfg, train_cfg)
        self.recurrent_model = RecurrentModel(env_cfg, train_cfg)

    def step(self, proprio, depth, prev_actions, dones):
        obs_enc = self.encoder(proprio, depth)

        if torch.any(dones):
            self.recurrent_model.init_model_state(dones)

        return self.recurrent_model.step(obs_enc, prev_actions)

    def train_step(self, prop, depth, action_his, state_deter, state_stoch):
        obs_enc = gru_wrapper(self.encoder.forward, prop, depth)

        state_deter_new, prior = self.recurrent_model.imagination_step(state_deter, state_stoch, action_his)

        post = gru_wrapper(self.recurrent_model.observation_step, state_deter_new, obs_enc)

        ot1, depth, rew = gru_wrapper(self.decoder, state_deter_new, post)

        return prior, post, ot1, depth, rew




    def reset(self, dones):
        self.recurrent_model.init_wm_state(dones)

    def get_deter(self):
        return self.recurrent_model.state_deter

    def get_stoch(self):
        return self.recurrent_model.state_stoch

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self.use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self.cfg.kl_free
                dyn_scale = self.cfg.dyn_scale
                rep_scale = self.cfg.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self.cfg.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self.use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        # obs = obs.copy()
        # obs["image"] = torch.Tensor(obs["image"]) / 255.0

        # discount in obs seems useless
        # if "discount" in obs:
        #     obs["discount"] *= self._config.discount
        # (batch_size, batch_length) -> (batch_size, batch_length, 1)
        # obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        # assert "is_terminal" in obs
        # obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self.cfg.device) for k, v in obs.items()}
        return obs
