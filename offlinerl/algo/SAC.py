import copy
import wandb

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Distribution

from offlinerl.utils.critic import Critic
from offlinerl.utils.tanhpolicy import TanhGaussianPolicy


class SAC:
    def __init__(self, state_shape, action_shape, max_action, args):
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.args = args

        max_steps = args.max_timesteps

        self.actor = TanhGaussianPolicy(state_shape, action_shape, max_action,
                                        args.actor_num_hidden_layers,
                                        args.hidden_layer_dim,
                                        conditioned_sigma = True,
                                        log_sig_max = args.log_sig_max,
                                        log_sig_min = args.log_sig_min).to(args.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.actor_learning_rate)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=int(max_steps))


        self.critic1 = Critic(state_shape, action_shape, args.critic_num_hidden_layers, args.hidden_layer_dim, use_ln=args.use_layernormalization).to(args.device)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=args.critic_learning_rate)
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2 = Critic(state_shape, action_shape, args.critic_num_hidden_layers, args.hidden_layer_dim, use_ln=args.use_layernormalization).to(args.device)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=args.critic_learning_rate)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_clip_grad_norm = args.actor_clip_grad_norm
        self.critic_clip_grad_norm = args.critic_clip_grad_norm

        self.log_alpha = None
        self.alpha_opt = None
        self.target_entropy = None
        if args.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(action_shape).item()
            self.log_alpha = torch.zeros(1,requires_grad=True, device=args.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=args.actor_learning_rate)

        self.num = args.vae_sampling_num
        self.lam = args.critic_penalty_coef

    def sync_weight(self, net_target, net, soft_target_tau = 5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)

    def get_action(self, states):
        states = torch.FloatTensor(states.reshape(1, -1)).to(self.args.device)
        with torch.no_grad():
            return self.actor(states).mode.cpu().data.numpy().flatten()

    def sample_actions(self, obs, requires_grad=False):
        if requires_grad:
            tanh_normal: Distribution = self.actor(obs)
            action = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(action)
        else:
            with torch.no_grad():
                tanh_normal: Distribution = self.actor(obs)
                action = tanh_normal.sample()
                log_prob = tanh_normal.log_prob(action)

        std = tanh_normal.scale
        return action, log_prob, std

    def update_actor(self, obs, actions):
        sampled_actions, log_pi, _ = self.sample_actions(obs, requires_grad=True)

        # update alpha
        if self.args.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = 0
            alpha = 0.01

        q_actions = torch.min(
            self.critic1(obs, sampled_actions),
            self.critic2(obs, sampled_actions),
        )
        q_actions = q_actions.squeeze(-1)
        assert q_actions.shape == (self.args.batch_size,), q_actions.shape
        assert log_pi.shape == (self.args.batch_size, ), log_pi.shape

        if self.args.actor_penalty_coef:
            bc_loss = ((sampled_actions - actions) ** 2).sum(-1)
            policy_loss = (alpha*log_pi - q_actions + self.args.actor_penalty_coef * bc_loss).mean()
        else:
            with torch.no_grad():
                bc_loss = ((sampled_actions - actions) ** 2).sum(-1)
            policy_loss = (alpha*log_pi - q_actions).mean()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self.actor_clip_grad_norm or float("inf"))
        policy_grad_norm = torch.max(torch.stack([p.grad.detach().norm() for p in self.actor.parameters()]))
        self.actor_opt.step()

        if self.args.use_actor_scheduler:
            self.actor_scheduler.step()

        actor_update_data = {"train/actor q": q_actions.mean(),
                             "train/actor loss": policy_loss.mean(),
                             "train/bc loss": bc_loss.mean(),
                             "train/actor gradient": policy_grad_norm,
                             "train/alpha": alpha}

        return actor_update_data

    def update_critic(self, obs, actions, next_obs, rewards, terminals):
        q1_pred = self.critic1(obs, actions)
        q2_pred = self.critic2(obs, actions)
        assert q1_pred.shape == (self.args.batch_size,1), q1_pred.shape
        assert q2_pred.shape == (self.args.batch_size,1), q2_pred.shape

        next_sampled_actions, next_log_pi, next_std = self.sample_actions(next_obs)

        target_q_values = torch.min(
            self.critic1_target(next_obs, next_sampled_actions),
            self.critic2_target(next_obs, next_sampled_actions),
        )
        assert target_q_values.shape == (self.args.batch_size, 1), target_q_values.shape
        assert rewards.shape == (self.args.batch_size, 1), rewards.shape

        q_target = rewards + (1. - terminals) * self.args.discount * target_q_values.detach()
        assert q_target.shape == (self.args.batch_size, 1), q_target.shape

        # calculate loss for ood actions
        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)

        self.critic1_opt.zero_grad()
        qf1_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic1.parameters(), self.critic_clip_grad_norm or float("inf"))
        critic1_grad_norm = torch.max(torch.stack([p.grad.detach().norm() for p in self.critic1.parameters()]))
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        qf2_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.critic2.parameters(), self.critic_clip_grad_norm or float("inf"))
        critic2_grad_norm = torch.max(torch.stack([p.grad.detach().norm() for p in self.critic2.parameters()]))
        self.critic2_opt.step()

        critic_update_data = {
                       "train/q1": q1_pred.mean(),
                       "train/q2": q2_pred.mean(),
                       "train/q target": q_target.mean(),
                       "train/max q target": q_target.max(),
                       "train/q1 critic loss": qf1_loss,
                       "train/q2 critic loss": qf2_loss,
                       "train/critic1 gradient": critic1_grad_norm,
                       "train/critic2 gradient": critic2_grad_norm,
                       "train/next_log_pi mean": next_log_pi.mean(),
                       "train/next_log_pi max": next_log_pi.max(),
                       "train/next_log_pi min": next_log_pi.min(),
                       "train/next_std mean": next_std.mean(),
                       "train/next_std min": next_std.mean(-1).min(),
                       }

        return critic_update_data

    def train(self, states, actions, next_states, rewards, dones, step):
        """
        Step1. Update critic
        """
        critic_update_data = self.update_critic(states, actions, next_states, rewards, dones)

        """
        Step2. Update actor
        """
        actor_update_data = self.update_actor(states, actions)

        if step % self.args.log_interval == 0:
            logging_data = {**critic_update_data, **actor_update_data}
            logging_data['total_step'] = step
            wandb.log(logging_data)

        """
        Step3. Soft Updates target network
        """
        self.sync_weight(self.critic1_target, self.critic1, self.args.soft_target_tau)
        self.sync_weight(self.critic2_target, self.critic2, self.args.soft_target_tau)
