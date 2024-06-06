import copy
import wandb

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Normal, kl_divergence
from torch.distributions import Distribution

from offlinerl.utils.critic import Critic
from offlinerl.utils.tanhpolicy import TanhGaussianPolicy

from offlinerl.algo.vae import VAE

class SCQ:
    def __init__(self, state_shape, action_shape, max_action, args):
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.args = args

        max_steps = args.max_timesteps

        self.vae = VAE(state_shape, action_shape, args.vae_hidden_layer_dim, args.vae_num_hidden_layers, max_action).to(args.device)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=args.vae_learning_rate)

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

        if args.lagrange_tau:
            self.log_lam = torch.zeros(1, requires_grad=True)
            self.lam_optimizer = optim.Adam(params=[self.log_lam], lr=3e-4)
        elif args.critic_penalty_coef is not None:
            self.lam = args.critic_penalty_coef
        else:
            raise NotImplementedError

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

    def update_vae(self, obs, actions):
        # train vae
        vae_dist, _action = self.vae(obs, actions)
        kl_loss = kl_divergence(vae_dist, Normal(0, 1)).sum(dim=-1).mean()
        recon_loss = ((actions - _action) ** 2).sum(dim=-1).mean()
        vae_loss = kl_loss + recon_loss

        self.vae_optim.zero_grad()
        vae_loss.backward()
        self.vae_optim.step()

    def get_ood_coeff(self, obs, actions, idd_action_threshold):
        with torch.no_grad():
            ood_action_dist = self.vae.calc_dist(obs, actions) # calculate distance from the dataset
            ood_idx = torch.where(ood_action_dist>idd_action_threshold)[0]
            coeff = torch.zeros_like(ood_action_dist, requires_grad=False)
            coeff[ood_idx] = 1.0

        return coeff

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

    def calculate_V(self, obs, network):
        batch_size = obs.shape[0]
        action_dim = np.prod(self.action_shape)

        # step1. sample behavior actions (pi_beta)
        with torch.no_grad():
            behavior_actions = self.vae.decode_multiple(obs, num=self.num)
        assert behavior_actions.shape == (self.num, batch_size, action_dim), behavior_actions.shape

        behavior_actions = behavior_actions.reshape(self.num*batch_size, action_dim)

        # step2. calculate Q values for behavior actions
        obs_repeat = obs.repeat((self.num,1,1)).reshape(-1, obs.shape[-1])
        with torch.no_grad():
            pred_Q = network(obs_repeat, behavior_actions)
        pred_Q = pred_Q.reshape(self.num, batch_size)

        # step3. calculate mean Q
        pred_V = torch.mean(pred_Q, dim=0)
        assert pred_V.shape == (batch_size, ), pred_V.shape

        return pred_V.detach()

    def update_lambda(self, obs, loss_coeff, q1_ood, q2_ood):
        if self.args.lagrange_tau:
            v1 = self.calculate_V(obs, self.critic1)
            v2 = self.calculate_V(obs, self.critic2)
            with torch.no_grad():
                tau = self.args.lagrange_tau
                lam_error1 = loss_coeff*(q1_ood - v1)
                lam_error2 = loss_coeff*(q2_ood - v2)
                lam_error1 = lam_error1.mean() - tau
                lam_error2 = lam_error2.mean() - tau
                total_lam_error = -0.5 * (lam_error1 + lam_error2)

            lagrange_lam = torch.clamp(self.log_lam.exp(), min=0.0, max=1000000.0).to(self.args['device'])

            self.lam_optimizer.zero_grad()
            lam_loss = total_lam_error * lagrange_lam
            lam_loss.backward(retain_graph=True)
            self.lam_optimizer.step()

            return lagrange_lam.detach(), total_lam_error

        return self.lam, 0.0

    def update_critic(self, obs, actions, next_obs, rewards, terminals, idd_action_threshold):
        q1_pred = self.critic1(obs, actions)
        q2_pred = self.critic2(obs, actions)
        assert q1_pred.shape == (self.args.batch_size,1), q1_pred.shape
        assert q2_pred.shape == (self.args.batch_size,1), q2_pred.shape

        next_sampled_actions, next_log_pi, next_std = self.sample_actions(next_obs)
        next_log_pi = next_log_pi.unsqueeze(-1)

        # sample OOD actions
        curr_sampled_actions, _, _ = self.sample_actions(obs)
        curr_loss_coeff = self.get_ood_coeff(obs, curr_sampled_actions, idd_action_threshold)
        next_loss_coeff = self.get_ood_coeff(next_obs, next_sampled_actions, idd_action_threshold)
        loss_coeff = torch.cat([curr_loss_coeff, next_loss_coeff])
        assert loss_coeff.shape == (obs.shape[0] + next_obs.shape[0], ), loss_coeff.shape

        target_q_values = torch.min(
            self.critic1_target(next_obs, next_sampled_actions),
            self.critic2_target(next_obs, next_sampled_actions),
        )
        assert target_q_values.shape == (self.args.batch_size, 1), target_q_values.shape
        assert next_log_pi.shape == (self.args.batch_size, 1), next_log_pi.shape
        assert rewards.shape == (self.args.batch_size, 1), rewards.shape

        q_target = rewards + (1. - terminals) * self.args.discount * target_q_values.detach()
        assert q_target.shape == (self.args.batch_size, 1), q_target.shape

        extended_obs = torch.cat([obs, next_obs])
        extended_acs = torch.cat([curr_sampled_actions, next_sampled_actions])

        ## OOD Q1
        q1_ood_pred = self.critic1(extended_obs, extended_acs).squeeze(-1)
        assert q1_ood_pred.shape == (obs.shape[0] + next_obs.shape[0], ), q1_ood_pred.shape

        ## OOD Q2
        q2_ood_pred = self.critic2(extended_obs, extended_acs).squeeze(-1)
        assert q2_ood_pred.shape == (obs.shape[0] + next_obs.shape[0], ), q2_ood_pred.shape

        # update lambda
        lam, total_lam_error = self.update_lambda(extended_obs, loss_coeff, q1_ood_pred, q2_ood_pred)

        # calculate loss for ood actions
        qf1_bellman_loss = F.mse_loss(q1_pred, q_target)
        qf2_bellman_loss = F.mse_loss(q2_pred, q_target)
        qf1_ood_loss = (loss_coeff*q1_ood_pred).mean()
        qf2_ood_loss = (loss_coeff*q2_ood_pred).mean()
        qf1_loss = qf1_bellman_loss + lam * qf1_ood_loss
        qf2_loss = qf2_bellman_loss + lam * qf2_ood_loss

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

        ood_ratio = (torch.sum(loss_coeff)/(len(obs)+len(next_obs)))
        critic_update_data = {
                       "train/q1": q1_pred.mean(),
                       "train/q2": q2_pred.mean(),
                       "train/q target": q_target.mean(),
                       "train/max q target": q_target.max(),
                       "train/q1 ood": q1_ood_pred.mean(),
                       "train/q2 ood": q1_ood_pred.mean(),
                       "train/q1 critic loss": qf1_loss,
                       "train/q2 critic loss": qf2_loss,
                       "train/q1 bellman loss": qf1_bellman_loss,
                       "train/q2 bellman loss": qf2_bellman_loss,
                       "train/q1 ood loss": qf1_ood_loss,
                       "train/q2 ood loss": qf2_ood_loss,
                       "train/ood ratio": ood_ratio,
                       "train/critic1 gradient": critic1_grad_norm,
                       "train/critic2 gradient": critic2_grad_norm,
                       "train/idd action dist": idd_action_threshold,
                       "train/lam": lam,
                       "train/next_log_pi mean": next_log_pi.mean(),
                       "train/next_log_pi max": next_log_pi.max(),
                       "train/next_log_pi min": next_log_pi.min(),
                       "train/next_std mean": next_std.mean(),
                       "train/next_std min": next_std.mean(-1).min(),
                       "train/total lam error": total_lam_error,
                       }

        return critic_update_data

    def train(self, states, actions, next_states, rewards, dones, step):
        """
        Step1. Update VAE
        """
        self.update_vae(states, actions)
        with torch.no_grad():
            idd_action_threshold = self.vae.calc_dist(states, actions).mean()

        """
        Step2. Update critic
        """
        critic_update_data = self.update_critic(states, actions, next_states, rewards, dones, idd_action_threshold)

        """
        Step3. Update actor
        """
        actor_update_data = self.update_actor(states, actions)

        if step % self.args.log_interval == 0:
            logging_data = {**critic_update_data, **actor_update_data}
            logging_data['total_step'] = step
            wandb.log(logging_data)

        """
        Step4. Soft Updates target network
        """
        self.sync_weight(self.critic1_target, self.critic1, self.args.soft_target_tau)
        self.sync_weight(self.critic2_target, self.critic2, self.args.soft_target_tau)
