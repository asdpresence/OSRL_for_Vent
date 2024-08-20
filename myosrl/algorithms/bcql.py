# reference: https://github.com/sfujim/BCQ
from copy import deepcopy
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa

from osrl.common.net import (VAE, EnsembleDoubleQCritic, LagrangianPIDController,
                             MLPGaussianPerturbationActor)


class BCQL(nn.Module):
    """
        Batch-Constrained deep Q-learning with PID Lagrangian (BCQL)

    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list): List of integers specifying the sizes
            of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes
            of the layers in the critic network.
        vae_hidden_sizes (int): Number of hidden units in the VAE.
        sample_action_num (int): Number of action samples to draw.
        gamma (float): Discount factor for the reward.
        tau (float): Soft update coefficient for the target networks.
        phi (float): Scale parameter for the Gaussian perturbation
            applied to the actor's output.
        lmbda (float): Weight of the Lagrangian term.
        beta (float): Weight of the KL divergence term.
        PID (list): List of three floats containing the coefficients
            of the PID controller.
        num_q (int): Number of Q networks in the ensemble.
        num_qc (int): Number of cost Q networks in the ensemble.
        cost_limit (int): Upper limit on the cost per episode.
        episode_len (int): Maximum length of an episode.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0').
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 # 修改为list
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 vae_hidden_sizes: int = 64,
                 sample_action_num: int = 10,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 phi: float = 0.05,
                 lmbda: float = 0.75,
                 beta: float = 0.5,
                 PID: list = [0.1, 0.003, 0.001],
                 num_q: int = 1,
                 num_qc: int = 1,
                 # cost_limit: int = 10,
                 cost_limit: int = 7,
                 # episode_len: int = 300,
                 # 最多72小时
                 episode_len: int = 18,
                 device: str = "cpu"):

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.latent_dim = self.action_dim * 2
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.vae_hidden_sizes = vae_hidden_sizes
        self.sample_action_num = sample_action_num
        self.gamma = gamma
        self.tau = tau
        self.phi = phi
        self.lmbda = lmbda
        self.beta = beta
        self.KP, self.KI, self.KD = PID
        self.num_q = num_q
        self.num_qc = num_qc
        self.cost_limit = cost_limit
        self.episode_len = episode_len
        self.device = device

        ################ create actor critic model ###############
        self.actor = MLPGaussianPerturbationActor(self.state_dim, self.action_dim,
                                                  self.a_hidden_sizes, nn.Tanh, self.phi,
                                                  self.max_action).to(self.device)
        self.critic = EnsembleDoubleQCritic(self.state_dim,
                                            self.action_dim,
                                            self.c_hidden_sizes,
                                            nn.ReLU,
                                            num_q=self.num_q).to(self.device)
        self.cost_critic = EnsembleDoubleQCritic(self.state_dim,
                                                 self.action_dim,
                                                 self.c_hidden_sizes,
                                                 nn.ReLU,
                                                 num_q=self.num_qc).to(self.device)
        self.vae = VAE(self.state_dim, self.action_dim, self.vae_hidden_sizes,
                       self.latent_dim, self.max_action, self.device).to(self.device)

        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()
        self.critic_old = deepcopy(self.critic)
        self.critic_old.eval()
        self.cost_critic_old = deepcopy(self.cost_critic)
        self.cost_critic_old.eval()

        # 固定
        self.qc_thres = cost_limit * (1 - self.gamma ** self.episode_len) / (
                1 - self.gamma) / self.episode_len
        self.controller = LagrangianPIDController(self.KP, self.KI, self.KD,
                                                  self.qc_thres)

    # 我的添加
    def compute_action_cost(self, actions, selected_actions):
        return nn.functional.mse_loss(actions, selected_actions, reduction='none').mean(dim=1)
    # def compute_action_cost(self, actions, selected_actions):
    #     # 假设 actions 和 selected_actions 的形状为 (batch_size, action_dim1, action_dim2, action_dim3)
    #     mse_loss = nn.functional.mse_loss(actions, selected_actions, reduction='none')
    #     # 在所有动作维度上求平均值，这里假设动作维度从第1维开始
    #     print(mse_loss)
    #     print(actions)
    #     cost = mse_loss.mean(dim=(0,1,2))
    #     return cost

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """
        Softly update the parameters of target module
        towards the parameters of source module.
        """
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def vae_loss(self, observations, actions):
        recon, mean, std = self.vae(observations, actions)
        recon_loss = nn.functional.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        loss_vae = recon_loss + self.beta * KL_loss

        self.vae_optim.zero_grad()
        loss_vae.backward()
        self.vae_optim.step()
        stats_vae = {"loss/loss_vae": loss_vae.item()}
        return loss_vae, stats_vae

    def critic_loss(self, observations, next_observations, actions, rewards, done):
        _, _, q1_list, q2_list = self.critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(next_observations, self.sample_action_num,
                                               0).to(self.device)

            act_targ_next = self.actor_old(obs_next, self.vae.decode(obs_next))
            q1_targ, q2_targ, _, _ = self.critic_old.predict(obs_next, act_targ_next)

            q_targ = self.lmbda * torch.min(
                q1_targ, q2_targ) + (1. - self.lmbda) * torch.max(q1_targ, q2_targ)
            q_targ = q_targ.reshape(batch_size, -1).max(1)[0]

            backup = rewards + self.gamma * (1 - done) * q_targ

        # 计算损失
        loss_critic = self.critic.loss(backup, q1_list) + self.critic.loss(
            backup, q2_list)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
        stats_critic = {"loss/critic_loss": loss_critic.item()}
        return loss_critic, stats_critic

    def cost_critic_loss(self, observations, next_observations, actions, costs, done):
        _, _, q1_list, q2_list = self.cost_critic.predict(observations, actions)
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(next_observations, self.sample_action_num,
                                               0).to(self.device)

            act_targ_next = self.actor_old(obs_next, self.vae.decode(obs_next))
            q1_targ, q2_targ, _, _ = self.cost_critic_old.predict(
                obs_next, act_targ_next)

            q_targ = self.lmbda * torch.min(
                q1_targ, q2_targ) + (1. - self.lmbda) * torch.max(q1_targ, q2_targ)
            q_targ = q_targ.reshape(batch_size, -1).max(1)[0]

            backup = costs + self.gamma * q_targ
        loss_cost_critic = self.cost_critic.loss(
            backup, q1_list) + self.cost_critic.loss(backup, q2_list)
        self.cost_critic_optim.zero_grad()
        loss_cost_critic.backward()
        self.cost_critic_optim.step()
        stats_cost_critic = {"loss/cost_critic_loss": loss_cost_critic.item()}
        return loss_cost_critic, stats_cost_critic

    def find_max_value(self,tensor):
        max_value, _ = torch.max(tensor, dim=0)
        return max_value.item()

    def actor_loss(self, observations):
        # 禁止梯度更新.因为在计算 Actor 网络的损失时，这些网络的参数不需要更新。
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.cost_critic.parameters():
            p.requires_grad = False
        for p in self.vae.parameters():
            p.requires_grad = False

        # 使用 Actor 网络和 VAE 网络生成对应于给定观测的动作。具体来说，Actor 网络接收观测和由 VAE 网络解码后的潜在变量作为输入，并输出动作。
        actions = self.actor(observations, self.vae.decode(observations))
        # 通过 Critic 网络和 Cost Critic 网络分别计算生成动作的 Q 值和约束成本。为了确保稳健性，取两个 Q 值和约束成本中的最小值。
        q1_pi, q2_pi, _, _ = self.critic.predict(observations, actions)  # [batch_size]
        qc1_pi, qc2_pi, _, _ = self.cost_critic.predict(observations, actions)
        qc_pi = torch.min(qc1_pi, qc2_pi)

        # 测试
        # print(f"QC_PI: {qc_pi}")

        q_pi = torch.min(q1_pi, q2_pi)

        # 使用 PID 控制器根据约束成本的值计算一个惩罚项（qc_penalty），然后将其加到 Q 值的负数上，得到最终的 Actor 网络损失。
        with torch.no_grad():
            multiplier = self.controller.control(qc_pi).detach()

            # 测试
            # print(f"Multiplier: {multiplier}")
            # max_value = self.find_max_value(qc_pi)
            # print("最大值为:", max_value)

        qc_penalty = ((qc_pi - self.qc_thres) * multiplier).mean()
        loss_actor = -q_pi.mean() + qc_penalty

        # 通过反向传播计算梯度，并使用优化器（self.actor_optim）更新 Actor 网络的参数。
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        # 返回 Actor 网络的损失值和一些统计信息，包括 Actor 网络的损失、约束成本的惩罚项和 PID 控制器的乘数。
        stats_actor = {
            "loss/actor_loss": loss_actor.item(),
            "loss/qc_penalty": qc_penalty.item(),
            "loss/lagrangian": multiplier.item()
        }

        # 恢复梯度更新
        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.cost_critic.parameters():
            p.requires_grad = True
        for p in self.vae.parameters():
            p.requires_grad = True
        return loss_actor, stats_actor

    def setup_optimizers(self, actor_lr, critic_lr, vae_lr):
        """
        Sets up optimizers for the actor, critic, cost critic, and VAE models.
        """
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(),
                                                  lr=critic_lr)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

    def sync_weight(self):
        """
        Soft-update the weight for the target network.
        """
        self._soft_update(self.critic_old, self.critic, self.tau)
        self._soft_update(self.cost_critic_old, self.cost_critic, self.tau)
        self._soft_update(self.actor_old, self.actor, self.tau)

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Given a single obs, return the action, value, logp.
        '''
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        act = self.actor(obs, self.vae.decode(obs))
        act = act.data.numpy() if self.device == "cpu" else act.data.cpu().numpy()
        return np.squeeze(act, axis=0), None


import numpy as np

class BCQLTrainer:
    """
    Constraints Penalized Q-learning Trainer

    Args:
        model (BCQL): The BCQL model to be trained.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        critic_lr (float): learning rate for critic
        vae_lr (float): learning rate for vae
        reward_scale (float): The scaling factor for the reward signal.
        cost_scale (float): The scaling factor for the constraint cost.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(
            self,
            model: BCQL,
            logger: WandbLogger = DummyLogger(),
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            vae_lr: float = 1e-4,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            device="cpu"):

        self.model = model
        self.logger = logger
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.model.setup_optimizers(actor_lr, critic_lr, vae_lr)

    def train_one_step(self, observations, next_observations, actions, rewards, done):
        """
        Trains the model by updating the VAE, critic, cost critic, and actor.
        """

        # update VAE
        loss_vae, stats_vae = self.model.vae_loss(observations, actions)

        with torch.no_grad():
            # Generate actions using actor
            selected_actions = self.model.actor(observations, self.model.vae.decode(observations))
        # Calculate cost as the difference between dataset actions and selected actions
        costs = self.model.compute_action_cost(actions, selected_actions)

        # update critic
        loss_critic, stats_critic = self.model.critic_loss(observations,
                                                           next_observations, actions,
                                                           rewards, done)
        # update cost critic
        loss_cost_critic, stats_cost_critic = self.model.cost_critic_loss(
            observations, next_observations, actions, costs, done)
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations)

        self.model.sync_weight()

        self.logger.store(**stats_vae)
        self.logger.store(**stats_critic)
        self.logger.store(**stats_cost_critic)
        self.logger.store(**stats_actor)

    @torch.no_grad()
    def evaluate(self, eval_episodes: int = 1000, eval_dataset: list = None):
        """
        Evaluates the performance of the model on a number of episodes from the offline dataset.
        Args:
            eval_dataset (list): The offline dataset containing multiple episodes.
            eval_episodes (int): Number of episodes to evaluate.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []

        random.seed(42)  # 选择一个固定的种子
        # 从eval_dataset中随机选择eval_episodes条轨迹
        sampled_episodes = random.sample(eval_dataset, eval_episodes)

        for episode in sampled_episodes:
            observations, actions, rewards, next_observations, done = (
                episode['observations'], episode['actions'], episode['rewards'],
                episode['next_observations'], episode['done']
            )
            # 将numpy数组转换为PyTorch张量
            observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_observations = torch.tensor(next_observations, dtype=torch.float32).to(self.device)
            done = torch.tensor(done, dtype=torch.float32).to(self.device)

            epi_ret, epi_len, epi_cost = self.offline_rollout(observations, actions, rewards, next_observations, done)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)

        # 将 CUDA 张量转移到 CPU 并转换为 NumPy 数组
        episode_rets_array = np.array([ret.cpu().detach().numpy() for ret in episode_rets])
        episode_costs_array = np.array([cost.cpu().detach().numpy() for cost in episode_costs])
        # 对于整型值，不需要进行cpu()和detach()转换，直接使用即可
        episode_lens_array = np.array(episode_lens)

        self.model.train()
        return np.mean(episode_rets_array) / self.reward_scale, np.mean(episode_costs_array) / self.cost_scale, np.mean(episode_lens_array)

    @torch.no_grad()
    def offline_rollout(self, observations, actions, rewards, next_observations, done):
        """
        Evaluates the performance of the model on a single episode from the offline dataset.
        """
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for i in range(len(observations)):
            obs = observations[i]
            act = actions[i]
            # reward = rewards[i]
            # obs_next = next_observations[i]
            done_flag = done[i]

            # 测试
            # print(obs,act)
            # print(obs.shape)

            #变成二维张量
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)

            # 防止梯度更新
            with torch.no_grad():
                # 计算cost
                selected_action = self.model.actor(obs, self.model.vae.decode(obs))
                cost = self.model.compute_action_cost(act, selected_action)

                # 计算reward
                q1_value, q2_value, _, _ = self.model.critic.predict(obs, selected_action)

            episode_ret += torch.min(q1_value, q2_value)
            episode_len += 1
            episode_cost += cost

            if done_flag:
                break

        return episode_ret, episode_len, episode_cost
