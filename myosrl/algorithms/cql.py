from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa

import random

from osrl.common.net import VAE, EnsembleQCritic, SquashedGaussianMLPActor

class CQL(nn.Module):
    """
    Conservative Q-Learning (CQL)

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list): List of integers specifying the sizes of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes of the layers in the critic network.
        gamma (float): Discount factor for the reward.
        tau (float): Soft update coefficient for the target networks.
        alpha (float): Temperature parameter for SAC.
        num_q (int): Number of Q networks in the ensemble.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0').
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 1.0,
                 num_q: int = 2,
                 device: str = "cpu"):

        super().__init__()
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.num_q = num_q
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        ################ create actor critic model ###############
        # Actor network and target network
        self.actor = SquashedGaussianMLPActor(self.state_dim, self.action_dim,
                                              self.a_hidden_sizes, nn.ReLU).to(self.device)
        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()

        # Critic networks and target networks
        self.critic = EnsembleQCritic(self.state_dim, self.action_dim,
                                      self.c_hidden_sizes, nn.ReLU, num_q=self.num_q).to(self.device)
        self.critic_old = deepcopy(self.critic)
        self.critic_old.eval()

        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        print("CQL Initialized")

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """
        Softly update the parameters of target module towards the parameters of source module.
        """
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def _actor_forward(self,
                       obs: torch.tensor,
                       deterministic: bool = False,
                       with_logprob: bool = True):
        """
        Return action distribution and action log prob [optional].
        """
        a, logp = self.actor(obs, deterministic, with_logprob)
        return a * self.max_action, logp

    def critic_loss(self, observations, next_observations, actions, rewards, done):
        """
        Compute the loss for the critic networks.
        """
        with torch.no_grad():
            next_actions, logp_next = self._actor_forward(next_observations, False, True)
            q_target = rewards + self.gamma * (1 - done) * (
                torch.min(*self.critic_old(next_observations, next_actions)) - self.alpha * logp_next
            )

        q1, q2 = self.critic(observations, actions)

        # Conservative Q-Learning Loss
        random_actions = torch.FloatTensor(observations.shape[0] * 10, self.action_dim).uniform_(
            -self.max_action, self.max_action).to(self.device)
        random_observations = observations.unsqueeze(1).repeat(1, 10, 1).view(
            observations.shape[0] * 10, observations.shape[1])

        q1_rand, q2_rand = self.critic(random_observations, random_actions)

        cql_loss = torch.logsumexp(q1_rand, dim=0).mean() + torch.logsumexp(q2_rand, dim=0).mean() - \
                   q1.mean() - q2.mean()

        loss_q = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)
        loss_critic = loss_q + cql_loss

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        return loss_critic.item()

    def actor_loss(self, observations):
        """
        Compute the loss for the actor network.
        """
        actions, logp = self._actor_forward(observations, False, True)
        q_pi = torch.min(*self.critic(observations, actions))

        loss_actor = (self.alpha * logp - q_pi).mean()

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        return loss_actor.item()

    def sync_weight(self):
        """
        Soft-update the weight for the target network.
        """
        self._soft_update(self.critic_old, self.critic, self.tau)
        self._soft_update(self.actor_old, self.actor, self.tau)

    def setup_optimizers(self, actor_lr, critic_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def act(self,
            obs: np.ndarray,
            deterministic: bool = False,
            with_logprob: bool = False):
        """
        Given a single observation, return the action and log probability.
        """
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        a, logp_a = self._actor_forward(obs, deterministic, with_logprob)
        a = a.data.numpy() if self.device == "cpu" else a.data.cpu().numpy()
        logp_a = logp_a.data.numpy() if self.device == "cpu" else logp_a.data.cpu().numpy()
        return np.squeeze(a, axis=0), np.squeeze(logp_a)

    def act_rollout(self,
                    obs: np.ndarray,
                    deterministic: bool = False,
                    with_logprob: bool = False):
        """
        Given a single observation, return the action and log probability.
        """
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        a, logp_a = self._actor_forward(obs, deterministic, with_logprob)
        a = a.data.numpy() if self.device == "cpu" else a.data.cpu().numpy()
        logp_a = logp_a.data.numpy() if self.device == "cpu" else logp_a.data.cpu().numpy()
        return a, logp_a


class CQLTrainer:
    """
    Conservative Q-Learning Trainer

    Args:
        model (CQL): The CQL model to be trained.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): Learning rate for the actor.
        critic_lr (float): Learning rate for the critic.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(
            self,
            model: CQL,
            logger: WandbLogger = DummyLogger(),
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            device="cpu") -> None:

        self.model = model
        self.logger = logger
        self.device = device
        self.model.setup_optimizers(actor_lr, critic_lr)

    def train_one_step(self, observations, next_observations, actions, rewards, done):
        """
        Perform one step of training for both the critic and actor.
        """
        loss_critic = self.model.critic_loss(observations, next_observations, actions, rewards, done)
        loss_actor = self.model.actor_loss(observations)
        self.model.sync_weight()

        self.logger.store(loss_critic=loss_critic, loss_actor=loss_actor)

    @torch.no_grad()
    def evaluate(self, eval_episodes: int = 1000, eval_dataset: list = None):
        """
        Evaluates the performance of the model on a number of episodes from the offline dataset.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []

        random.seed(42)
        sampled_episodes = random.sample(eval_dataset, eval_episodes)

        for episode in sampled_episodes:
            observations, actions, rewards, next_observations, done = (
                episode['observations'], episode['actions'], episode['rewards'],
                episode['next_observations'], episode['done']
            )

            observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_observations = torch.tensor(next_observations, dtype=torch.float32).to(self.device)
            done = torch.tensor(done, dtype=torch.float32).to(self.device)

            epi_ret, epi_len, epi_cost = self.offline_rollout(observations, actions, rewards, next_observations, done)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)

        episode_rets_array = np.array([ret.cpu().detach().numpy() for ret in episode_rets])
        episode_costs_array = np.array([cost.cpu().detach().numpy() for cost in episode_costs])
        episode_lens_array = np.array(episode_lens)

        self.model.train()
        return np.mean(episode_rets_array), np.mean(episode_costs_array), np.mean(episode_lens_array)

    @torch.no_grad()
    def offline_rollout(self, observations, actions, rewards, next_observations, done):
        """
        Evaluates the performance of the model on a single episode from the offline dataset.
        """
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for i in range(len(observations)):
            obs = observations[i]
            act = actions[i]
            done_flag = done[i]

            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)

            with torch.no_grad():
                selected_action, _ = self.model.actor(obs, True, True)
                cost = self.model.compute_action_cost(act, selected_action)
                q_value, _ = self.model.critic.predict(obs, selected_action)

            episode_ret += q_value
            episode_len += 1
            episode_cost += cost

            if done_flag:
                break

        return episode_ret, episode_len, episode_cost
