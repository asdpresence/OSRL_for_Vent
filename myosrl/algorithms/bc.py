import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa
import random

from osrl.common.net import MLPActor

class BC(nn.Module):
    """
    Behavior Cloning (BC)

    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list, optional): List of integers specifying the sizes
            of the layers in the actor network.
        episode_len (int, optional): Maximum length of an episode.
        device (str, optional): Device to run the model on (e.g. 'cpu' or 'cuda:0').
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 episode_len: int = 18,
                 device: str = "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.a_hidden_sizes = a_hidden_sizes
        self.episode_len = episode_len
        self.device = device

        self.actor = MLPActor(self.state_dim, self.action_dim, self.a_hidden_sizes,
                              nn.ReLU, self.max_action).to(self.device)

    def actor_loss(self, observations, actions):
        pred_actions = self.actor(observations)
        loss_actor = F.mse_loss(pred_actions, actions)
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        stats_actor = {"loss/actor_loss": loss_actor.item()}
        return loss_actor, stats_actor

    def setup_optimizers(self, actor_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def act(self, obs):
        '''
        Given a single obs, return the action.
        '''
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        act = self.actor(obs)
        act = act.data.numpy() if self.device == "cpu" else act.data.cpu().numpy()
        return np.squeeze(act, axis=0)


class BCTrainer:
    """
    Behavior Cloning Trainer

    Args:
        model (BC): The BC model to be trained.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        bc_mode (str): specify bc mode
        cost_limit (int): Upper limit on the cost per episode.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(
            self,
            model: BC,
            # env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            actor_lr: float = 1e-4,
            bc_mode: str = "all",
            cost_limit: int = 7,
            device="cpu"):

        self.model = model
        self.logger = logger
        # self.env = env
        self.device = device
        self.bc_mode = bc_mode
        self.cost_limit = cost_limit
        self.model.setup_optimizers(actor_lr)

    # 计算动作距离
    def compute_action_cost(self, actions, selected_actions):
        return nn.functional.mse_loss(actions, selected_actions, reduction='none').mean(dim=1)

    def set_target_cost(self, target_cost):
        self.cost_limit = target_cost

    def train_one_step(self, observations, actions):
        """
        Trains the model by updating the actor.
        """
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations, actions)
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
        episode_costs, episode_lens = [], []

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

            epi_len, epi_cost = self.offline_rollout(observations, actions, rewards, next_observations, done)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)

        # 将 CUDA 张量转移到 CPU 并转换为 NumPy 数组
        episode_costs_array = np.array([cost.cpu().detach().numpy() for cost in episode_costs])
        # 对于整型值，不需要进行cpu()和detach()转换，直接使用即可
        episode_lens_array = np.array(episode_lens)

        self.model.train()
        return  np.mean(episode_costs_array), np.mean(episode_lens_array)

    @torch.no_grad()
    def offline_rollout(self, observations, actions, rewards, next_observations, done):
        """
        Evaluates the performance of the model on a single episode from the offline dataset.
        """
        episode_cost, episode_len =  0.0, 0
        for i in range(len(observations)):
            obs = observations[i]
            act = actions[i]
            # reward = rewards[i]
            # obs_next = next_observations[i]
            done_flag = done[i]

            # 测试
            # print(obs,act)
            # print(obs.shape)

            # 变成二维张量
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)

            # 防止梯度更新
            with torch.no_grad():
                # 计算cost
                selected_action = self.model.actor(obs)
                cost = self.compute_action_cost(act, selected_action)

            episode_len += 1
            episode_cost += cost

            if done_flag:
                break

        return  episode_len, episode_cost
