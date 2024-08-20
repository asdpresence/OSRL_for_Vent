import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import sys
import os
import pandas as pd
from myosrl.common import Mimic3_Vent_Dataset,IterableDataset

from myosrl.algorithms import CPQ
from osrl.common.exp_util import load_config_and_model, seed_all
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class EvalConfig:
    # path: str = "log/.../checkpoint/model.pt"
    path: str = "/home/wyf/projects/OSRL_for_Vent/examples/vent/logs/CPQ/cpq/cpq"
    noise_scale: List[float] = None
    eval_episodes: int = 1000
    best: bool = False
    device: str = "cuda"
    threads: int = 4


class SequentialDataset(IterableDataset):
    def __init__(self,
                 dataset: dict,
                 reward_scale: float = 1.0,
                 cost_scale: float = 1.0,
                 state_init: bool = False):
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.sample_prob = None
        self.state_init = state_init
        self.dataset_size = 95002

        if self.state_init:
            self.dataset["is_init"] = self.dataset["done"].copy()
            self.dataset["is_init"][1:] = self.dataset["is_init"][:-1]
            self.dataset["is_init"][0] = 1.0


    def __len__(self):
        return self.dataset_size

    def __prepare_sample(self, idx):
        observations = self.dataset["states"][idx, :]
        # next_observations = self.dataset["states"][idx+1, :]
        actions = self.dataset["actions"][idx, :]
        rewards = self.dataset["rewards"][idx] * self.reward_scale
        done = self.dataset["done"][idx]
        # q_value = self.dataset["q_value"][idx]


        if self.state_init:
            is_init = self.dataset["is_init"][idx]
            return observations, actions, rewards,  done,is_init

        return observations, actions, rewards, done

    def __iter__(self):
        for idx in range(self.dataset_size):
            yield self.__prepare_sample(idx)


class OfflineDataset(IterableDataset):
    def __init__(self,
                 dataset: dict,
                 reward_scale: float = 1.0,
                 cost_scale: float = 1.0,
                 state_init: bool = False):
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.sample_prob = None
        self.state_init = state_init
        # self.dataset_size = self.dataset["states"].shape[0]
        self.dataset_size = 95002

        if self.state_init:
            self.dataset["is_init"] = self.dataset["done"].copy()
            self.dataset["is_init"][1:] = self.dataset["is_init"][:-1]
            self.dataset["is_init"][0] = 1.0


    def __len__(self):
        return self.dataset_size

    def __prepare_sample(self, idx):
        observations = self.dataset["states"][idx, :]
        next_observations = self.dataset["states"][idx+1, :]
        actions = self.dataset["actions"][idx, :]
        rewards = self.dataset["rewards"][idx] * self.reward_scale
        done = self.dataset["done"][idx]
        q_value = self.dataset["q_value"][idx]

        if done == 1:
            next_observations = observations

        if self.state_init:
            is_init = self.dataset["is_init"][idx]
            return observations, actions, rewards,  done,q_value, is_init

        return observations, actions, rewards, done, q_value

    def __iter__(self):
        for _ in range(self.dataset_size):
            idx = np.random.choice(self.dataset_size, p=self.sample_prob)
            yield self.__prepare_sample(idx)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_actions(cpq_model, states):
    with torch.no_grad():
        actions = cpq_model.actor(states)
        # actions = bcql_model.actor(states, cpq_model.vae.decode(states))
        return actions


def evaluate_strategy(cpq_model, q_network, dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # dataloader_iter = iter(dataloader)


    total_q_value = 0.0
    total_count = 0

    with torch.no_grad():
        for states, _, _, _ in dataloader:
            states = states.to(eval_config.device)
            actions = generate_actions(cpq_model, states)
            actions = actions.to(eval_config.device)
            q_values = q_network(states,actions)
            # q_values = q_values.gather(1, actions.unsqueeze(-1).long()).squeeze(-1)
            # total_q_value += q_values.sum().item()
            total_q_value += q_values.sum().item()
            total_count += len(states)

    average_q_value = total_q_value / total_count
    print(f"Average Q Value: {average_q_value}")


eval_config = EvalConfig()


cfg, model = load_config_and_model(eval_config.path, eval_config.best)
seed_all(cfg["seed"])

if eval_config.device == "cpu":
    torch.set_num_threads(eval_config.threads)

cpq_model = CPQ(
        state_dim=38,
        action_dim=3,
        max_action=1.0,
        a_hidden_sizes=cfg["a_hidden_sizes"],
        c_hidden_sizes=cfg["c_hidden_sizes"],
        vae_hidden_sizes=cfg["vae_hidden_sizes"],
        sample_action_num=cfg["sample_action_num"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        beta=cfg["beta"],
        num_q=cfg["num_q"],
        num_qc=cfg["num_qc"],
        qc_scalar=cfg["qc_scalar"],
        cost_limit=cfg["cost_limit"],
        episode_len=cfg["episode_len"],
        device=eval_config.device,
    )
cpq_model.load_state_dict(model["model_state"])
cpq_model.to(eval_config.device)


q_network = QNetwork(state_dim=38, action_dim=3, hidden_dim=256).to(eval_config.device)


q_network.load_state_dict(torch.load("q_network.pth"))
q_network.eval()


current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))

pkl_file = os.path.join(data_dir, 'processed_mimic3_data.pkl')
with open(pkl_file, 'rb') as file:
    df = pd.read_pickle(pkl_file)

offline_data = df

dataset = SequentialDataset(df)

evaluate_strategy(cpq_model, q_network, dataset)
