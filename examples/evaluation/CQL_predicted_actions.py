import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from myosrl.algorithms import CQL, CQLTrainer
from osrl.common.exp_util import load_config_and_model, seed_all
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import seaborn as sns

from myosrl.common import IterableDataset
# offline dataset
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


        # self.dataset["done"] = np.logical_or(self.dataset["terminals"],
        #                                       self.dataset["timeouts"]).astype(np.float32)



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

        if done == 1:
            next_observations = observations

        if self.state_init:
            is_init = self.dataset["is_init"][idx]
            return observations, next_observations, actions, rewards,  done, is_init

        return observations, next_observations, actions, rewards, done

    def __iter__(self):
        # Sequentially iterate through all samples in the dataset
        for idx in range(self.dataset_size):
            yield self.__prepare_sample(idx)


# generate actions
def generate_actions(cql_model, states):
    with torch.no_grad():
        actions = cql_model.actor(states, cql_model.vae.decode(states))
    return actions

# load evaluation configs
class EvalConfig:
    path: str = "/home/wyf/projects/OSRL_for_Vent/examples/vent/logs/CQL/cql/cql"
    noise_scale: List[float] = None
    eval_episodes: int = 1000
    best: bool = False
    device: str = "cpu"
    threads: int = 4

eval_config = EvalConfig()

# load CQL model and configs
cfg, model = load_config_and_model(eval_config.path, eval_config.best)
seed_all(cfg["seed"])

if eval_config.device == "cpu":
    torch.set_num_threads(eval_config.threads)

# Initialize CQL
cql_model = CQL(
        state_dim=38,
        action_dim=3,
        max_action=1.0,
        a_hidden_sizes=cfg["a_hidden_sizes"],
        c_hidden_sizes=cfg["c_hidden_sizes"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        alpha=cfg["alpha"],
        num_q=cfg["num_q"],
        device=eval_config.device
    )
cql_model.load_state_dict(model["model_state"])
cql_model.to(eval_config.device)


# collect offline data
current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
pkl_file = os.path.join(data_dir, 'processed_mimic3_data.pkl')
with open(pkl_file, 'rb') as file:
    df = pd.read_pickle(pkl_file)

data=df
# print(df["states"].shape[0])


# Create an offline dataset
dataset = OfflineDataset(data)
trainloader = DataLoader(dataset, batch_size=1, pin_memory=True)
trainloader_iter = iter(trainloader)

all_predicted_actions = []
from tqdm.auto import trange
for step in trange(95002, desc="Predicting"):
    batch = next(trainloader_iter)

    observations, next_observations, actions, rewards, done = [
        b.to("cpu") for b in batch
    ]
    predicted_actions = generate_actions(cql_model, observations)
    all_predicted_actions.append(predicted_actions)

all_predicted_actions = torch.cat(all_predicted_actions, dim=0).cpu().numpy()


def denormalize_actions(normalized_actions):
    # Original range
    original_ranges = np.array([[0, 20], [21, 100], [0, 20]])
    # Calculate the span of the original range
    ranges = original_ranges[:, 1] - original_ranges[:, 0]
    # Perform denormalization
    denormalized_actions = normalized_actions * ranges + original_ranges[:, 0]
    return denormalized_actions

# denormalize
denormalized_actions = denormalize_actions(all_predicted_actions)
# # Print some predicted action values to check their distribution
# print("Sample of predicted actions:")
# print(denormalized_actions[:10])


import pickle

# save denormalized_actions to pkl
output_file = 'cql_predicted_actions_50000.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(denormalized_actions, file)

print(f"Data saved to {output_file}")


