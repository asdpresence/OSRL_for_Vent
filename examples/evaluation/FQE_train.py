import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import sys
import os
import pandas as pd
from myosrl.common import Mimic3_Vent_Dataset,IterableDataset

import pandas as pd
import numpy as np

# Linear dataset class, only repeats once
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
            return observations, actions, rewards,  done, q_value,is_init

        return observations, actions, rewards, done,q_value

    def __iter__(self):
        # Sequentially iterate through all samples in the dataset
        for idx in range(self.dataset_size):
            yield self.__prepare_sample(idx)


# Random dataset class, repeatedly selects data randomly
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

    # Define the iteration method for the dataset. It will loop indefinitely, randomly selecting a sample index 'idx' (based on the 'sample_prob' probability distribution), and yield the prepared sample.
    def __iter__(self):
        for _ in range(self.dataset_size):  # 这里使用 dataset_size 作为终止条件
            idx = np.random.choice(self.dataset_size, p=self.sample_prob)
            yield self.__prepare_sample(idx)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state,action):
        x = torch.cat((state, action), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_actions(bc_model, states):
    with torch.no_grad():
        actions = bc_model.actor(states)
        # self.model.actor(obs, self.model.vae.decode(obs))
    return actions

# Train the Q network

Device = "cuda:1"

def train_q_network(q_network, dataset, optimizer, criterion, num_epochs=500,save_path="q_network.pth"):
    dataloader = DataLoader(dataset, batch_size=64)

    q_network.train()
    for epoch in range(num_epochs):
        for states, actions, rewards,dones ,q_value in dataloader:

            states = states.to(Device)
            actions = actions.to(Device)
            q_value = q_value.to(Device)

            # Calculate the estimated Q-values
            current_q_values = q_network(states, actions)
            # next_q_values = q_network(next_states, next_actions)

            # Compute the loss function
            target_q_values = q_value
            target_q_values = target_q_values.unsqueeze(1)
            loss = criterion(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Save the Q network parameters
    torch.save(q_network.state_dict(), save_path)
    print(f"Q Network saved to {save_path}")

def evaluate_strategy(bcql_model, q_network, dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # dataloader_iter = iter(dataloader)


    total_q_value = 0.0
    total_count = 0

    with torch.no_grad():
        for states, _, _, _, _ in dataloader:
            actions = generate_actions(bcql_model, states)
            q_values = q_network(states,actions)
            # q_values = q_values.gather(1, actions.unsqueeze(-1).long()).squeeze(-1)
            # total_q_value += q_values.sum().item()
            total_q_value += q_values.sum().item()
            total_count += len(states)

    average_q_value = total_q_value / total_count
    print(f"Average Q Value: {average_q_value}")


current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
# pkl_file = os.path.join(data_dir, 'processed_mimic3_data_pos_reward.pkl')
pkl_file = os.path.join(data_dir, 'processed_mimic3_data.pkl')
with open(pkl_file, 'rb') as file:
    df = pd.read_pickle(pkl_file)

offline_data = df

def calculate_q_values(rewards, dones, gamma=0.99):
    q_values = np.zeros_like(rewards)
    if dones[-1] == 1:
        q_values[-1] = rewards[-1]
    for i in range(len(rewards)-2, -1, -1):
        if dones[i] == 1:
            q_values[i] = rewards[i]
        else:
            q_values[i] = rewards[i] + gamma * q_values[i+1]

    print(q_values)
    return q_values

# Calculate the Q-values and add them to the DataFrame
df['q_value'] = calculate_q_values(df['rewards'], df['done'])

print(df)

dataset = SequentialDataset(df)

Dataset = OfflineDataset(df)


q_network = QNetwork(state_dim=38, action_dim=3, hidden_dim=256).to(Device)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()


train_q_network(q_network, Dataset, optimizer, criterion, num_epochs=500)

