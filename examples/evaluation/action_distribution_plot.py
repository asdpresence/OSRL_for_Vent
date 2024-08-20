import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from myosrl.algorithms import BCQL, BCQLTrainer
from osrl.common.exp_util import load_config_and_model, seed_all
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import seaborn as sns


# Use offline data for evaluation

# Collect offline data
current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
pkl_file = os.path.join(data_dir, 'processed_mimic3_data.pkl')
with open(pkl_file, 'rb') as file:
    df = pd.read_pickle(pkl_file)

original_actions = df["actions"]

def denormalize_actions(normalized_actions):
    # Original range
    original_ranges = np.array([[0, 20], [21, 100], [0, 20]])
    # Calculate the span of the original range
    ranges = original_ranges[:, 1] - original_ranges[:, 0]
    # Perform denormalization
    denormalized_actions = normalized_actions * ranges + original_ranges[:, 0]
    return denormalized_actions

# denormalization
original_actions = denormalize_actions(original_actions)

import pickle
# CQL actions
input_file = 'cql_predicted_actions_50000.pkl'
with open(input_file, 'rb') as file:
    loaded_data = pickle.load(file)

cql_predicted_actions = loaded_data

# BCQL actions
input_file = 'bcql_predicted_actions_50000.pkl'
with open(input_file, 'rb') as file:
    loaded_data = pickle.load(file)

bcql_predicted_actions = loaded_data

# CPQ actions
input_file = 'cpq_predicted_actions_50000.pkl'
with open(input_file, 'rb') as file:
    loaded_data = pickle.load(file)

cpq_predicted_actions = loaded_data

bins_1 = [0, 5, 7, 9, 11, 13, 15, 20]
bins_2 = [25, 30, 35, 40, 45, 50, 55, 100]
bins_3 = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15, 20]

# Visualize the action distribution
def plot_combined_action_distribution(baseline_actions,original_actions, predicted_actions, predicted_actions_2, bins_list, titles, x_labels, last_labels):
    width = 0.2

    fig, axes = plt.subplots(len(bins_list), 1, figsize=(12, 12), sharex=False)

    for i, (bins, title, x_label, last_label) in enumerate(zip(bins_list, titles, x_labels, last_labels)):

        baseline_hist, _ = np.histogram(baseline_actions[:, i], bins=bins)
        original_hist, _ = np.histogram(original_actions[:, i], bins=bins)
        predicted_hist, _ = np.histogram(predicted_actions[:, i], bins=bins)
        predicted_hist_2, _ = np.histogram(predicted_actions_2[:, i], bins=bins)

        bin_centers = np.arange(len(bins) - 1)

        # Plot the histogram
        axes[i].bar(bin_centers - 1.5 * width, baseline_hist, width=width, label='CQL', color='#F7D58B',
                    align='center')
        axes[i].bar(bin_centers - 0.5 * width, original_hist, width=width, label='Physician', color='#9BC985',
                    align='center')
        axes[i].bar(bin_centers + 0.5 * width, predicted_hist, width=width, label='BCQ-L', color='#B595BF',
                    align='center')
        axes[i].bar(bin_centers + 1.5 * width, predicted_hist_2, width=width, label='CPQ', color='#797BB7',
                    align='center')

        # Set the x-axis interval labels
        bin_labels = [f'{bins[j]}-{bins[j + 1]}' for j in range(len(bins) - 2)] + [last_label]
        axes[i].set_xticks(bin_centers)
        axes[i].set_xticklabels(bin_labels, rotation=45, ha='right')

        # Set the labels and title
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel('Action Counts')
        axes[i].set_title(title)
        axes[i].legend(loc='upper right')

    plt.tight_layout(pad=2.0)
    plt.show()

# Plot a combination chart
plot_combined_action_distribution(
    cql_predicted_actions,
    original_actions,
    bcql_predicted_actions,
    cpq_predicted_actions,
    [bins_1, bins_2, bins_3],
    ['PEEP', 'FiO2', 'Adjusted Tidal Volume'],
    ['cmH2O', 'Percentage (%)', 'ml/Kg'],
    ['15+', '55+', '15+']
)