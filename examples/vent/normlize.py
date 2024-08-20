import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

current_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
pkl_file = os.path.join(data_dir, 'imputed.pkl')
with open(pkl_file, 'rb') as file:
    df = pd.read_pickle(pkl_file)



# In a standard normal distribution, the distribution of values follows the 68-95-99.7 rule (also known as the three-sigma rule):
# Approximately 68% of the values fall within 1 standard deviation of the mean.
# Approximately 95% of the values fall within 2 standard deviations of the mean.
# Approximately 99.7% of the values fall within 3 standard deviations of the mean.
def replace_extreme_values(data, method='mean', threshold=5):
    """
    Replace extreme values in the data using the specified method.
    :param data: The input data, a numpy array
    :param method: The replacement method, 'mean' indicates replacing with the mean, 'median' indicates replacing with the median
    :param threshold: The threshold for extreme values, typically a multiple of the standard deviation
    :return: The data with extreme values replaced
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    extreme_indices = np.abs(data - mean) > threshold * std

    if method == 'mean':
        replacement = mean
    elif method == 'median':
        replacement = np.median(data, axis=0)
    else:
        raise ValueError("method should be 'mean' or 'median'")

    for i in range(data.shape[1]):
        data[extreme_indices[:, i], i] = replacement[i]

    return data

# treatment_columns
treatment_columns = ['PEEP', 'FIO2', 'ADJUSTED_TIDAL_VOLUME']

# actions
actions = df[treatment_columns].values.astype(np.float32)
# replace extreme values
actions = replace_extreme_values(actions, method='mean', threshold=3)

# Specify the maximum and minimum values used during the normalization process.
specified_min = np.array([0.0, 21.0, 0.0])
specified_max = np.array([20.0, 100.0, 20.0])

# Normalize the action data.
normalized_actions = (actions - specified_min) / (specified_max - specified_min)
normalized_actions[normalized_actions > 1] = 1

# Check the normalization results.
for i, col in enumerate(treatment_columns):
    print(f"Maximum value after normalization for {col}: {np.max(normalized_actions[:, i])}")
    print(f"Minimum value after normalization for {col}: {np.min(normalized_actions[:, i])}")

## Output the maximum and minimum values for each dimension separately.
# max_action = np.max(actions, axis=0)
# min_action = np.min(actions, axis=0)
#
# for i, col in enumerate(treatment_columns):
#     print(f"{col}的最大值: {max_action[i]}")
#     print(f"{col}的最小值: {min_action[i]}")

