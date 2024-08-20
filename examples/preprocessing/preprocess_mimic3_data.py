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

# In a standard normal distribution, the values follow the 68-95-99.7 rule (i.e., the three-sigma rule):
# - About 68% of the values fall within 1 standard deviation of the mean.
# - About 95% of the values fall within 2 standard deviations of the mean.
# - About 99.7% of the values fall within 3 standard deviations of the mean.
def replace_extreme_values(data, method='mean', threshold=3):
    """
    Replace extreme values in the data using the specified method.
    :param data: Input data as a numpy array
    :param method: Replacement method, 'mean' to replace with mean, 'median' to replace with median
    :param threshold: Threshold for extreme values, typically a multiple of the standard deviation
    :return: Data with extreme values replaced
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

# The 'ICUSTAY_ID' column contains unique patient identifiers
df = df.sort_values(by=['ICUSTAY_ID', 'START_TIME'])

# Change gender to numerical data
# Convert 'F' (female) to 1, and 'M' (male) to 0
print("Transforming gender into numerical data...")
df['GENDER'] = df['GENDER'].apply(lambda x: 1 if x == 'F' else 0)

# Vital sign columns
# Removing six columns: 'BILIRUBIN', 'ALBUMIN', 'PTT', 'PT', 'INR', 'LACTATE', leaving 38 state variables
vitals_columns = ['FIRST_ADMIT_AGE', 'GENDER', 'WEIGHT',
                  'ICU_READM', 'ELIXHAUSER_SCORE', 'SOFA', 'SIRS', 'GCS', 'HEART_RATE', 'BP_SYS',
                  'BP_DIA', 'BP_MEAN', 'SHOCK_INDEX', 'RESP_RATE', 'TEMPERATURE', 'SPO2', 'POTASSIUM', 'SODIUM',
                  'CHLORIDE', 'GLUCOSE', 'BUN', 'CREATININE', 'MAGNESIUM', 'CALCIUM', 'CO2',
                  'HEMOGLOBIN', 'WBC', 'PLATELET',
                  'PH', 'PAO2', 'PACO2', 'BASE_EXCESS', 'BICARBONATE', 'PAO2FIO2RATIO',
                  'URINEOUTPUT', 'VASO_TOTAL', 'IV_TOTAL', 'CUM_FLUID_BALANCE']
# Treatment columns
treatment_columns = ['PEEP', 'FIO2', 'ADJUSTED_TIDAL_VOLUME']
# 90-day survival outcome column
rewards = df['HOSPMORT90DAY'].apply(lambda x: 1 if x == 0 else -1)

# Extract state variables (vital signs)
states = df[vitals_columns].values.astype(np.float32)
# Replace extreme values
states = replace_extreme_values(states, method='mean', threshold=3)
# Normalize state data
scaler = StandardScaler()
states = scaler.fit_transform(states)

# Extract action variables (three-dimensional treatment measures)
actions = df[treatment_columns].values.astype(np.float32)
# Replace extreme values
actions = replace_extreme_values(actions, method='mean', threshold=3)
# Specify the min and max values for normalization
specified_min = np.array([0.0, 21.0, 0.0])  # Replace with actual values
specified_max = np.array([20.0, 100.0, 20.0])  # Replace with actual values
# Normalize action data
normalized_actions = (actions - specified_min) / (specified_max - specified_min)
normalized_actions[normalized_actions > 1] = 1
normalized_actions = normalized_actions.astype(np.float32)

# Extract rewards (90-day survival outcome)
rewards_array = rewards.to_numpy().astype(np.float32)
# Generate 'done' flags
# [:-1] and [1:] select all values except the last and first value, respectively, allowing pairwise comparison.
done = (df['ICUSTAY_ID'].values[:-1] != df['ICUSTAY_ID'].values[1:]).astype(np.float32)
done = np.append(done, 1.0)

# Create a new dictionary to store processed data
data_dict = {
    'states': states,
    'actions': normalized_actions,
    'rewards': rewards_array,
    'done': done
}

# Save the new dictionary to a PKL file
print(data_dict)

pkl_file = os.path.join(data_dir, 'processed_mimic3_data.pkl')
with open(pkl_file, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"New dataset saved to {pkl_file}")
