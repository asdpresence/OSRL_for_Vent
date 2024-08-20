# Perform missing value imputation and preprocessing on raw medical data,
# then save the processed data as a DataFrame and persist it.

# Importing necessary libraries
import sys
import os
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

# Import custom function named preprocess_imputation to perform missing value imputation on the DataFrame
current_dir = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER_PATH = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
from imputation_utils import preprocess_imputation

DATA_TABLE_FILE_NAME        = os.path.join(DATA_FOLDER_PATH, "pre_imputed.csv")
IMPUTED_DATAFRAME_PATH      = os.path.join(DATA_FOLDER_PATH, "imputed.pkl")

# Load data into pandas DataFrame
df = pd.read_csv(DATA_TABLE_FILE_NAME)

# Print the shape of the DataFrame after dropping unnecessary columns
df = df.drop(columns=['BILIRUBIN', 'ALBUMIN', 'PTT', 'PT', 'INR', 'LACTATE'])
print(df.shape)

# Define a function to calculate the ratio of missing values
def calc_null_ratio(group):
    total_values = group.size  # Get the total number of values in the group (rows * columns)
    null_values = group.isnull().sum().sum()  # Count the total number of missing values
    null_ratio = null_values / total_values  # Calculate the ratio of missing values
    return null_ratio

# Calculate the proportion of missing values
empty_ratio = df.groupby('ICUSTAY_ID').apply(calc_null_ratio)

# Remove groups with a missing value ratio greater than 0.8
# Find groups with a missing value ratio greater than 0.8
groups_to_remove = empty_ratio[empty_ratio > 0.8].index
# Remove those groups from the DataFrame
df_filtered = df[~df['ICUSTAY_ID'].isin(groups_to_remove)]
print(df_filtered)

# Specify n and k values for k-nearest neighbors imputation
IMPUTED_DATA_DIR_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
IMPUTATION_N                = 1
IMPUTATION_K                = 3

# Impute missing values
# After imputing missing values, drop rows that still contain missing values
# to ensure the final processed data has no missing values.
df = preprocess_imputation(df_filtered, IMPUTATION_N, IMPUTATION_K)

print(df.shape)

# Identify columns with missing values
columns_with_missing_values = df.columns[df.isnull().any()]

# Output columns that contain missing values
print("Columns with missing values:")
print(columns_with_missing_values)

# Drop rows with missing values
df = df.dropna()

print(df.shape)

# Store the imputed and processed DataFrame data into a pickle file
df.to_pickle(IMPUTED_DATAFRAME_PATH)
