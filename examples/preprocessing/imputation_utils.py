from sklearn import metrics
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dateutil.parser import parse
import argparse
import fancyimpute
import pyprind
import pytz
from fancyimpute import KNN
import math

"""
This function implements the "sample-and-hold" missing value imputation method, 
which is suitable for time-varying clinical trial data such as the MIMIC-III dataset.
The function accepts parameters including the input DataFrame `input`, the holding period parameter `hold_period` 
(which specifies the holding time for each variable), and an optional `adjust` parameter.
The function iterates over each row of the DataFrame to impute missing values. 
If the current time step corresponds to a different ICUSTAY_ID from the next time step, 
the observation value at the last time step is held for the current time step.
By iterating through different columns and rows, missing values are gradually filled, 
resulting in a new imputed DataFrame.
"""


def sample_and_hold(input: pd.DataFrame, hold_period: np.array, adjust=0):
    '''
      Implementation of "sample-and-hold" imputation, for time-varying clinical trial data such as MIMIC-III.
      Args:
        input: pd.DataFrame, with chart times in correct order
        hold_period: specific holding period, if specified for vital signs
        adjust = 0: adjustment level
    '''
    icustayid_list = list(input['ICUSTAY_ID'])
    i_ICUSTAY_ID = input.columns.get_loc('ICUSTAY_ID')
    i_charttime = input.columns.get_loc('START_TIME')
    temp = np.copy(input)
    n_row, n_col = temp.shape
    last_charttime = np.empty_like(input.iloc[0, i_charttime], shape=n_row)
    last_charttime[0] = parse(temp[0, i_charttime])
    last_value = np.empty(n_col)
    last_value.fill(-float(math.inf))
    curr_stay_id = temp[0, i_ICUSTAY_ID]

    mean_sah = np.nanmean(input.iloc[:, 4:n_col], axis=0)
    # loop through each column
    for i in range(4, n_col):
        for j in range(n_row):
            if curr_stay_id != temp[j, i_ICUSTAY_ID]:
                # start over since we've encountered a new ICUSTAY_ID
                last_charttime[j] = parse(temp[j, i_charttime])
                last_value = np.empty(n_col)
                last_value.fill(-float(math.inf))
                curr_stay_id = temp[j, i_ICUSTAY_ID]
            if not np.isnan(temp[j, i]):
                # set the replacement if we observe one later
                last_charttime[j] = parse(temp[j, i_charttime])
                last_value[i] = temp[j, i]
            if j >= 0:
                time_del = 0
                if (np.isnan(temp[j, i])) and (temp[j, i_ICUSTAY_ID] == curr_stay_id) and (
                        time_del <= hold_period[i - 3]):
                    # replacement if NaN via SAH
                    if last_value[i] == -float(math.inf):
                        k = j
                        while (k < temp.shape[0] and np.isnan(temp[k, i]) and curr_stay_id == temp[k, i_ICUSTAY_ID]):
                            k += 1
                        # If entire episode is NaN, then replace by mean
                        if k == temp.shape[0] or curr_stay_id != temp[k, i_ICUSTAY_ID]:
                            temp[j, i] = mean_sah[i - 4]
                            last_value[i] = mean_sah[i - 4]
                        # Else, there is at least one non-NaN value in this episode
                        else:
                            last_value[i] = temp[k, i]
                            temp[j, i] = temp[k, i]
                    else:
                        temp[j, i] = last_value[i]
        print("completed preprocessing {}".format(i))
    return temp


"""
This function performs preprocessing and imputation on the input DataFrame, 
combining "sample-and-hold" and KNN imputation methods.
It accepts parameters such as `df` (the initial DataFrame), `N` (the block size for KNN), 
`k_param` (the parameter for KNN), and optional parameters `weights`, `col_names_knn`, and `col_names_sah`.
First, it filters columns based on the extent of missing values, removing columns with a high missing rate from the DataFrame.
The data is then divided into two parts: the portion with a high level of missing values is imputed using 
the "sample-and-hold" method, while the portion with a low level of missing values is imputed using the KNN method.
Finally, the imputed DataFrame is combined, resulting in a DataFrame that has been preprocessed and imputed.
"""


def preprocess_imputation(df: pd.DataFrame, N: int, k_param: int, weights=None, col_names_knn=None, col_names_sah=None):
    """
    Preprocesses input DataFrame `df` via SAH and KNN imputation.
      N - KNN block size
      k - KNN parameter
      df - initial DataFrame in the correct column name representation
    """
    df_drop = df
    # Separate data for SAH and KNN imputation, keeping 'subject_id', 'ICUSTAY_ID', 'hadm_id', 'start_time' columns intact:
    miss_level = df_drop.isnull().sum() / df_drop.shape[0]
    id_part = df_drop[['ICUSTAY_ID', 'START_TIME']]
    df_sah = df_drop.loc[:, (miss_level >= 0.3)]
    df_sah = pd.concat([id_part, df_sah], axis=1)

    # Remove deathtime as it doesn't need to be imputed in our case
    df_sah.drop(labels='DEATHTIME', axis=1, inplace=True)
    df_knn = df_drop.loc[:, (miss_level < 0.3)]

    # For high levels of missingness: use SAH
    hp = np.full(len(df_sah.columns), np.inf)
    temp = sample_and_hold(df_sah, hold_period=hp)
    df_sah = pd.DataFrame(data=temp, columns=df_sah.columns)  # New imputed version (to combine later)

    # For small levels of missingness: use KNN
    c_s = N * 1000
    knn_removed_cols = df_knn[['ICUSTAY_ID', 'START_TIME', 'GENDER', 'DISCHTIME']]
    df_knn_dropped_columns = df_knn.drop(labels=['ICUSTAY_ID', 'START_TIME', 'GENDER', 'DISCHTIME'], axis=1).columns
    temp = df_knn.drop(labels=['ICUSTAY_ID', 'START_TIME', 'GENDER', 'DISCHTIME'], axis=1).to_numpy()
    for i in range(0, temp.shape[0], (N * 1000)):
        if (i + (N * 1000) > temp.shape[0] - 1):
            temp[i:temp.shape[0] - 1, :] = KNN(k=k_param).fit_transform(temp[i:temp.shape[0] - 1, :])
        else:
            temp[i:i + c_s, :] = KNN(k=k_param).fit_transform(temp[i:i + c_s, :])  # Impute via KNN
    df_knn_imp = pd.DataFrame(data=temp, columns=df_knn_dropped_columns)
    df_knn = knn_removed_cols.join(df_knn_imp)
    final = df_sah.set_index(['ICUSTAY_ID', 'START_TIME']).join(other=df_knn.set_index(['ICUSTAY_ID', 'START_TIME']))
    return final.reset_index(drop=False)
