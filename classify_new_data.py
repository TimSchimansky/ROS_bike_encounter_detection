# Native
import os
import datetime
import warnings

# 3rd party
import pandas as pd
import numpy as np
import traces
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle
from sklearn.metrics import *

# Own
from load_feather import *
from main import *


def generate_sliding_windows_simple(in_mag, in_pressure, length, stride, time_offset):
    # Convert timestamp
    in_pressure['ts'] = pd.to_datetime(time_offset) + pd.to_timedelta(in_pressure['Time (s)'], unit='s')
    in_pressure.set_index('ts', inplace=True)
    in_pressure = in_pressure.drop(['Time (s)'], axis=1)
    in_pressure = in_pressure * 100

    in_mag['ts'] = pd.to_datetime(time_offset) + pd.to_timedelta(in_mag['Time (s)'], unit='s')
    in_mag.set_index('ts', inplace=True)
    in_mag = in_mag.drop(['Time (s)'], axis=1)
    in_mag = in_mag * 10e-7

    # Get time bounds
    start_time, end_time = get_common_time_bounds([in_mag, in_pressure])

    # Generate window bounds
    win_beg_bounds = pd.date_range(start = start_time, end = end_time - pd.Timedelta(value=length, unit='s'),
                                   freq = str(stride) + 'S').values
    win_end_bounds = pd.date_range(start = start_time + pd.Timedelta(value=length, unit='s'), end = end_time,
                                   freq = str(stride) + 'S').values

    # Empty X
    X = []
    t = []

    drop_list = ['count', '25%', '50%', '75%']

    # Iterate over all window positions
    for win_beg_bound, win_end_bound in tqdm(zip(win_beg_bounds, win_end_bounds)):
        t.append(win_beg_bound + ((win_end_bound - win_beg_bound) / 2))

        tmp_X = []

        # Window for X
        for df in [in_mag, in_pressure]:

            # Get samples for timeframe
            df_filtered = df.loc[(df.index > win_beg_bound) & (df.index <= win_end_bound)]

            # Create features from window X
            tmp_X.extend(df_filtered.describe().drop(drop_list).values.ravel(order='F'))
            tmp_X.extend(df_filtered.diff().describe().drop(drop_list).values.ravel(order='F'))

        # Append to X collector
        X.append(tmp_X)

    return np.array(X), len(X), t

#working_dir = '../phyphox/My Experiment 2022-09-12 13-09-27'
#time_offset = '2022-09-12 13:09:27'

#working_dir = '../phyphox/My Experiment 2022-09-13 16-34-53'
#time_offset = '2022-09-13 16:34:53'

working_dir = '../phyphox/My Experiment 2022-09-17 12-33-21'
time_offset = '2022-09-17 12:33:21'

file_list = os.listdir(working_dir)

dict_of_data = dict()

for file_name in file_list:
    file_path = os.path.join(working_dir, file_name)

    if os.path.isdir(file_path):
        continue

    dict_of_data[file_name[:-4]] = pd.read_csv(file_path)

collector_X, collector_y, collector_z, clf, X_train, X_test, y_train, y_test, o, z_train, z_test, sc, SCORERS = pickle.load(open('all_7.5_0.5_diff_small.pickle', 'rb'))

X_new, X_size, t = generate_sliding_windows_simple(dict_of_data['Magnetometer'], dict_of_data['Pressure'], 7.5, 0.2, time_offset)

X_new_scaled = sc.fit_transform(X_new)
o = clf.predict(X_new_scaled)

prox = dict_of_data['Proximity']
prox['ts'] = pd.to_datetime(time_offset) + pd.to_timedelta(prox['Time (s)'], unit='s')
prox.set_index('ts', inplace=True)
prox = prox.drop(['Time (s)'], axis=1)

print(np.unique(o, return_counts=True))

plt.plot(t,o)
plt.plot(prox)

"""colors = [sns.color_palette()[0]] * 7 + [sns.color_palette()[1]] * 7 + [sns.color_palette()[2]] * 7 + [sns.color_palette()[3]] * 7

plt.bar(np.arange(32), clf.feature_importances_, color=colors)"""
plt.show()

print(1)