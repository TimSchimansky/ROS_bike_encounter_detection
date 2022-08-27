# Native
import os
import datetime
import time

# 3rd party
import pandas as pd
import numpy as np
import traces
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle
from sklearn.metrics import *

# Own
from load_feather import *

# Globals
SAMPLING_INTERVAL_S = 0.05
SAMPLING_INTERVAL_US = SAMPLING_INTERVAL_S * 1000000


def generate_sliding_windows(in_data_df, length, stride, col_X, col_y, positive_subsampling=False):
    # Generate window bounds
    win_beg_bounds = np.arange(0, len(in_data_df) - length, stride)
    win_end_bounds = win_beg_bounds + length

    # Empty X and y for Output
    X = []
    y = []

    # Create norm of magnetometer
    in_data_df['magnetic_field_sensor_0_norm'] = np.linalg.norm(in_data_df[['magnetic_field_sensor_0_x',
                                                                            'magnetic_field_sensor_0_y',
                                                                            'magnetic_field_sensor_0_z']].values,
                                                                axis=1)
    col_X = [X for X in col_X if 'magnetic_field_sensor' not in X]
    col_X.append('magnetic_field_sensor_0_norm')

    # Iterate over all window positions
    for win_beg_bound, win_end_bound in zip(win_beg_bounds, win_end_bounds):
        # Window of x and y columns
        win_X = dataframe.iloc[win_beg_bound:win_end_bound][col_X]
        win_y = dataframe.iloc[win_beg_bound:win_end_bound][col_y]

        # Create y value from window
        curr_y = perc_y_true(win_y.values, 0.1)
        y.append(curr_y)

        # Create features from window X
        X.append(win_X.describe().drop('count').values.ravel(order='F'))

        if positive_subsampling and curr_y in [0, 1]:
            # Add another y
            y.append(curr_y)

            # Shift window by halve stride
            win_beg_bound += int(stride / 2)
            win_end_bound += int(stride / 2)

            win_X = dataframe.iloc[win_beg_bound:win_end_bound][col_X]
            X.append(win_X.describe().drop('count').values.ravel(order='F'))

    return np.array(X), np.array(y)


def balance_to_middle_class(X, y):
    # Determine occurance of class with medium most values
    destination_value = int(np.median(np.unique(y, return_counts=True)[1]))
    print(f"All classes are resampled to {destination_value} samples")

    X_result_list = []
    y_result_list = []

    for class_value in np.unique(y):
        # Get indices for class occurance
        class_indices = np.where(y == class_value)[0]

        # Skip or switch replacement modi
        if len(class_indices) == destination_value:
            X_result_list.append(X[class_indices, :])
            y_result_list.append(y[class_indices])
            continue
        elif len(class_indices) < destination_value:
            is_smaller = True
        else:
            is_smaller = False

        # Resample data
        X_tmp, y_tmp = resample(X[class_indices, :], y[class_indices], replace=is_smaller, n_samples=destination_value)
        X_result_list.append(X_tmp)
        y_result_list.append(y_tmp)

    return np.vstack(X_result_list), np.hstack(y_result_list)


def generate_timeseries_true_positives(encounter_db, bagfile, begin_bag):
    # Trim to relevant entries
    encounter_db = encounter_db[encounter_db["bag_file"] == bagfile]
    encounter_db = encounter_db[encounter_db["is_encounter"] == True]

    # Iterate over encounters and make entries for begin and end
    encounter_list = []

    #Add initial datapoint
    encounter_list.append([begin_bag, 0, -1])

    for counter, line in encounter_db.iterrows():
        # TODO: fix issue of zero duration encounters
        if line['end'] == line['begin']:
            continue

        # Add begin
        encounter_list.append([pd.to_datetime(line['begin'], unit='s'), 1, line['direction']])

        # Add end
        encounter_list.append([pd.to_datetime(line['end'], unit='s'), 0, -1])

    # Make to pandas
    encounter_ground_truth = pd.DataFrame(encounter_list, columns=['timestamp_bagfile', 'ground_truth', 'direction'])
    encounter_ground_truth.set_index('timestamp_bagfile', inplace=True)

    return encounter_ground_truth


def rescale_pandas_column(input_series):
    # Get value center to zero
    series = input_series - input_series.mean()

    # Get value range down
    return (series) / (series.max() - series.min())


def round_datetime(dt, floor=True):
    """if floor is false, ceil wil be used"""
    closest_microseconds = SAMPLING_INTERVAL_US

    if floor:
        microseconds_new = np.floor(dt.microsecond // (closest_microseconds / 10) / 10) * closest_microseconds
    else:
        microseconds_new = np.ceil(dt.microsecond // (closest_microseconds / 10) / 10) * closest_microseconds
        if microseconds_new > 999999:
            microseconds_new = np.floor(dt.microsecond // (closest_microseconds / 10) / 10) * closest_microseconds

    # Replace microsecond value in dt
    return dt.replace(microsecond=int(microseconds_new)).replace(nanosecond=0)


def correct_with_phone_times(input_df):
    # Replace index with more stable sensor timestamps (with applied offset)
    input_df.index = input_df.timestamp_sensor.values + np.min(input_df.index - input_df.timestamp_sensor)

    # Remove column of sensor timestamp
    return input_df.drop("timestamp_sensor", axis=1)


def resample_dataframe(input_df, start_timestamp, end_timestamp, phone_mode=False, mode='linear'):
    """Phone mode: The data from the phone sensors is streamed via wifi. Therefore, the timestamp in the bagfile is not
    stable. The timestamp of the sensor is very stable but wrong (needs offset)"""
    # Switch between bagfile and sensor timestamp for phone data
    if phone_mode:
        input_df = correct_with_phone_times(input_df)
    else:
        if "timestamp_sensor" in input_df.keys():
            input_df.drop("timestamp_sensor", axis=1)

    # Column collector
    pandas_columns = []

    # Repeat for all topics
    for key in input_df.keys():
        # Convert to traces timeseries
        tmp_ts = traces.TimeSeries(input_df[key])

        # Resample at an interval of 10ms
        tmp_ts_resampled = tmp_ts.sample(sampling_period=datetime.timedelta(microseconds=SAMPLING_INTERVAL_US),
                                         start=start_timestamp, end=end_timestamp, interpolate=mode)

        # Convert to df
        pandas_columns.append(pd.DataFrame(tmp_ts_resampled, columns=['timestamp', key]).set_index('timestamp'))

    # Merge columns back together
    output_df = pd.concat(pandas_columns, axis=1)
    return output_df


def find_common_bagfile_bounds(bag_pandas):
    # Create empty lists for begin and end timestamps
    start_timestamps = []
    end_timestamps = []

    # Iterate over all keys
    for key in bag_pandas.dataframes:
        start_timestamps.append(min(bag_pandas.dataframes[key].dataframe.index))
        end_timestamps.append(max(bag_pandas.dataframes[key].dataframe.index))

    # Get largest common bounds
    start_timestamp = max(start_timestamps)
    end_timestamp = min(end_timestamps)

    # Round to full 10ms
    start_timestamp = round_datetime(start_timestamp, floor=False)
    end_timestamp = round_datetime(end_timestamp, floor=True)
    return start_timestamp, end_timestamp


def resample_bagfile(bag_pandas_tmp, sensor_keys, encounter_ground_truth_tmp):
    # Get time bounds
    start_timestamp, end_timestamp = find_common_bagfile_bounds(bag_pandas_tmp)

    sensors_resampled = dict()
    for sensor_key in sensor_keys:
        # Copy data into tmp dataframe
        tmp_df = bag_pandas_tmp.dataframes[sensor_key].dataframe.copy()

        # Resample stream from current sensor
        sensors_resampled[sensor_key] = resample_dataframe(tmp_df, start_timestamp, end_timestamp, phone_mode=True,
                                                           mode='linear')

    # Add ground truth
    sensors_resampled['ground_truth'] = resample_dataframe(encounter_ground_truth_tmp, start_timestamp, end_timestamp,
                                                           phone_mode=False, mode='previous')

    return sensors_resampled


def any_y_true(y_tmp):
    if np.count_nonzero(y_tmp + 1) == 0:
        return np.ones_like(y_tmp) * -1

    if len(np.where(y_tmp == 1)[0]) >= len(np.where(y_tmp == 0)[0]):
        return np.ones_like(y_tmp)

    else:
        return np.zeros_like(y_tmp)


def any_y_true_bin(y_tmp):
    if np.any(y_tmp):
        return np.ones_like(y_tmp)

    else:
        return np.zeros_like(y_tmp)


def perc_y_true_bin(y_tmp):
    if np.mean(y_tmp) >= 0.10:
        return np.ones_like(y_tmp)

    else:
        return np.zeros_like(y_tmp)

def perc_y_true(y_tmp, percentage):
    if np.count_nonzero(y_tmp + 1) <= percentage * len(y_tmp):
        return -1

    if len(np.where(y_tmp == 1)[0]) >= len(np.where(y_tmp == 0)[0]):
        return 1

    else:
        return 0

if __name__ == "__main__":
    # -------------------------------------------------------------------------------
    # Set working directory
    working_dir = 'H:/bagfiles_unpack/'

    # Get detected encounters
    encounter_db = pd.read_feather(os.path.join(working_dir, 'encounter_db_v2_backup_after_manual.feather'))
    encounter_db = encounter_db.sort_values("begin")
    encounter_db = encounter_db.drop_duplicates(subset=["begin", "end"])

    # Get trajectories
    trajectory_db = pd.read_feather(os.path.join(working_dir, 'trajectory_db.feather'))

    # Hyperparameters
    window_length = 150
    window_stride = 15

    sensor_keys = ['magnetic_field_sensor_0', 'pressure_sensor_0']
    columns_X = ['magnetic_field_sensor_0_x', 'magnetic_field_sensor_0_y', 'magnetic_field_sensor_0_z', 'pressure_sensor_0_fluid_pressure']
    columns_y = ['ground_truth_direction']

    use_savepoint_resample = False
    create_savepoint_resample = True
    savepoint_resample = 'sp_resample_1.pickle'

    use_savepoint_window = False
    create_savepoint_window = True
    savepoint_window = 'sp_window_1.pickle'

    use_savepoint_classifier = False
    create_savepoint_classifier = True
    savepoint_classifier = 'sp_classifier_1.pickle'

    # RESAMPLING -------------------------------------------------------------------------------------------------------
    # Switch between cases depending on flag state
    if not use_savepoint_resample:
        # Create empty list
        resampled_data = []

        # Iterate over all bagfiles
        for counter, bagfile_name in enumerate(trajectory_db.name.values):
            print(bagfile_name[:-4])

            # Load bagfile
            bagfile_path = os.path.join(working_dir, bagfile_name[:-4])
            bag_pandas = Data_As_Pandas(bagfile_path)
            bag_pandas.load_from_working_directory()

            # Get ground truth for current bagfile and
            encounter_ground_truth = generate_timeseries_true_positives(encounter_db, os.path.split(bagfile_path)[-1], pd.to_datetime(bag_pandas.overview['general_meta']['start_time_unix']))

            # Resample chosen sensors and fuse in one dataframe
            sensors_resampled = resample_bagfile(bag_pandas, sensor_keys, encounter_ground_truth)
            resampled_data_df = pd.concat(sensors_resampled, axis=1)
            resampled_data_df.columns = resampled_data_df.columns.map('_'.join).str.strip('_')
            resampled_data.append(resampled_data_df)

        if create_savepoint_resample:
            with open(savepoint_resample, 'wb') as handle:
                pickle.dump(resampled_data, handle)

    else:
        resampled_data = pickle.load(open(savepoint_resample, 'rb'))

    # SLIDING WINDOW ---------------------------------------------------------------------------------------------------
    # Switch between cases depending on flag state
    if not use_savepoint_window:
        collector_X = np.empty((0, 14))
        collector_y = np.empty((0))

        for dataframe in tqdm(resampled_data):
            # Generate windows
            X, y = generate_sliding_windows(dataframe, window_length, window_stride, columns_X, columns_y, positive_subsampling=True)
            # working: 150/25; 100/10

            # Concat to existing array
            collector_X = np.concatenate((collector_X, X), axis=0)
            collector_y = np.concatenate((collector_y, y), axis=0)

            if create_savepoint_window:
                with open(savepoint_window, 'wb') as handle:
                    pickle.dump((columns_X, columns_y), handle)

    else:
        collector_X, collector_y = pickle.load(open(savepoint_window, 'rb'))

    X_train, X_test, y_train, y_test = train_test_split(collector_X, collector_y, test_size=0.2, random_state=0,
                                                        shuffle=True)

    X_train_balanced, y_train_balanced = balance_to_middle_class(X_train, y_train)
    X_train_balanced, y_train_balanced = shuffle(X_train_balanced, y_train_balanced, random_state=0)

    sc = StandardScaler()
    X_train_balanced = sc.fit_transform(X_train_balanced)
    X_test = sc.transform(X_test)

    clf = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=20)
    clf.fit(X_train_balanced, y_train_balanced)

    o = clf.predict(X_test)

    print("Klassifikationsreport")
    print(classification_report(y_test, o))
    print("Konfusionsmatrix")
    print(confusion_matrix(y_test, o))
    print("Feature Wichtigkeit")
    print(clf.feature_importances_)

    """plt.plot(o, alpha=0.5)
    plt.plot(y_test, alpha=0.5)
    plt.show()"""

    print(1)