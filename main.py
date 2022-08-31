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


def get_common_time_bounds(list_of_df):
    start_index = []
    end_index = []
    for dataframe in list_of_df[:-1]:
        start_index.append(dataframe.index[0])
        end_index.append(dataframe.index[-1])

    return np.max(start_index), np.min(end_index)


def calc_y(y_complete, win_beg_bound, win_end_bound, acceptance_amount):
    # Get values for boundary
    win_beg_index = y_complete.index.searchsorted(pd.Timestamp(win_beg_bound), side='left') - 1
    win_end_index = y_complete.index.searchsorted(pd.Timestamp(win_end_bound), side='right') + 1

    # Cut boundary from complete df
    y_cut = y_complete.iloc[win_beg_index: win_end_index]

    # Calc y value
    y_value = perc_y_true(y_cut.direction.values, acceptance_amount)

    # Get Distance for y-value
    z_value = np.average(y_cut[y_cut.direction == y_value].distance.values)

    return y_value, z_value

def generate_sliding_windows(in_data, length, stride, use_keys, positive_subsampling=False, norm_mag=False, use_diff=False):
    # Get time bounds
    start_time, end_time = get_common_time_bounds(in_data)

    # Generate window bounds
    # TODO: Add stride from parameter
    win_beg_bounds = pd.date_range(start = start_time, end = end_time - pd.Timedelta(value=length, unit='s'),
                                   freq = str(stride) + 'S').values
    win_end_bounds = pd.date_range(start = start_time + pd.Timedelta(value=length, unit='s'), end = end_time,
                                   freq = str(stride) + 'S').values

    # Empty X and y for Output
    X = []
    y = []
    z = []

    # Iterate over all window positions
    for win_beg_bound, win_end_bound in zip(win_beg_bounds, win_end_bounds):
        tmp_X = []

        # Window for X
        for df, key in zip(in_data[:-1], use_keys.keys()):
            if key == 'magnetic_field_sensor_0' and norm_mag:
                df['norm'] = np.linalg.norm(df[['x','y', 'z']].values, axis=1)
                df = df.drop(['x','y', 'z'], axis=1)

            # Get samples for timeframe
            df_filtered = df.loc[(df.index > win_beg_bound) & (df.index <= win_end_bound)]

            # Create features from window X
            tmp_X.extend(df_filtered.describe().drop('count').values.ravel(order='F'))

        # Append to X collector
        X.append(tmp_X)

        # Window of y columns
        curr_y, curr_z = calc_y(in_data[-1][-1].resample('100L').bfill(), win_beg_bound, win_end_bound, 0.1)
        y.append(curr_y)
        z.append(curr_z)

        # Makes halve stride length move and adds that X as well
        if positive_subsampling and curr_y in [0, 1]:
            # Add another y
            y.append(curr_y)
            z.append(curr_z)

            # Shift window by halve stride
            win_beg_bound += pd.Timedelta(stride/2, unit='s')
            win_end_bound += pd.Timedelta(stride/2, unit='s')

            tmp_X = []
            # Window for X
            for df, sub_key in zip(in_data[:-1], use_keys.keys()):
                if sub_key == 'magnetic_field_sensor_0' and norm_mag:
                    df['norm'] = np.linalg.norm(df[['x', 'y', 'z']].values, axis=1)
                    df = df.drop(['x', 'y', 'z'], axis=1)

                # Get samples for timeframe
                df_filtered = df.loc[(df.index > win_beg_bound) & (df.index <= win_end_bound)]

                # Create features from window X
                tmp_X.extend(df_filtered.describe().drop('count').values.ravel(order='F'))

            # Append to X collector
            X.append(tmp_X)

    return np.array(X), np.array(y), np.array(z), len(X)


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


def generate_timeseries_true_positives(encounter_db, bagfile, begin_bag, end_bag):
    # Trim to relevant entries
    encounter_db = encounter_db[encounter_db["bag_file"] == bagfile]
    encounter_db = encounter_db[encounter_db["is_encounter"] == True]

    # Iterate over encounters and make entries for begin and end
    encounter_list = []

    #Add initial datapoint
    encounter_list.append([begin_bag, 0, -1, np.nan])

    for counter, line in encounter_db.iterrows():
        # TODO: fix issue of zero duration encounters
        if line['end'] == line['begin']:
            continue

        # Add begin
        begin_ts = pd.to_datetime(line['begin'], unit='s')
        encounter_list.append([begin_ts - pd.Timedelta(1,unit='ns'), 0, -1, np.nan])
        encounter_list.append([begin_ts, 1, line['direction'], line['distance']])

        # Add end
        end_ts = pd.to_datetime(line['end'], unit='s')
        encounter_list.append([end_ts - pd.Timedelta(1,unit='ns'), 1, line['direction'], line['distance']])
        encounter_list.append([end_ts, 0, -1, np.nan])

    # Add final datapoint
    encounter_list.append([end_bag, 0, -1, np.nan])

    # Make to pandas
    encounter_ground_truth = pd.DataFrame(encounter_list, columns=['timestamp_bagfile', 'ground_truth', 'direction', 'distance'])
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
    window_length = 7.5
    window_stride = 0.75
    norm_of_magnetometer = False
    use_diff = True

    sensor_keys = ['magnetic_field_sensor_0', 'pressure_sensor_0']
    columns_X = ['magnetic_field_sensor_0_x', 'magnetic_field_sensor_0_y', 'magnetic_field_sensor_0_z', 'pressure_sensor_0_fluid_pressure']
    columns_y = ['ground_truth_direction']
    use_keys = dict(magnetic_field_sensor_0=['x', 'y', 'z'], pressure_sensor_0=['fluid_pressure'])

    use_savepoint_window = False
    create_savepoint_window = True
    savepoint_window = 'sp_window_1.pickle'

    use_savepoint_classifier = False
    create_savepoint_classifier = True
    savepoint_classifier = 'sp_classifier_1.pickle'

    # LOAD DATA --------------------------------------------------------------------------------------------------------
    bag_list = []
    for counter, bagfile_name in enumerate(tqdm(trajectory_db.name.values)):
        #print(bagfile_name[:-4])

        # Load bagfile
        bagfile_path = os.path.join(working_dir, bagfile_name[:-4])
        bag_pandas = Data_As_Pandas(bagfile_path)
        bag_pandas.load_from_working_directory()

        # Get ground truth for current bagfile and
        encounter_ground_truth = generate_timeseries_true_positives(encounter_db, os.path.split(bagfile_path)[-1], pd.to_datetime(bag_pandas.overview['general_meta']['start_time_unix'], unit='s'), pd.to_datetime(bag_pandas.overview['general_meta']['end_time_unix'], unit='s'))

        # Put data into list for further processing
        df_list = []
        for sensor_key in use_keys.keys():
            df_list.append(correct_with_phone_times(bag_pandas.dataframes[sensor_key].dataframe))

        # Add weather info for additional stats
        df_list.append(bag_pandas.overview['weather']['hourly'] + [encounter_ground_truth])
        bag_list.append(df_list)

    # SLIDING WINDOW ---------------------------------------------------------------------------------------------------
    # Switch between cases depending on flag state
    if not use_savepoint_window:
        if norm_of_magnetometer:
            parameter = 14
        else:
            parameter = 28

        collector_X = np.empty((0, parameter))
        collector_y = np.empty((0))
        collector_z = np.empty((0))

        for df_list in tqdm(bag_list):
            # Generate windows
            X, y, z, X_shape = generate_sliding_windows(df_list, window_length, window_stride, use_keys, positive_subsampling=True, norm_mag=norm_of_magnetometer, use_diff=use_diff)
            # working: 150/25; 100/10

            # Concat to existing array
            collector_X = np.concatenate((collector_X, X), axis=0)
            collector_y = np.concatenate((collector_y, y), axis=0)
            collector_z = np.concatenate((collector_z, z), axis=0)

            if create_savepoint_window:
                with open(savepoint_window, 'wb') as handle:
                    pickle.dump((collector_X, collector_y, collector_z), handle)

    else:
        collector_X, collector_y, collector_z = pickle.load(open(savepoint_window, 'rb'))

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(collector_X, collector_y, collector_z, test_size=0.2, random_state=0,
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