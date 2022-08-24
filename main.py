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

# Own
from load_feather import *

# Globals
SAMPLING_INTERVAL_S = 0.05
SAMPLING_INTERVAL_US = SAMPLING_INTERVAL_S * 1000000

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
        tmp_ts_resampled = tmp_ts.sample(sampling_period=datetime.timedelta(microseconds=SAMPLING_INTERVAL_US), start=start_timestamp, end=end_timestamp, interpolate=mode)

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

def resample_bagfile(bag_pandas, sensor_keys, encounter_ground_truth):
    # Get time bounds
    start_timestamp, end_timestamp = find_common_bagfile_bounds(bag_pandas)

    sensors_resampled = dict()
    for sensor_key in sensor_keys:
        # Copy data into tmp dataframe
        tmp_df = bag_pandas.dataframes[sensor_key].dataframe.copy()

        # Resample stream from current sensor
        sensors_resampled[sensor_key] = resample_dataframe(tmp_df, start_timestamp, end_timestamp, phone_mode=True, mode='linear')

    # Add ground truth
    sensors_resampled['ground_truth'] = resample_dataframe(encounter_ground_truth, start_timestamp, end_timestamp, phone_mode=False, mode='previous')

    return sensors_resampled

def any_y_true(y):
    if np.count_nonzero(y + 1) == 0:
        return np.ones_like(y) * -1

    if len(np.where(y==1)[0]) >= len(np.where(y==0)[0]):
        return np.ones_like(y)

    else:
        return np.zeros_like(y)

def any_y_true_bin(y):
    if np.any(y):
        return np.ones_like(y)

    else:
        return np.zeros_like(y)

def perc_y_true_bin(y):
    if np.mean(y) >= 0.10:
        return np.ones_like(y)

    else:
        return np.zeros_like(y)

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

    # Use these sensors
    sensor_keys = ['magnetic_field_sensor_0', 'pressure_sensor_0']

    # Hyperparameters
    window_length = 100
    window_stride = 10

    # Script settings
    use_savepoint_resample = True
    create_savepoint_resample = True
    savepoint_resample = 'resampled_df_list_3.pickle'

    use_savepoint_window = True
    use_savepoint_classifier = True


    # Switch between cases depending on flag state
    if use_savepoint_resample:
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