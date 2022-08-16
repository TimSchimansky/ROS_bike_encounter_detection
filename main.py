# Native
import os
import datetime
import time

# 3rd party
import pandas as pd
import numpy as np
import traces

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Own
from load_feather import *


def generate_timeseries_true_positives(encounter_db, bagfile, begin_bag):
    # Trim to relevant entries
    encounter_db = encounter_db[encounter_db["bag_file"] == bagfile]
    encounter_db = encounter_db[encounter_db["is_encounter"] == True]

    # Iterate over encounters and make entries for begin and end
    encounter_list = []

    #Add initial datapoint
    encounter_list.append([begin_bag, 0, -1])

    for counter, line in encounter_db.iterrows():
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

def round_datetime(dt, closest_microseconds=10000, floor=True):
    """if floor is false, ceil wil be used"""
    if floor:
        microseconds_new = np.floor(dt.microsecond // (closest_microseconds / 10) / 10) * closest_microseconds
    else:
        microseconds_new = np.ceil(dt.microsecond // (closest_microseconds / 10) / 10) * closest_microseconds

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
        tmp_ts_resampled = tmp_ts.sample(sampling_period=datetime.timedelta(microseconds=10000), start=start_timestamp, end=end_timestamp, interpolate=mode)

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


if __name__ == "__main__":
    # -------------------------------------------------------------------------------
    encounter_db = pd.read_feather("H:/bagfiles_unpack/encounter_db.feather")
    encounter_db = encounter_db.sort_values("begin")
    encounter_db = encounter_db.drop_duplicates(subset=["begin", "end"])

    bagfile = 'H:/bagfiles_unpack/2022-08-08-06-39-53_1'
    bag_pandas = Data_As_Pandas(bagfile)
    bag_pandas.load_from_working_directory()

    sensor_keys = ['magnetic_field_sensor_0', 'pressure_sensor_0']
    # -------------------------------------------------------------------------------

    encounter_ground_truth = generate_timeseries_true_positives(encounter_db, os.path.split(bagfile)[-1], pd.to_datetime(bag_pandas.overview['general_meta']['start_time_unix']))
    sensors_resampled = resample_bagfile(bag_pandas, sensor_keys, encounter_ground_truth)

    print(1)



"""fig,ax = plt.subplots()
# Make plot
ax.plot(bag_pandas.dataframes["pressure_sensor_0"].dataframe.fluid_pressure)

ax2=ax.twinx()
# Make plot
ax2.step(encounter_ground_truth.index, encounter_ground_truth['ground_truth'])"""
