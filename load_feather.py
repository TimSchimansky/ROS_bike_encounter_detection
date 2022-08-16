# Native
import os
import json

# 3rd party
import pandas as pd
import geopandas as gpd

class Dataframe_With_Meta:
    def __init__(self, dataframe, meta, orig_file_name):
        # Put dataframe into struct
        self.dataframe = dataframe

        # Add meta info
        self.message_type = meta['message_type']
        self.orig_topic_name = meta['topic_name']
        self.orig_file_name = orig_file_name
        self.frequency = meta['frequency']
        self.message_count = meta['message_count']
        self.is_geo = meta['is_geo']
        self.is_in_folder = meta['is_in_folder']
        self.orig_file_type = '.feather'

class Data_As_Pandas:
    def __init__(self, directory):
        self.working_directory = directory

        # Load overview json
        with open(os.path.join(self.working_directory, 'overview.json'), 'r') as f:
            self.overview = json.loads(f.read())

        # remove working directory entry
        self.working_directory_original = self.overview['working_directory']

        # Create sensor list
        self.sensor_list = self.overview['sensor_streams']

        # Create empty dict for pandas dataframes
        self.dataframes = dict()

    def load_from_working_directory(self, exclude=[]):
        # Iterate over available files
        for key, value in self.sensor_list.items():
            # Check if folder or 1d data
            if not value['is_in_folder'] and key not in exclude:
                # Data is available as pandas dataframe in feather file
                import_file_path = os.path.join(self.working_directory, key + '.feather')

                # Decide between pandas and geopandas dataframe
                if value['is_geo']:
                    # Import data and save in dictionary as geopandas
                    self.dataframes[key] = Dataframe_With_Meta(gpd.read_feather(import_file_path), value, key)
                else:
                    # Import data and save in dictionary as pandas
                    self.dataframes[key] = Dataframe_With_Meta(pd.read_feather(import_file_path), value, key)

                # Change unix timestamp to datetime
                self.dataframes[key].dataframe['timestamp_sensor'] = pd.to_datetime(self.dataframes[key].dataframe['timestamp_sensor'], unit='s')
                self.dataframes[key].dataframe['timestamp_bagfile'] = pd.to_datetime(self.dataframes[key].dataframe['timestamp_bagfile'], unit='s')
                self.dataframes[key].dataframe.set_index('timestamp_bagfile', inplace=True)