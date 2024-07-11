import numpy as np
from threading import Thread
from time import sleep
import pickle
import os
import pandas as pd


class DataSaverThread(Thread):

    def __init__(self,c_p,data_channels):
        Thread.__init__(self)
        self.c_p = c_p
        self.data_channels = data_channels
        self.running = True
        self.sleep_time = 0.1
        self.start_idx = 0
        self.start_idx_motors = 0
        self.saving = False
        self.data_idx = 0

    def start_saving(self):
        self.start_idx = self.data_channels['PSD_A_P_X'].index
        self.start_idx_motors = self.data_channels['Motor_x_pos'].index # Fewer data points for motors
        self.saving = True
        self.data_idx += 1
        self.filename = self.c_p['recording_path'] + '/' + self.c_p['filename'] + str(self.data_idx)

        print("Saving started")

    def save_data(self):
            # Convert data to DataFrame
            #data = {channel: channel_data.data for channel, channel_data in self.data_channels.items()}
            df_new = pd.DataFrame(self.get_data_dict())

            # Append mode for CSV
            if not os.path.exists(self.filename):
                df_new.to_csv(self.filename, mode='w', index=False)  # Create new file and write headers
            else:
                df_new.to_csv(self.filename, mode='a', header=False, index=False)  # Append without headers

    def get_data_dict(self):
        self.stop_idx = self.data_channels['PSD_A_P_X'].index
        self.stop_idx_motors = self.data_channels['Motor_x_pos'].index
        # TODO this won't handle channels sampled at different speeds(such as those tracked by DL thread) particularly well
        
        sleep(0.1) # Waiting for all channels to reach this point
        data = {}

         # TODO make sure that it saves continously and not just once at the end...
         # TODO handle the case in which some data saving is toggled on/off while recording the data. Maybe turn on/off saving?
        for channel in self.data_channels:
            if self.data_channels[channel].saving_toggled:
                if channel in self.c_p['multi_sample_channels'] or channel in self.c_p['derived_PSD_channels']:
                    if self.start_idx < self.stop_idx:
                        data[channel] = self.data_channels[channel].data[self.start_idx:self.stop_idx]
                    else:
                        data[channel] = np.concatenate([self.data_channels[channel].data[self.start_idx:],
                                                        self.data_channels[channel].data[:self.stop_idx]])
                else:
                    if self.start_idx_motors < self.stop_idx_motors:
                        data[channel] = self.data_channels[channel].data[self.start_idx_motors:self.stop_idx_motors]
                    else:
                        data[channel] = np.concatenate([self.data_channels[channel].data[self.start_idx_motors:],
                                                        self.data_channels[channel].data[:self.stop_idx_motors]])

        self.start_idx = self.stop_idx
        self.start_idx_motors = self.stop_idx_motors
        return data

    def run(self):
        while self.running:
            
            if self.c_p['saving_data']:
                if not self.saving:
                    self.start_saving()
                else:
                    self.save_data()
            if not self.c_p['saving_data'] and self.saving:
                self.stop_saving()
            sleep(self.sleep_time)




    """
    def stop_saving(self):
        # TODO ensure that the there is no maximum limit on the filesize
        # - too many ifs and elses to make the code nice
        # - test this carefully and see if there is some better way to do this,
        #    i.e can we handle also the other channels such as computer time efficiently here?
        # - derived channels did not get saved correctly here, check if the fix works.
        #   Need to add parameter to the channel to indicate the sampling rate of it.
        # For instance by stopping, saving and making a new file. Not the most elegant solution but it should work.
        self.saving = False
        print("Saving stopped")
        # BETTER to use a different format. This is not very efficient.
        self.stop_idx = self.data_channels['PSD_A_P_X'].index
        self.stop_idx_motors = self.data_channels['Motor_x_pos'].index
        sleep(0.1) # Waiting for all channels to reach this point
        data = {}
         # TODO make sure that it saves continously and not just once at the end...
        for channel in self.data_channels:
            if self.data_channels[channel].saving_toggled:
                if channel in self.c_p['multi_sample_channels'] or channel in self.c_p['derived_PSD_channels']:
                    if self.start_idx < self.stop_idx:
                        data[channel] = self.data_channels[channel].data[self.start_idx:self.stop_idx]
                    else:
                        data[channel] = np.concatenate([self.data_channels[channel].data[self.start_idx:],
                                                        self.data_channels[channel].data[:self.stop_idx]])
                else:
                    if self.start_idx_motors < self.stop_idx_motors:
                        data[channel] = self.data_channels[channel].data[self.start_idx_motors:self.stop_idx_motors]
                    else:
                        data[channel] = np.concatenate([self.data_channels[channel].data[self.start_idx_motors:],
                                                        self.data_channels[channel].data[:self.stop_idx_motors]])

        filename = self.c_p['recording_path'] + '/' + self.c_p['filename'] + str(self.data_idx)
        with open(filename, 'wb') as f:
                pickle.dump(data, f)
    """