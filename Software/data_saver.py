import numpy as np
from threading import Thread
from time import sleep
import pickle
import os
import pandas as pd
import abc
import cv2

class SaverThreadInterface(abc.ABC, Thread):
    @abc.abstractmethod
    def start_saving(self):
        pass

    @abc.abstractmethod
    def is_saving(self):
        pass

    @abc.abstractmethod
    def stop_saving(self):
        pass

    @abc.abstractmethod
    def snapshot(self):
        pass

class DataSaverThread(SaverThreadInterface):
    """
    Thread used to save data in the background while the program is running.
    """

    def __init__(self, c_p, data_channels):
        Thread.__init__(self)
        self.c_p = c_p # Common control parameters
        self.data_channels = data_channels # Data which is to be saved
        self.running = True
        self.sleep_time = 0.1
        self.start_idx = 0
        self.start_idx_motors = 0
        self.start_idx_prediction = 0
        self.saving = False
        self.data_idx = 0

    def start_saving(self):
        self.start_idx = self.data_channels['PSD_A_P_X'].index
        self.start_idx_motors = self.data_channels['Motor_x_pos'].index # Fewer data points for motors
        self.start_idx_prediction = self.data_channels['trapped_particle_x_position'].index
        self.saving = True
        self.data_idx += 1
        self.filename = self.c_p['recording_path'] + '/' + self.c_p['filename'] + str(self.data_idx)

        print("Saving started")

    def save_data(self):
            # Convert data to DataFrame
            df_new = pd.DataFrame(self.get_data_dict())

            # Append mode for CSV
            if not os.path.exists(self.filename):
                # Create new file and write headers
                df_new.to_csv(self.filename, mode='w', index=False)  
            else:
                # Append without headers
                df_new.to_csv(self.filename, mode='a', header=False, index=False)

    def is_saving(self):
        return self.saving

    def snapshot(self, filename_save=None):
        """
        Captures a snapshot of what the camera is viewing and saves that
        in the fileformat specified by the image_format parameter.
        """
        if filename_save is None or isinstance(filename_save, bool):
            filename_save = self.c_p['filename']
        idx = str(self.c_p['image_idx'])
        filename = (self.c_p['recording_path'] + '/' + filename_save + 'image_' + idx + '.' +
                    self.c_p['image_format'])
        if self.c_p['image_format'] == 'npy':
            np.save(filename[:-4], self.c_p['image'])
        else:
            cv2.imwrite(filename, cv2.cvtColor(self.c_p['image'],
                                           cv2.COLOR_RGB2BGR))

        self.c_p['image_idx'] += 1

    def get_data_dict(self):
        self.stop_idx = self.data_channels['PSD_A_P_X'].index
        self.stop_idx_motors = self.data_channels['Motor_x_pos'].index
   
        sleep(0.1) # Waiting for all channels to reach this point
        data = {}

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

    def stop_saving(self):
        """
        Stops the data saving process, collects the recorded data from all relevant channels,
        and saves it to a file.
        This method finalizes the current data recording session by:
        - Setting the saving flag to False.
        - Determining the stop indices for each data channel.
        - Handling cases where channels may not have sampled correctly by adjusting indices.
        - Extracting the relevant data slices for each channel, accounting for channels sampled at
          different rates.
        - Saving the collected data to a file using pickle.
        - Incrementing the data file index for future recordings.
        The method ensures that data from all channels is synchronized as much as possible,
        and handles wrap-around cases where the stop index is less than the start index.
        """

        self.saving = False
        print("Saving stopped")
        self.stop_idx = self.data_channels['PSD_A_P_X'].index
        self.stop_idx_motors = self.data_channels['Motor_x_pos'].index
        self.stop_idx_prediction = self.data_channels['trapped_particle_x_position'].index
        sleep(0.1) # Waiting for all channels to reach this point
        data = {}

        # QuickFIx to error which happens if one of the channels is not sampling correctly.
        if self.start_idx == self.stop_idx:
            self.stop_idx = self.stop_idx + 1

        if self.start_idx_motors == self.stop_idx_motors:
            self.stop_idx_motors = self.stop_idx_motors + 1

        if self.start_idx_prediction == self.stop_idx_prediction:
            self.stop_idx_prediction = self.stop_idx_prediction + 1

        for channel in self.data_channels:
            if self.data_channels[channel].saving_toggled:
                # Handle the different rates at which the channels are sampled to get the right data to be saved.
                if channel in self.c_p['multi_sample_channels'] or \
                    channel in self.c_p['derived_PSD_channels']:

                    if self.start_idx < self.stop_idx:
                        data[channel] = self.data_channels[channel].data[self.start_idx:
                                                                         self.stop_idx]
                    else:
                        data[channel] = np.concatenate(
                            [self.data_channels[channel].data[self.start_idx:],
                             self.data_channels[channel].data[:self.stop_idx]])

                elif channel in self.c_p['prediction_channels']:
                    if self.start_idx_prediction < self.stop_idx_prediction:
                        data[channel] = self.data_channels[channel].data[
                            self.start_idx_prediction:self.stop_idx_prediction]
                    else:
                        data[channel] = np.concatenate(
                            [self.data_channels[channel].data[self.start_idx_prediction:],
                             self.data_channels[channel].data[:self.stop_idx_prediction]])
                else:
                    if self.start_idx_motors < self.stop_idx_motors:
                        data[channel] = self.data_channels[channel].data[
                            self.start_idx_motors:self.stop_idx_motors]
                    else:
                        data[channel] = np.concatenate(
                            [self.data_channels[channel].data[self.start_idx_motors:],
                             self.data_channels[channel].data[:self.stop_idx_motors]])

        filename = self.c_p['recording_path'] + '/' + self.c_p['filename'] + str(self.data_idx)
        if self.c_p['data_file_format'] == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(filename + '.csv', index=False)
        elif self.c_p['data_file_format'] == 'xlsx':
            df = pd.DataFrame(data)
            df.to_excel(filename + '.xlsx', index=False)
        else:
            with open(filename, 'wb') as f:
                    pickle.dump(data, f)
        self.data_idx += 1 # Moved here from start saving

    def run(self):
        while self.c_p['program_running']:
            # Will implement continous saving here.
            sleep(self.sleep_time)
        if not self.c_p['program_running'] and self.saving:
            self.stop_saving()
