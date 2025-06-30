# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:48:24 2022

@author: Martin Selin
"""

import numpy as np
from queue import Queue

from PyQt6.QtWidgets import (
 QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import QTimer

import numpy as np

class ControlParametersViewer(QWidget):
    """
    Window for displaying the current values of the data channels.
    """
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        self.setWindowTitle("Control parameters viewer")
        self.resize(800, 800)

        self.vBox = QVBoxLayout()
        self.CreateTable()
        self.vBox.addWidget(self.table)        

        self.timer = QTimer()
        self.timer.setInterval(500) # sets the fps of the timer
        self.timer.timeout.connect(self.set_data)

        self.timer.start()
        self.setLayout(self.vBox)

    def CreateTable(self):

        self.table = QTableWidget(len(self.c_p), 2)  # +2 now for the fps row and a buffer
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])

        for idx, parameter in enumerate(self.c_p):
            self.table.setItem(idx, 0, QTableWidgetItem(f"{parameter}"))
            self.table.setItem(idx, 1, QTableWidgetItem(f"{self.c_p[parameter]}"))

    def set_data(self):
        for idx, parameter in enumerate(self.c_p):
            self.table.setItem(idx, 1, QTableWidgetItem(f"{self.c_p[parameter]}"))
            


def default_c_p():
    """
    Initiates the control parameters to default values.

    Returns
    -------
    c_p : TYPE
        DESCRIPTION. The control parameters. Dictionary containing the important parameters and values
        that need to be shared between different threads or easily monitored from the control parameters window.

    """
    c_p = {
            # General c_p of the UI
           'program_running': True,
           'mouse_params': [0, 0, 0, 0, 0, 0],
           'click_tools': [],
           'central_circle_on': False,

           # Camera and image parameters
           'image': np.ones([500, 500]),#, 1]),
           'image_idx': 0, # Index of snapshot image, used in naming.
           'color': "mono",  # Options are mono and color
           'new_settings_camera': [False, None],
           'camera_width': 1920, # Default camera width, will automatically update when connecting
           'camera_height': 1080,
           'camera_type': "Basler", # Options are Thorlabs and Basler
           'recording': False,
           'exposure_time': 5000,
           'fps': 50,  # Frames per second of camera, measured
           'target_frame_rate': 50, # Target frame rate of the camera, if you want it limited.
           'filename': '',
           'video_name': 'Video',
           'video_format': 'mp4', # Changed default to npy to reduce risk of losing data, alternatives are mp4, avi, npy
           'image_format': 'png',
           'image_gain': 1, 
           'image_offset': 0,
           'AOI':[0,1000,0,1000], # Area of interest of camera
           'recording_path': '../Example data/',
           'bitrate': '30000000', # Bitrate of video to be saved
           'frame_queue': Queue(maxsize=2_000_000),  # Frame buffer essentially
           'image_scale': 1,
           'microns_per_pix': 1/(21.022), #*(4.8/2.74), # Note this parameter is system dependent! 
           # Defualt pixel size is 2.74 microns(Basler 23 um), our Thorcam has 4.8 micron in pixel size. 

           # Temperature c_p
           'temperature_output_on':False,
            
           'piezo_targets': [10,10,10],
           'piezo_pos': [10,10,10],
           'pic_channels':[# Channels to read from the controller
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',
                            'T_time','Time_micros_low','Time_micros_high', # Moved this up
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos',                            
                            'message',
                            'dac_ax','dac_ay','dac_bx','dac_by',
                           ],
           'offset_channels':[
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',
                            'message',
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos'],

            # These are the channels which are sampled once per sample cycle. Default 
            'single_sample_channels':[
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos',
                            'message', # This is used for debugging, i.e sending data from the controller to the computer
                            # for testing purposes.
                            'dac_ax','dac_ay','dac_bx','dac_by',
                            'PSD_Force_A_saved',
            ],

            # These channels are sampled multiple times per sample cycle.
            'multi_sample_channels':[
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',
                            'T_time','Time_micros_low','Time_micros_high', # Moved this up
                            ],

            # These channels have values calculated from the "mulit sample channels" (i.e force converted from PSD reading)
            'derived_PSD_channels': ['F_A_X','F_A_Y','F_B_X','F_B_Y','F_A_Z','F_B_Z',
                                     'F_total_X','F_total_Y','F_total_Z',
                                     'Position_A_X', 'Position_A_Y','Position_B_X','Position_B_Y',
                                     'Position_X','Position_Y'
                                     ],
            'prediction_channels': ['trapped_particle_x_position', 'trapped_particle_y_position', 'trapped_particle_z_position',
                                    'pipette_particle_x_position', 'pipette_particle_y_position', 'pipette_particle_z_position',
                                    'pipette_particle_radii', 'trapped_particle_radii', 'particle_trapped', 'particle_in_pipette',
                                    'prediction_time','trapped_x_force', 'trapped_y_force'],

            'save_idx': 0, # Index of the saved data
            'averaging_interval': 1_000, # How many samples to average over in the data channels window

            # Piezo actuator outputs (for moving the lasers)
            'piezo_A': np.uint16([32768, 32768]), # Target values for the piezos, in the range 0 to 2^15
            'piezo_B': np.uint16([32768, 32768]),
            'portenta_command_1': 0, # Command to send to the portenta, zero force etc.
            'portenta_command_2': 0, # Command to send to the portenta, dac controls, 0=> no autoalign, 1=> autoalign on A, 2=> autoalign on B
            'PSD_means':  np.uint16([0,0,0,0]), # Means of the PSD channels
            'PSD_force_means':  np.array([0,0,0,0]), # Means of the PSD channels
            'PSD_position_means': np.array([0,0,0,0]), # Means of the PSD channels

            # Deep learning tracking
            'network': None, # The type of network used, will default to YOLO.
            'tracking_on': True,
            'z-tracking': True,
            'locate_pipette': True, 
            'draw_pipette': False, # If the pipette should be drawn in the image
            'draw_particles': False, # If the particles should be drawn in the image
            'draw_z_text': False, # If the z-position of the particles should be drawn in the image
            'draw_TnP_particles': False, # If the particles in the trap and pipette should be drawn in the image
            'draw_lasers': False, # If the laser positions should be drawn in the image
            'draw_force': False,

            'crop_width': 64,
            'prescale_factor': 2, # Factor with which the image is to be prescaled before doing the tracking/traing
            'alpha': 1,
            'cutoff': 0.9995,
            # 'train_new_model': False,
            'model':None,
            'z-model':None,
            'device': None, # Pytorch device on which the model runs
            'training_image': np.zeros([64,64]),
            'epochs': 30,
            'epochs_trained': 0,
            'predicted_particle_positions': np.array([]),
            'z-predictions': np.array([]),
            'z-threshold': 8, # Threshold for the z-tracking
            'z-offset': 0, # Offset for the z-tracking used to set a focused particles z-position to 0.
            'predicted_particle_radii': np.array([]),
            'tmp_predictions': np.array([]), # temporary predictions, usd in debugging.
 
            'pipette_located': False,
            'particle_trapped': False,
            'particle_in_pipette': False,
            'accurate_tip_detection_needed': True,
            'multiple_particles_trapped': False,

            'pipette_tilt': 0, # tilt of the pipette, used to find the tip position.
            'Trapped_particle_position': [0,0,0,0], # Position of the trapped particle in the image; x,y,z, radii in pixels
            'pipette_particle_location': [1200,1200,0,0], # Location of the pipette particle in the image, TODO Use either position or location and be consistent in naming
            'pipette_location': [0,0,0,0], # Location of the pipette in the image,x,y position of tip as well as width and height of pipette(in this order).
            'pipette_tip_location': [0,0], # Location of the pipette tip in the image

            'default_unet_path': "NeuralNetworks\TorchBigmodelJune_1", # Not used anymore, remove.
            'yolo_path': "NeuralNetworks\YOLOV5Weights.pt",
            'default_z_model_path': "NeuralNetworks\Z_model_large_range.pth",

            # Autocontroller parameters
            'loop_execution_time': 0, # Time it takes to execute the loop
            'autocontroller_current_step': 'checking_pipette', # The current step of the autocontroller
            'autonomous_experiment_type': 'molecule_stretching', # The type of autonomous experiment to be performed, alternatives are molecule_stretching and electrostatic(currently)
            'autonomous_experiment_types': ['molecule_stretching','electrostatic_interactions', 'RBC_experiment', 'auto_stokes','hairpin_stretching'],

            'autonomous_experiment': False, # If this is toggled we will try to do a full autonomous experiment.
            'autonomous_experiment_states': ['checking_pipette','focusing_pipette','searching_for_particle_1','sucking_into_pipette','searching_for_particle_2','move2area_above_pipette','touching_particles'],
            'centering_on': False,
            'trap_particle': False,
            'search_and_trap': False,
            'focus_z_trap_pipette': False, # Focus the particle in the trap with the one in the pipette
            # parameters for focusing pipette
            'center_pipette': False,
            'suck_into_pipette': False, # used to suck particle into pipette.
            'move2area_above_pipette': False,
            'move_avoiding_particles': False,
            'pipette_z_found': False,
            'move_piezo_2_target': False,
            'touch_particles': False,
            'touch_counter': 0, # Keeps track of how many times we have tried to attach a molecule and failed. Needs to be reset when we have successfully attached a molecule or when starting on a new molecule
            'drop_particle': False, # If set to true the piezos will move to drop the particle(s) in the trap
            'stretch_molecule': False,
            #'attach_DNA_automatically': False,
            'move_particle2pipette': False,
            'find_laser_position': False, # Updates the laser position to the current closest particle
            'focus_pipette': False, # Focus the pipette by optimizing sharpness
            'pipette_focus_startup': True,
            'pipette_sharpnesses': [],
            'pipette_sharpness_Z_pos': [],

            'piezo_target_positions': [0,0,0,0], # A position rading of the PSDs can be saved and moved to on the position sensors, by default this is position 0,0 on both lasers.
            'laser_position_A': [2660, 1502.3255814], # Default
            'laser_position_B': [2660, 1502.3255814], # Default,
            'laser_position_A_predicted': [2660, 1502.3255814],
            'laser_position_B_predicted': [2660, 1502.3255814],
            'laser_position': [2660, 1502.3255814], # Updated as the average of position A and B
            # Laser a approximate x position is lpx = laser_a_transfer_matrix[0]*psd_a_x + laser_a_transfer_matrix[1]*psd_a_y
            'laser_a_transfer_matrix': np.array([ 13.62547902 , 0.39582976, -0.98140442, 13.65848671]), # Remember to recalibrate these every now and then.
            'laser_b_transfer_matrix': np.array([ -13.75365959 , -2.95635977,-2.87762914, 16.21314373]), 

            # Position of the capillaries that push out beads 1 and 2, 1 being the beads going to the pipette and 2 the beads going to the trap
            'capillary_1_position': [0,0,0], 
            'capillary_2_position': [0,0,0],
            'pipette_location_chamber': [0,0,0], # Location of the pipette in the chamber, motor coordinates
            'particle_type': 1, 
            'capillary_1_fluidics_channel': [0, 30, 3], # first number is the channel numnber(0,1 or 2) and the second is the pressure when pushing particles[mbar]. The last is the index of the valve used.
            'capillary_2_fluidics_channel': [2, 30, 3],
            'central_fluidics_channel': [1, 50,1], # The central channel where the pipette is.
            

            # Stretching parmeters
            'molecule_attached': False,
            'stretching_speed': 20, # Speed of stretching in a.u 
            'stretching_distance': 6, # Maximum distance to stretch in microns, without overstretching
            "min_stretch_distance": 4, # Minimum distance to stretch in microns, including overstretching
            'stretch_force': 69, # Maximum force to stretch with in pN in auto-experiments
            'max_force': 100, # Maximum force allowable in pN, essentially the force at which we risk loosing the bead.
            'protocol_limits_dac': [20_000, 40_000], # The limits of the protocol in DAC units,
            'measurement_time': 180, # Time(seconds) during which we will do the stretching experiment.
            'external_save_toggled': False,
            'experiment_finished': False,
            

            # Minitweezers controller parameters
            'COM_port': 'COM4', # The COM port of the minitweezers controller, default is COM6
            'minitweezers_connected': False,
            'blue_led': 0, # Wheter the blue led is on or off, 0 for on and 1 for off
            'objective_stepper_port': 'COM10', # COM4
            #'PSD_bits_per_micron_sum': [0.0703,0.0703], # Conversion factor between the PSD x(or y)/sum channel and microns i.e x/sum / psd_bits_per_micron_sum = microns
            'PSD_to_pos': [14.08,13.80,13.89,13.04], # Updated the 24th of April 2025, system specific
            'PSD_to_force': np.array([0.01699591, 0.01515298, 0.01761022, 0.01883884]), # Calibration - system specific
            'Photodiode_sum_to_force': [1200,-700,210], # The calibration factor for the photodiode/PSD sum channel to force
            'minitweezers_goto_speed': 20_000,

            # Minitweezers protocols parameters
            'protocol_running': False,
            'protocol_type': 'Constant speed', # Options are constant force, constant velocity, constant distance
            'protocol_data': np.uint8(np.zeros(13)),            

            # Minitweezers calibration parameters
            'grid_size': 10, # must match the numbers below. # Changed from 10
            'calibration_points': np.zeros([10,10,17]),
            'calibration_start': True, # Used to tell if the calibration should be reset (started from scratch).
            'calibration_running': False,
            'calibration_performed': False, # Sets to true when a new calibration has been performed and this should be updated in the read-portenta thread

            # Protocol for electrostatic interactions:
            "electrostatic_protocol_toggled": False,
            'electrostatic_protocol_running': False,
            'electrostatic_protocol_finished': False,
            'electrostatic_experiment_alignment': False,
            'electrostatic_auto_experiment': False,
            'electrostatic_touch_force_limit': 10,
            'electrostatic_speed': 2,
            'electrostatic_separation': 0.5, # Separation in microns between the particles surface at maximum
            'electrostatic_protocol_start': 20_000, # First postiion
            'electrostatic_protocol_end': 30_000, # Last postion
            'electrostatic_protocol_steps': 10, # stops of the protocol
            'electrostatic_protocol_duration': 20, # Duration of the protocol in seconds per step

            # RBC experiment parameters
            'RBC_experiment_running': False,
            'RBC_laser_currents': [ 
                                    [87,88,20], # 5 mW / laser
                                    [96,98,5], # 10 mW per laser
                                    [87,88,10],
                                    [110,116,5], # 20
                                    [87,88,10],
                                    [137,147,5], # 42
                                    [87,88,10],
                                    [162,175,5], # 60
                                    [87,88,10],
                                    [187,203,5], # 80
                                    [87,88,20],
                                ], # Ordedered as [laser_A_current, laser_B_current, duration]

            # Laser parameters
            'laser_A_port':'COM12',
            'laser_B_port':'COM11',
            'laser_A_current': 249, # Current in mA
            'laser_B_current': 235, # Current in mA
            'laser_A_on': False,
            'laser_B_on': False,
            'reflection_A': 0.0693, # Used to calculate the actual laser power in the sample.
            'reflection_B': 0.0816,#0.1579,
            'sum2power_A': 0.00692*94/135,
            'sum2power_B': 0.00682*94/135,
            'reflection_fac': 1.0057, #1.0111, # Factor relatets to the compensation when calculating the true sum readings.

            # Microfluidics system  parameters
            'pump_adress': 'COM7', # com 13 before
            'target_pressures': np.array([0.0, 0.0 , 0.0, 0.0]),
            'current_pressures': np.array([0.0, 0.0 , 0.0, 0.0]),
            # Valve parameters
            'valve_controller': None,
            'valve_adress': 'COM3',
            'valves_controller_connected': False,
            'valves_used': [1,3], # indices of the valves used
            'valves_open': [False,False,False,False,False,False,False,False],
            # Parameters for the home made pump for the pipette
            'pump_PSU_adress': 'COM5',
            'pump_PSU_on': False,
            'pump_PSU_max_voltage': 8,
            'pump_PSU_max_current': 1,
            'pump_PSU_current_voltage': 0,
            'pump_PSU_current_current': 0,

           # Minitweezers motors
           'motor_x_target_speed': 0,
           'motor_y_target_speed': 0,
           'motor_z_target_speed': 0,
           'minitweezers_target_pos': [0,0,0], # Should these start at 0?
           'minitweezers_target_speed': [0,0,0],
           'motor_travel_speed': [2_000, 2_000], # 5000 was somewhat high Speed of move to location.
           'move_to_location': False, # Should the motors move to a location rather than listen to the speed?
           'ticks_per_micron': 6.24,#24.45, # How many ticks per micron
           'microns_per_tick': 1/6.24, #0.0408, # How many microns per tick
           'ticks_per_pixel': 6.24/(18.28*1.15), #1.337, # How many pixels per micron
           'saved_positions':[],

           # Stokes test parameters
           'stokes_left_pos': [0,0,0],
           'stokes_right_pos': [0,0,0],
           'stokes_up_pos': [0,0,0],
           'stokes_down_pos': [0,0,0],
           'stokes_center_pos': [0,0,0],
           'stokes_size_threshold': 2,
           'stokes_stage': "stokes_startup",
           'stokes_test_running': False,
           'stokes_test_step': "startup",

           # Hairpins experiment parameters
           'hairpin_experiment_running': False,
           'hairpin_experiment_step': "startup",
           'hairpin_experiment_steps': ["startup","waiting_for_particle","waiting_for_hairpin","waiting_for_stretching","waiting_for_release"],
           'hairpin_max_pull_distance': 1, # Maximum distance the laser will move when trying to attach a hairpin
           'hairpin_counter': 0,
           'hairpin_max_force': 30, # Will use force protocol between min and max force
           'hairpin_min_force': -5,

           'steppers_connected': [False, False, False], # 
        }
    return c_p


from dataclasses import dataclass
# TODO have the max_len be a tunable parameter for configuration.
# TODO implement a custom return funciton for the "derived" channels, i.e the ones which are calculated from other channels.
# TODO have sampling rate as a parameter for the data channels.
@dataclass
class DataChannel:
    name: str
    unit: str
    data: np.array
    saving_toggled: bool = True
    max_len: int = 10_000_000 # 10_000_000 default
    index: int = 0
    full: bool = False
    max_retrivable: int = 1 # number of datapoints which have been saved.

    def __post_init__(self):
        # Preallocate memory for the maximum length
        self.data = np.zeros(self.max_len)

    def put_data(self, d):
        try:
            if len(d) > self.max_len:
                return
        except TypeError:
            d = [d]

        if self.index + len(d) >= self.max_len:
            end_points = self.max_len - self.index
            self.data[self.index:] = d[:end_points]
            self.data[:len(d) - end_points] = d[end_points:]
            self.full = True
            self.index = (self.index + len(d)) % self.max_len
            self.max_retrivable = self.max_len
        else:
            self.data[self.index:self.index + len(d)] = d
            self.index += len(d)
            self.max_retrivable = max(self.index, self.max_retrivable) # Update the maximum number of points which have been saved.

    def get_data(self, nbr_points):
        nbr_points = min(nbr_points, self.max_retrivable)
        if nbr_points <= self.index:
            return self.data[self.index-nbr_points:self.index]
        else:
            return np.concatenate([self.data[self.index-nbr_points:], self.data[:self.index]])

    def get_data_spaced(self, nbr_points, spacing=1):
        nbr_points = min(nbr_points, self.max_retrivable)
        final = self.index
        start = final - (final % spacing) - (nbr_points * spacing)
        if start >= 0:
            return self.data[start:final:spacing]
        else:
            last = (nbr_points * spacing + start) % self.max_len
            return np.concatenate([self.data[start::spacing], self.data[:last:spacing]])

def get_data_dicitonary_new():
    """
    Creates and returns a dictionary of data channels used in the control parameters.
    Each key in the returned dictionary is a string representing the channel name, and each value is a DataChannel object
    initialized with the channel's name, unit, a default value list ([0]), and a boolean indicating if the channel is saved.
    Returns:
        dict: A dictionary mapping channel names to their corresponding DataChannel objects.
    """

    data = [
    ['Time', 'Seconds', False], # Time measured by the computer.
    ['prediction_time','microseconds', True],
    ['particle_trapped','(bool)', False],
    ['trapped_particle_x_position','microns', True],
    ['trapped_particle_y_position','microns', True],
    ['trapped_particle_z_position','microns', True],
    ['trapped_x_force', 'pN', True],
    ['trapped_y_force', 'pN', True],
    ['trapped_particle_radii','microns', True],
    ['particle_in_pipette','(boolish)', False], # Can take values,1,2,0 - 1 No particle in pipette, 2- particle in pipette, 0 no pipette visible
    ['pipette_particle_x_position','microns', True],
    ['pipette_particle_y_position','microns', True],
    ['pipette_particle_z_position','microns', True],
    ['pipette_particle_radii','microns', True],
    ['Temperature', 'Celsius', False],
    ['Motor_x_pos', 'ticks', True],
    ['Motor_y_pos','ticks', True],
    ['Motor_z_pos', 'ticks', True],
    ['Motor_x_speed','microns/s', True],
    ['Motor_y_speed','microns/s', True],
    ['Motor_z_speed','microns/s', True],
    ['Motor time','microseconds', True],
    ['PSD_A_P_X','bits', True],
    ['PSD_A_P_Y','bits', True],
    ['PSD_A_P_sum','bits', True],
    ['PSD_A_F_X', 'bits', True],
    ['PSD_A_F_Y','bits', True],
    ['PSD_A_F_sum','bits', True],
    ['PSD_A_F_sum_compensated','bits', True],
    ['PSD_B_P_X', 'bits', True],
    ['PSD_B_P_Y','bits', True],
    ['PSD_B_P_sum','bits', True],
    ['PSD_B_F_X', 'bits', True],
    ['PSD_B_F_Y','bits', True],
    ['PSD_B_F_sum','bits', True],
    ['PSD_B_F_sum_compensated','bits', True],
    ['Photodiode_A','bits', True],
    ['Photodiode_B','bits', True],
    ['Laser_A_power','mW', True],
    ['Laser_B_power','mW', True],
    ['T_time','microseconds', True], # Time measured on the controller
    ['Time_micros_high','microseconds', False],
    ['Time_micros_low','microseconds', False],
    ['F_A_X','pN', False],
    ['F_A_Y','pN', False],
    ['F_B_X','pN', False],
    ['F_B_Y','pN', False],
    ['F_A_Z','pN', False],
    ['F_B_Z','pN', False],
    ['F_total_X','pN', True],
    ['F_total_Y','pN', True],
    ['F_total_Z','pN', True],
    ['Position_A_X','microns', False],
    ['Position_A_Y','microns', False],
    ['Position_B_X','microns', False],
    ['Position_B_Y','microns', False],
    ['Position_X','microns', True],
    ['Position_Y','microns', True],
    ['PSD_Force_A_saved','pN', False],
    ['Photodiode/PSD SUM A','a.u.', False],
    ['Photodiode/PSD SUM B','a.u.', False],
    ['message','string', False],
    ['dac_ax','bits', True],
    ['dac_ay','bits', True],
    ['dac_bx','bits', True],
    ['dac_by','bits', True],
    ]

    data_dict = {}
    for channel in data:
        data_dict[channel[0]] = DataChannel(channel[0], channel[1], [0], channel[2])
    return data_dict

def get_unit_dictionary(self):
    # Currently not in use
    units = {
        'Time':'(s)',
        'X-force':'(pN)',
        'Y-force':'(pN)',
        'Z-force':'(pN)',
        'Motor_position':'ticks',
        'X-position':'(microns)',
        'Y-position':'(microns)',
        'Z-position':'(microns)',
        'Temperature': 'Celsius',
        'T_time':'Seconds',
    }
    return units
