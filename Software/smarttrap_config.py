
import numpy as np

# TODO be consistent with the use of port or adress
# This will be (quite) systemd dependent.

def get_devices():
    """
    Return the devices used in the smarttrap system.
    """
    pass


def get_defualt_config():
    """
    Here is the default configuration of the smartTrap system
    """
    config = {
        ############### Ports #########################
        'COM_port': 'COM4', # COM port used with the tweezers controller
        'objective_stepper_port': 'COM10',
        'laser_A_port':'COM12',
        'laser_B_port':'COM11',
        'pump_adress': 'COM5',
        'valve_adress': 'COM3',
        'pipette_pump_adress': 'COM6',

        # Paths
        'recording_path': '../TestData/',
        'yolo_path': "NeuralNetworks/YOLOV5Weights.pt",
        'default_z_model_path': "NeuralNetworks/Z_model_large_range.pth",

        ############ Calibration parameters##################
        # Position calibration
        'laser_a_transfer_matrix': np.array([ 13.62547902 , 0.39582976, -0.98140442, 
                                                 13.65848671]), 
        'laser_b_transfer_matrix': np.array([ -13.75365959 , -2.95635977,-2.87762914,
                                                16.21314373]), 
        'PSD_to_pos': [14.08,13.80,13.89,13.04], # System specific            
        # Force calibration
        'PSD_to_force': np.array([0.0173, 0.0164, 0.0178, 0.0182]), # Calibration - system specific
        'Photodiode_sum_to_force': [1200,-700,210], # The calibration factor for the photodiode/PSD sum channel to force
        'reflection_A': 0.0693, # Used to calculate the actual laser power in the sample.
        'reflection_B': 0.0816,
        'sum2power_A': 0.00692*94/135,
        'sum2power_B': 0.00682*94/135,
        'reflection_fac': 1.0057, # Factor relatets to the compensation when calculating the true sum readings.
        # Camera and motor calibration (TODO should be moved to camera and motor classes)
        'ticks_per_micron': 6.24,#24.45, # How many ticks per micron
        'microns_per_tick': 1/6.24, #0.0408, # How many microns per tick
        'ticks_per_pixel': 6.24/(18.28*1.15), #1.337, # How many pixels per micron
        # other
        'valves_used': [1,3], # indices of the valves used
    }
    return config

def save_config():
    pass

def load_config():
    pass

def create_devices(c_p):
    """
    Creates and initializes the various hardware controllers used in the optical tweezers
    setup, including cameras, object tracking, motor controls, lasers, and microfluidics
    controllers. This method sets up the necessary connections and configurations for each
    controller based on the provided control parameters (`c_p`). It handles exceptions
    that may arise during the initialization of each controller, ensuring that the system
    can continue to operate even if some components fail to initialize.

    # NOTE that it is in here changes are made to use different controllers
    """

    # Set up the camera
    camera = None
    try:            
        # Cameras from two manufacturors are currently implemented, Thorlabs and Basler.
        # They use different classes. To change manufacturor change the camera_type in the
        # control parameters
        if c_p['camera_type'] == "Thorlabs":
            print("Thorlabs camera selected")
            from thorlabs_scientific_cameras import ThorlabsScientificCamera as TSC
            camera = TSC()
        else:
            print("Basler camera selected")
            from basler_cameras import BaslerCamera
            camera = BaslerCamera()            
        c_p['camera_width'] = camera.get_sensor_size()[0]
        c_p['camera_height'] = camera.get_sensor_size()[1]
    except Exception as E:
        print(f"Camera error!\n{E}")


    # Set up the object tracker
    from smarttrap_tracker import ObjectTrackerYOLO, ParticleCNN  # noqa: F401
    object_tracker = ObjectTrackerYOLO(
        YOLO_model_path=c_p['yolo_path'],
        z_model_path=c_p['default_z_model_path'],
        particle_size_limits = [1.3/c_p['microns_per_pix'], 7/c_p['microns_per_pix']],
        )

    # Set up the motor controllers
    from smarttrap_motors import ObjectiveMotor, SmartTrapMotor

    motor_controller = SmartTrapMotor(c_p, data_channels)
    
    objective_motor = ObjectiveMotor()
    objective_motor.connect(c_p['objective_stepper_port'])

    from OSTech_laser_controller import OSTechLaserController
    laser_A = OSTechLaserController()
    laser_B = OSTechLaserController()

    # Set up the microfluidics controllers
    from elvesys_pump import (
        ElvesysMicrofluidicsController, MUXWireValveController, PipettePump)
    microfluidics_controller = ElvesysMicrofluidicsController()
    try:
        microfluidics_controller.connect(c_p['pump_adress'])
    except Exception as E:
        print(E)
        print("Could not connect to the microfluidics pump")

    valve_controller = MUXWireValveController()
    try:
        valve_controller.connect(c_p['valve_adress'])
    except Exception as E:
        print(E)
        print("Could not connect to the valve controller")

    pipette_pump = PipettePump()
    try:
        pipette_pump.connect(c_p['pipette_pump_adress'])
    except Exception as E:
        print(E)
        print("Could not connect to the pipette pressure controller")