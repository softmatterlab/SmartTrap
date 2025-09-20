# This folder contains all the software needed for running the SmartTrap interface

The different 

To use the software with other than the default devices (e.g. a camera from a different manufacuter) implement the protocol of the corresponding device. 
These protocols need to be implemented for the different devices to work with the interface and the autonomous control functions. In the case of camera implement the "CameraProtocol" from camera_controls.py and update the create_controls function.

## install_auto
Run this file to install the packages used in the SmartTrap system.

## Main

Run this file to start the program with the interface. Adding the flag -testmode will let you run it without any devices connected.

## smarttrap_interface

### Usage
This file contains the graphical user interface of the program.

### Using the software with other devices
In the create_controllers function the different devices are connected. To use your own devices, create a new create_controllers functions.


## camera_controls
Contains the classes used to interface the cameras and connect these to the user interface.
Also has the functionality used to record videos.

## data_saver

## Microfluidics controllers
This file defines the interfaces used for the 3 different microfluidics control devices; the pump, the valves and the pipette pump.

### 

### MicrofluidicsControllerWidget
This is a control widget for the microfludics. From this the different functionality of the microfluidics systems can be controlled and monitored graphically.

## live_plots
This module provides a customizable PyQt6/pyqtgraph plotting window designed for live data visualization during operation. All the different data_channels in the data_channels can be easily plotted and monotired at will.

- **Core class – `PlotWindow`**  
  - Inherits `QMainWindow`.  
  - Manages multiple plots with per-trace metadata stored in a shared `plot_data` dict (`x`, `y`, `pen`, `L`, `sub_sample`, `averaging`).  
  - Updates plots on a `QTimer` loop (`update_plot_data`) pulling from a user-provided `data` mapping.  
  - Designed for `data[key]` objects exposing:  
    - `.get_data(n)` / `.get_data_spaced(n, step)`  
    - `.unit` (for axis labels)  
    - `.index` (for alignment logic).  

- **Supporting dialogs**  
  - `PlotAxisWindow` – manual x/y axis bounds.  
  - `PlotLengthWindow` – number of points retained per trace.  
  - `PlotSubsamplehWindow` – subsampling/averaging interval per trace.  

- **Customization hooks**  
  - Per-trace styling: line color, marker symbol, symbol color.  
  - Background color presets.  
  - Aspect ratio locking.  
  - Non-overlapping averaging helper (`non_overlapping_average`).  

- **Programmer notes**  
  - Plots are registered dynamically with `add_plot(x_key, y_key)`.  
  - Each plot can be removed via `delete_plot(idx)`.  
  - Menus are auto-rebuilt from `plot_data` state (`create_plot_menus`).  
  - Designed to integrate with an external control dict (`c_p`) that tracks program state (e.g. `c_p['program_running']`).  

This structure makes it easy to embed real-time plotting into larger control/analysis pipelines while keeping plotting logic isolated and extensible.

## real_time_tracking
This module defines a protocol for particle and pipette tracking (ObjectTracker) together with a simple test implementation (TestTracker) and a PyQt-based control widget (TrackingControlWidget).

 - ObjectTracker: a protocol specifying the expected interface for tracking backends (frame analysis, particle and pipette detection, z-position prediction, and model loading).
- TestTracker: a mock tracker that generates random particle positions and simulates pipette detection, useful for testing the GUI without real models.
- TrackingControlWidget: a Qt widget for toggling tracking modes, loading 2D/z models, and adjusting z-offsets during experiments.

This provides both the interface layer and a test harness for integrating tracking models into the larger system.

# SmartTrap specific files
Certain devices are specific to the SmartTrap system and thus rely on the specific hardware, e.g. specific microfluidics pump.
To use these devices they implement the protocols specified in the above files.

## basler_cameras
Contains the BaslerCamera class which is an implementaion of the CameraProtocol which can be used with cameras from Basler.
Uses the pypylon package for this

## thorlabs_scientific_cameras
Contains the ThorlabsScientificCamera class which is an implementaion of the CameraProtocol which can be used with cameras from Thorlabs.
To use this you need to have the thorlabs scientific camera sdk installed see <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>

## elvesys_pump

