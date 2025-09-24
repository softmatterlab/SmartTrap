# This folder contains all the software needed for running the SmartTrap interface

The different modules handle different parts of the instrument controls.

To use the software with other than the default devices (e.g. a camera from a different manufacuter) implement the protocol of the corresponding device. 
These protocols need to be implemented for the different devices to work with the interface and the autonomous control functions. In the case of camera implement the "CameraProtocol" from camera_controls.py and update the create_controls function.

Below we describe briefly the functionality of the different modules. More detailed descriptions are available in the separate modules and examples of how to use them are in the Examples folder.

## install_auto
Run this file to install the packages used by the SmartTrap system.

## main
Run this file to start the program with the interface. Adding the flag -testmode will let you run it without any devices connected.

## smarttrap_interface
This contains the graphical user interface (GUI) of the smarttrap system. The entire system can be controlled using the GUI.
Contains the following two classes:
- **MainWindow** - The MainWindow is a subclass of QMainWindow and contains the main GUI.
- **ScreenWorker** - class is used to update the contents of the camera screen, displaying the camera feed and drawing things like particle markers.

### Using the software with other devices
In the create_controllers function the different devices are connected. To use your own devices, create a new create_controllers functions.

## camera_controls
Contains the classes used to interface the cameras and connect these to the user interface. Also has the functionality used to record videos.

The module contains the following classes
- **CameraClicks** - Class used to allow users to click and drag to set the area of interest of the camera. Implements the MouseTool protocol.
- **CameraMeasurement** - Used to measure distance on screen by left and right clicking. Implements the MouseTool protocol.
- **CameraProtocol** - The protocol that camera objects need to implement to function together with the rest of the program.
- **TestCamera** - A mock implementation of CameraProtocol used for testing software without any (physical) camera connected.
- **CameraThread** - A thread which runs in the background and continuously captures images using the connected camera.
- **VideoWriterThread** - Runs in the background and records videos on demand.

## data_saver
This module contains a thread saves data into files when prompted to. By running in a separate thread this limits the risk of the saving of large files interfering with other processes.

Contains the following classes:
- **SaverThreadInterface** - Interface which needs to be implemented by any DataSaver class to function with the interface.
- **DataSaverThread** - Implementation of SaverThreadInterface which is used in SmartTrap to efficiently save the data.

## control_parameters
This file contains classes and functions used to control the various devices and handle data.

Contains the following classes:
- **DataChannel** - A class which stores the data and in an array of fixed length and which can be easily, and efficiently, updated with new data.
- **ControlParametersViewer** - Wdiget which displays the current values of the various control parameters. Used primarily in development to monitor the parameters.
- **CurrentValueWindow** - Class which is a PyQT widget that is used to monitor the various data channels, their current values and an rolling average of the current values. The widget can also be used to select which channels are saved when saving.
- 


## Microfluidics controllers
This file defines the interfaces used for the 3 different microfluidics control devices; the pump, the valves and the pipette pump.
- **MicrofluidicsController (Protocol)** – Interface for multi-channel pressure controllers (connect/disconnect, set/get pressure, channel count).  
- **ValveController (Protocol)** – Interface for valve array controllers (connect, connection status, toggle/query valves).  
- **PipettePump (Protocol)** – Interface for pipette pump PSUs (connect/disconnect, power control, suction on/off, status).  
- **TestMicrofluidicsController** – Mock 3-channel pressure controller (0–2000 mbar, 1-based indexing) for testing without hardware.  
- **TestValveController** – Mock valve controller storing in-memory valve states for UI/logic testing.  
- **TestPipettePump** – Mock pipette pump PSU with % power setting and suction state simulation.  
- **MicrofluidicsMonitorThread** – QThread that syncs target/current pressures, valve states, and pipette pump power with shared state every 500 ms.  
- **ConfigurePumpWidget** – Qt form to configure channel/valve assignments and per-channel max pressures for autonomous use.  
- **MicrofluidicsControllerWidget** – Qt live control panel for setting pressures, toggling valves, and controlling pipette pump with auto-refresh.  

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
This provides both the interface layer and a test harness for integrating tracking models into the larger system.

Contains the following classes:
- **ObjectTracker** - A protocol specifying the expected interface for tracking backends (frame analysis, particle and pipette detection, z-position prediction, and model loading).
- **TestTracker** - A mock tracker that generates random particle positions and simulates pipette detection, useful for testing the GUI without real models.
- **TrackingControlWidget** - Qt widget for toggling tracking modes, loading 2D/z models, and adjusting z-offsets during experiments.

## laser_protocols
The laser protocols define various protocol used to move the lasers in a controlled manner and thereby execute various experimental protocols.
This module defines a common `ExperimentLaserProtocol` interface and several concrete laser-control protocols, plus a Qt widget for selecting, configuring, and running them.

#### Protocol Interface
- **ExperimentLaserProtocol (Protocol)** – Unified API for start/stop/run state, human-readable names/descriptions, and numeric parameters (with descriptions, tooltips, and limits).

#### Protocols
- **ConstantSpeedProtocol** – Moves a laser along a chosen axis at constant speed between two position limits.  
- **Push2ForceProtocol** – Pushes a trapped particle in a given direction until a target force is reached, then stops.  
- **ConstantForceProtocol** – Maintains constant force along X/Y by auto-aligning the second trap and updating force setpoints.  
- **ForceLimitProtocol** – Sweeps between lower/upper force limits at a set speed, reversing direction when thresholds are crossed.  
- **PushAndWaitProtocol** – Pushes toward a force limit, waits for a set duration, then transitions to pulling based on position/force bounds.

#### UI
- **PullingProtocolWidget (Qt)** – Dropdown to pick a protocol, per-parameter editors (labels, tooltips, limits, current values), and a toggle to start/stop; refreshes at 40 ms to call `run_protocol()` while active.


## auto_controller
In this module all the autonomous protocols are defined. 


- **AutoControllerThread** - Thread which runs the autonomous protocols. 
- 
# SmartTrap minitweezers instrument specific files
Certain devices are specific to the SmartTrap system and thus rely on the specific hardware, e.g. specific microfluidics pump.
To use these devices they implement the protocols specified in the above files.

## smarttrap_driver
This module contains the classes that handles communications to and from the SmartTrap electronics controller. Specifically a class which only handles the serial communications running in a separate process and a monitoring thread which parses the commands sent to and from the instrument. During operation both these are running continously.

## basler_cameras
Contains the BaslerCamera class which is an implementaion of the CameraProtocol which can be used with cameras from Basler.
Uses the pypylon package for this

## thorlabs_scientific_cameras
Contains the ThorlabsScientificCamera class which is an implementaion of the CameraProtocol which can be used with cameras from Thorlabs.
To use this you need to have the thorlabs scientific camera sdk installed see <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>

## elvesys_pump
Contains an implementation of the microfluidics controllers that work with the OB1 micrfluidics controller from elvesys as well as their microfluidics valve.

## OSTech_laser_controller
Implementations of the lasercontroller that work with laser current drivers from OSTech.

## smarttrap_tracker
The default tracking used in the SmartTrap system. Here the real-time tracking interfaces is implemented to work with yolov5 and the convoluitonal neural network that monitors the z-position.






## SmartTrap driver
