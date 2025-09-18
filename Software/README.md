# This folder contains all the software needed for running the SmartTrap interface

The different 

To use the software with other than the default devices (e.g. a camera from a different manufacuter) implement the protocol of the corresponding device. 
These protocols need to be implemented for the different devices to work with the interface and the autonomous control functions. In the case of camera implement the "CameraProtocol" from camera_controls.py and update the create_controls function.

## Main

Run this file to start the program with the interface. Adding the flag -testmode will let you run it without any devices connected.

## smarttrap_interface

### Usage
This file contains the graphical user interface of the program.

### Using the software with other devices
In the create_controllers function the different devices are connected. To use your own devices, create a new create_controllers functions.

## Microfluidics controllers
This file defines the interfaces used for the 3 different microfluidics control devices; the pump, the valves and the pipette pump.


### 

### MicrofluidicsControllerWidget
This is a control widget for the microfludics. From this the different functionality of the microfluidics systems can be controlled and monitored graphically.
