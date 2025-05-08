# SmartTrap
Files needed for the SmartTrap system. The main folder contains all the python the files needed for the user interface and automation.

The Firmware folder has the code which is to be run on the Arduino Portenta.

Schematics for the instrument are in the folder schematics.

## Graphical user interface
### Installation
To run the Graphical user interface (GUI) first download the files. It is recommended to create a separate python environment for the SmartTrap software. Recommended to use python 3.10
The software makes use of several third party packages which are installed by running the pip install -r requirements.txt command.

### Running the software
Once the installation is complete the program is run by from a anaconda prompt with the command:

python OT_GUI.py

### Neural networks
The networks needed for the automation.

Note that other networks can be loaded directly from the user interface.
## Firmware
Code which runs on the microcontroller (an arduino portenta). Is easiest to install from the Arduino IDE https://www.arduino.cc/en/software/

To install download the code and open the project in the Arduino IDE. Next

The microcontroller will automatically run the program when turned on.

## Schematics
Schematics of the electronics controller.

# Supplementary Videos
The supplementary videos folder contains 5 videos which showcase the capabilites of the SmartTrap system.
