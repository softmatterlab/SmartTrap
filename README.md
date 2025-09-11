# SmartTrap
Files needed for the SmartTrap system. The main folder contains all the python the files needed for the user interface and automation. 
## Hardware
The SmartTrap system is built on the MiniTweezers system, details of which you can find here: <http://tweezerslab.unipr.it/cgi-bin/home.pl>
Here we provide lists of the components you need to assemble your own system and some guidlines of how to do so.
### Components

### Schematics and custom mechanical components.

### Assembly instructions


## Electronics
The custom electronics controller for the optical tweezers system consists of 3 separate PCBs. These are great.

### Electronic board designs
The 3 circuit boards of the controller have separate functions listed below. This enables users to replace just a single board if need be, e.g. to use a different type of position sensor.
- PCB 1 - Microcontroller board: Hosts the microcontroller which is an Arduino Portenta H7. This board is connected to both the other boards and the host computer.
- PCB 2 - Sensor board:
- PCB 3 - Actuator board:

### Bill of Materials

### Firmware
The 
Can be reprogrammed easily using a USB-C cable connection to a host computer. For this we recommend the use of Arduinos IDE which you can find here: <https://www.arduino.cc/en/software/>

### Separate controllers
Separate controllers are used for some of the components.

- Laser power control: 
- Microfluidics: 
- Motorized objective movement: This is an optional addition and allows the user to move the objectives from the user interface for adjusting focus. Is an arduino UNO. The program for it can be found in the firmware folder.
- Pipette puller: 

### Drivers

### Connecting the controllers
— video showing the addition of the electronics (and possibly the unit testing and examples?)



### Testing

## Graphical user interface
### Installation
To install the Graphical user interface (GUI) first download the files. It is recommended to create a separate python environment for the SmartTrap software and to use python 3.10 as well as a computer running Windows 11 and a CUDA ready graphics card.
The software makes use of several third party packages which are listed in the PackagesNeeded.txt file. The packages needed for YOLO are installed by running the pip install -r requirements.txt command.

### Running the software
Once the installation is complete the program is run from the command prompt with the command:

python OT_GUI.py

Note this command needs to be run from the folder where the software is placed.
### Neural networks
The weight of the networks needed for the automation are stored in the folder /NeuralNetworks and the weights of a pretrained network are available in the file YOLOV5Weights.pt. 
Note that other custom trained networks can be loaded directly from the user interface. 
## Firmware
The /Firmware folder contains the cCode which runs on the microcontroller (an Aarduino Pportenta).
To install, download the code and open the project in the Arduino IDE (https://www.arduino.cc/en/software/). Then, once the firmware has been uploaded onto the arduino it will automatically run when turned on. 

## Schematics
The /Instrument Schematics Schematicsfolder contrains the schematics of the electronics controller.

# Supplementary Videos
The /Supplementary Videos folder contains 5 videos which showcase the capabilites of the SmartTrap system.

# Cite us!
You can find more information in our paper:
<https://doi.org/10.48550/arXiv.2505.05290>
```
Martin Selin, Antonio Ciarlo, Giuseppe Pesce, Lars Bengtsson, Joan
Camunas-Soler, Vinoth Sundar Rajan, Fredrik Westerlund, L. Marcus Wil-
helmsson, Isabel Pastor, Felix Ritort, Steven B. Smith, Carlos Bustamante,
and Giovanni Volpe. SmartTrap: Automated Precision Experiments with
Optical Tweezers, May 2025. arXiv:2505.05290 [physics]
```
