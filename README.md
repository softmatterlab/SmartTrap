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

The Bill of Materials (BOM) files are in the components folder.

### Firmware
The firmware needed for the controller can be found in the folder fimrware. 
To flash it to the controller do the following.
 - Connect the microcontroller to the host computer. For this use a USB-C cable.
 - Download the folder.
 - Open the folder in the arduino IDE. Arduinos IDE which you can find here: <https://www.arduino.cc/en/software/>
 - Under tools select the arduino portenta H7 board and the correct COM port (should show up as Arduino Portenta )
 - Hit upload in the top left corner.
Once correctly uploaded and connected to both the controller a green light will be flashing periodically on the arduino.

Also other IDEs such as Visual Studio code are possible to use but require a bit more work to set up.

### Separate controllers
Not all the components of the SmartTrap are controlled directly by the electronics controller used for the tweezers instrument. Listed below are the separate controllers are used. All have interfaces in the python GUI which can be used to instead use your own other controllers with minimal changes
- Laser power control: 
- Microfluidics: Pump and valves used are from the ElveFlow, in particular the OB1 microfluidics pump and the ... valve controller. Links:
- Micropipette: Uses a ... pump controlled by a separate power supply.
- Motorized objective movement: This is an optional addition and allows the user to move the objectives from the user interface for adjusting focus. Is an arduino UNO. The program for it can be found in the firmware folder.
- Pipette puller: 

### Drivers

### Connecting the controllers
— video showing the addition of the electronics (and possibly the unit testing and examples?)


### Testing

## Graphical user interface
### Installation
To install the Graphical user interface (GUI) first download the files. It is recommended to use Anaconda (<https://www.anaconda.com/>) to create a separate python environment for the SmartTrap.
The software is tested with Python 3.13 as well as a computer running Windows 11 and a CUDA ready graphics card.
Once the environment is 
- Open a terminal in the desired environment
- Navigate to the target folder with the downloaded files
- Run the command: "pip install -r requirements.py" to install the required packages
  - Optional: Install pytorch with CUDA support, check which <https://pytorch.org/get-started/locally/>

The software needs to know which ... component si connected to which port. To configure this open up the windows device manager.
Change in the config file and insert the appropriate COM ports to have the software automatically connect to the devices. Note that port numbers can change if for instance the computer is updated, if that happens just update the config file accordingly.

### Running the software
Once the installation is complete the program is run from the command prompt with the command:
"python main.py" 
This will start the interface and open it up.

Note this command needs to be run from the folder where the software is placed.
### Neural networks
The weight of the networks needed for the automation are stored in the folder /NeuralNetworks and the weights of a pretrained network are available in the file YOLOV5Weights.pt. 
Note that other custom trained networks can be loaded directly from the user interface. 

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
