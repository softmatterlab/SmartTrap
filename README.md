# SmartTrap system
This GitHub page describes the SmartTrap system, including how to assemble and set up your own system. The main folder contains all the python the files needed for the user interface and automation.

The system is described in our publication: <https://doi.org/10.48550/arXiv.2505.05290>
## Hardware
The SmartTrap system is built on the MiniTweezers system, details of which you can find here: <http://tweezerslab.unipr.it/cgi-bin/home.pl>

Below we provide lists of the components you need to assemble your own system along with guidlines on assembly and installation of the system.
### Components
The components of the system are split into separate units.

### Schematics and custom mechanical components.


### Assembly instructions
Instructions of how to assemble the 

## Electronics
The optical tweezers instrument is controlled by a custom electronics controller. This controller consists of 3 separate circuit boards (PCBs).

### Electronic board designs
The 3 circuit boards of the controller have separate functions listed below. This enables users to replace just a single board if need be, e.g. to use a different type of position sensor.
- PCB 1 - Microcontroller board: Hosts the microcontroller which is an Arduino Portenta H7. This board is connected to both the other boards and the host computer.
- PCB 2 - Sensor board:
- PCB 3 - Actuator board:

The Bill of Materials (BOM) files are in the components folder.

### Firmware
The firmware is the program running on the controller, specifically the microcontroller, to steer it. It needs to be installed for the controller to work and it can be found in the folder Fimrware on this github page.
To install (flash) the firmware onto the controller do the following.

 - Connect the microcontroller to the host computer. For this use a USB-C cable.
 - Download the folder.
 - Open the folder in the arduino IDE. Arduinos IDE which you can find here: <https://www.arduino.cc/en/software/>
 - Under tools select the arduino portenta H7 board and the correct COM port (should show up as Arduino Portenta )
 - Hit upload in the top left corner.
Once the firmware is correctly uploaded, and the microcontroller is connected to both the controller PCB and the host computer, a green light will be flashing periodically on the microcontroller.

Note: also other IDEs, such as Visual Studio Code, are possible to use but require a bit more work to set up.

### Separate controllers
Not all the components of the SmartTrap are controlled directly by the electronics controller used for the tweezers instrument. Listed below are the separate controllers are used. All have interfaces in the python GUI which can be used to instead use your own other controllers with minimal changes
- Laser power control: 
- Microfluidics: Pump and valves used are from the ElveFlow, in particular the OB1 microfluidics pump and the ... valve controller. Links:
- Micropipette: Uses a ... pump controlled by a separate power supply.
- Motorized objective movement: This is an optional addition and allows the user to move the objectives from the user interface for adjusting focus. Is an arduino UNO. The program for it can be found in the firmware folder.
- Pipette puller: 

### Drivers
Each component comes with a python driver class. These drivers are used to communicate between the device and the main program.
The drivers each implements a protocol. To replace the driver, create a class implementing the corresponding protocol and change the import in the main interface.

- Camera
 - Two options currently implemented: Thorlabs scientific cameras and Basler cameras
- Optical tweezers instrument
 - Motors
 - Laser movement
 - Instrument communications
- Microfluidics pump
 - OB1 controller from elvesys
- Microfluidics valves
 - from elvesys
- Pipette pump
 - Sparkfun ... pump which is controlled with a PSU

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

The software needs to know which device is connected to which port. To configure this open up the windows device manager to check the COM ports.
Then change in the config file and insert the appropriate COM ports to have the software automatically connect to the devices. Note that port numbers can change if for instance the computer is updated, if that happens just update the config file accordingly.

### Starting the software
Once the installation is complete the program is run from the command prompt with the command:
"python main.py" 
This will start the interface and open it up.

Note this command needs to be run from the folder where the software is placed.

### Using the interface
The various controls of the instrument are divided into different by different widgets.

#### Testing
You can run the instrument also without any of the devices connected. To do this follow the instructions start the software with the extra argument -testmode by running the command:
"python main.py -testmode"
This will create testdrivers for the different devices which act similarily to the real devices from a software perspective, but are not connected to any physical hardware.
These testdriverds can also be used to test the functionality of just one or two devices separately from the rest of the system during development.

### Autonomous control
There are various autonomous protocols that the instrument can execute.
You can choose to run entire experiments, such as a single molecule force spectroscopy, or just 

### Neural networks
The weight of the networks needed for the automation are stored in the folder /NeuralNetworks and the weights of a pretrained network are available in the file YOLOV5Weights.pt. 
Note that other custom trained networks can be loaded directly from the user interface. 

# Expanding on the software
The software provided here is open source and as such you are free to download and modify it to your own needs, but not to sell it commercially.

## Adapting the software to other systems
The software suit, and in particular the interface and its back end, has been designed with the use in other system and experimental procedures in mind. 
The different protocols of the various devices simply needs to be implemented.

## Making a sample chamber

### Items needed

 **Tools**:
- Laser cutter
- Hotplate
- Pipette puller
- Scalpel
- Tweezers

**Consumables**:
- Parafilm
- Coverglass slides, thickness 1.5 , 60x24 mm
- Micropipette
- Channel capillaries

Both the parafilm and the holes in the coverglasses can be cut with a lasercutter.

**Cutting parafilm**
First prepare the parafilm by cutting it. Place it stretched horisontally on the laser cutter with the paper peeled off. 
You can for instance use the nescofilm roller from the tweezerlab website <http://tweezerslab.unipr.it/cgi-bin/assemblies.pl/Show?_id=ddd9> for this.

Next cut the pattern into the .

**Drilling in glass slides**
To make holes in the glass slides you can either use glass drill or a laser cutter.
The optimal settings will depend on the model of the laser cutter. 

### Assembling a chamber
To assemble a chamber from the prepared material perform the following steps:

- Use a scalpel to cut two pieces of parafilm free.
- Gently peel away the parafilm covering the channels using the tweezers.
- Place the parafilm on top of a glass slides with the holes overlapping as per picture.
- Next place the micropipette in the center of the chamber and the capillaries next to it, see picture.
- Place the second sheet of parafilm on top of the first and align it carefully. Be careful not to move the pipette or the capillaries out of place.
- Place a coverslip without holes on top and align it with the bottom slide.
- Cut away any excess parafilm
- Heat the chamber for ca 4 minutes at 110 C on the hotplate with a weight of ca 400g applied.
 - It is recommended to sandiwich the chamber in two thicker glass slides for more even weight distribution.
- Remove the chamber and let it cool off.

### Installing a chamber

## Pipette puller
When making chambers you need a micropipette. You can make these yourself from glass capillaries using the pipette puller described here. 
The puller consists of a 

- Pipette capillaries:
- Channel capillaries:

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
