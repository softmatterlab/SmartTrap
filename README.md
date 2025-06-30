# SmartTrap
Files needed for the SmartTrap system. The main folder contains all the python the files needed for the user interface and automation. 
## Graphical user interface
### Installation
To install the Graphical user interface (GUI) first download the files. It is recommended to create a separate python environment for the SmartTrap software and to use python 3.10 as well as a computer running Windows 11 and a CUDA ready graphics card.
The software makes use of several third party packages which are lised in the PackagesNeeded file. The packages needed for YOLO are installed by running the pip install -r requirements.txt command.

### Running the software
Once the installation is complete the program is run from the command prompt with the command:

python OT_GUI.py

### Neural networks
The weight of the networks needed for the automation are stored in the folder /NeuralNetworks and the weights of a pretrained network are available in the file YOLOV5Weights.pt. 
Note that other custom trained networks can be loaded directly from the user interface. 
## Firmware
The /Firmware folder contains the cCode which runs on the microcontroller (an Aarduino Pportenta).
To install, download the code and open the project in the Arduino IDE (https://www.arduino.cc/en/software/). Then, once the firmware has been flashed onto the arduino it will automatically run the program when turned on. 

## Schematics
The /Instrument Schematics Schematicsfolder contrains the schematics of the electronics controller.

# Supplementary Videos
The /Supplementary Videos folder contains 5 videos which showcase the capabilites of the SmartTrap system.
