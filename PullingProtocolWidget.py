from PyQt6.QtWidgets import QVBoxLayout, QLabel, QSpinBox, QWidget, QApplication, QPushButton, QComboBox,QDoubleSpinBox

from PyQt6.QtCore import Qt

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIntValidator
from PyQt6.QtCore import QTimer, QTime
from threading import Thread
import numpy as np
import serial
from time import sleep, time
from functools import partial



class force_limits_protocoL_widget(QWidget):
    # NOt finished and maybe not needed
    def __init__(self, c_p, data_channels):

        super().__init__()
        self.c_p = c_p
        self.setWindowTitle("Pulling Protocol")
        self.initUI()
        self.data_channels = data_channels
    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel("Protocol controller")
        layout.addWidget(self.label)
    

class PullingProtocolWidget(QWidget):
    def __init__(self, c_p, data_channels):

        super().__init__()
        self.c_p = c_p
        self.setWindowTitle("Pulling Protocol")
        self.initUI()
        self.data_channels = data_channels
        # TODO handle the different piezos and how they are shared in a better way.
        # Maybe have all the controls in the same widget?
        self.protocol_axis_index = 3 # AX=1, AY=2, BX=3, BY=4, BY default
        self.timer = QTimer()
        self.timer.setInterval(100) # sets the delay of the timer and thereby how often it should update.
        self.timer.timeout.connect(self.refresh)
        self.upper_limit_old_value = 60_000 # saved values that are used in case lower limit > upper limit
        self.lower_limit_old_value = 2_000

        self.current_force = 0
        self.previous_force = 0
        self.current_position = 0
        self.previous_position = 0
        #self.previous_force_time = 0
        self.force_move_direction = 1 # 1 for increasing, -1 for decreasing
        self.force_limit_protocol_running = False

        self.timer.start()
        self.updateParameters()  # Initial update

    def initUI(self):
        layout = QVBoxLayout()

        self.lowerLimitSpinBox = QSpinBox()
        self.lowerLimitSpinBox.setRange(0, 65535)
        self.lowerLimitSpinBox.valueChanged.connect(self.updateParameters)
        self.lowerLimitSpinBox.setValue(self.c_p['protocol_data'][3]*256 + self.c_p['protocol_data'][4])
        self.lowerLimitSpinBox.setToolTip("Lower limit of the protocol, in nm. \n NOTE: The lower limit must be smaller than the upper limit!\n also acts to set the threshold force value in the approach to surface experiments.")
        layout.addWidget(QLabel("Lower Limit:"))
        layout.addWidget(self.lowerLimitSpinBox)

        self.upperLimitSpinBox = QSpinBox()
        self.upperLimitSpinBox.setRange(0, 65535)
        self.upperLimitSpinBox.valueChanged.connect(self.updateParameters)
        self.upperLimitSpinBox.setValue(self.c_p['protocol_data'][1]*256 + self.c_p['protocol_data'][2])
        layout.addWidget(QLabel("Upper Limit:"))
        layout.addWidget(self.upperLimitSpinBox)

        self.stepSizeSpinBox = QSpinBox()
        self.stepSizeSpinBox.setRange(0, 65535)
        self.stepSizeSpinBox.valueChanged.connect(self.updateParameters)
        self.stepSizeSpinBox.setValue(self.c_p['protocol_data'][5]*256 + self.c_p['protocol_data'][6])
        layout.addWidget(QLabel("Step Size:"))
        layout.addWidget(self.stepSizeSpinBox)


        # Add toggle protocol button
        self.toggleProtocolButton = QPushButton("Toggle constant speed protocol")
        self.toggleProtocolButton.clicked.connect(self.toggleProtocol)
        self.toggleProtocolButton.setCheckable(True)
        self.toggleProtocolButton.setChecked(self.c_p['protocol_data'][0])
        self.toggleProtocolButton.setToolTip("Toggle the constant speed protocol on/off. \n NOTE: You cannot control either piezo manually when a protocol is running!")
        layout.addWidget(self.toggleProtocolButton)

        self.axisComboBox = QComboBox()
        self.axisComboBox.addItem("A-X")
        self.axisComboBox.addItem("A-Y")
        self.axisComboBox.addItem("B-X")
        self.axisComboBox.addItem("B-Y")
        
        self.axisComboBox.addItem("Force A_X positive")
        self.axisComboBox.addItem("Force B-X positive")
        self.axisComboBox.addItem("Force A-X negative")
        self.axisComboBox.addItem("Force B-X negative")
        self.axisComboBox.addItem("Force A-Y positive")
        self.axisComboBox.addItem("Force B-Y positive")
        self.axisComboBox.addItem("Force A-Y negative")
        self.axisComboBox.addItem("Force B-Y negative")

        self.axisComboBox.addItem("Force A-X positive reverse") # When using this the lower limit is the amount the puller will pull back and the upper limit is the force limit at which it will start reversing.
        self.axisComboBox.addItem("Force B-X positive reverse")
        self.axisComboBox.addItem("Force A-X negative reverse")
        self.axisComboBox.addItem("Force B-X negative reverse")
        self.axisComboBox.addItem("Force A-Y positive reverse")
        self.axisComboBox.addItem("Force B-Y positive reverse")
        self.axisComboBox.addItem("Force A-Y negative reverse")
        self.axisComboBox.addItem("Force B-Y negative reverse")


        self.axisComboBox.setCurrentIndex(4)
        self.axisComboBox.currentIndexChanged.connect(self.selectProtocolAxis)
        layout.addWidget(QLabel("Select axis:"))
        layout.addWidget(self.axisComboBox)

        # FORCE LIMIT PROTOCOL CONTROLS.
        # TODO put them in a separate widget or something to clean it up a bit.

        self.lowerForceSpinBox = QDoubleSpinBox()
        self.lowerForceSpinBox.setRange(-120,120)
        self.lowerForceSpinBox.setValue(0)
        self.lowerForceSpinBox.setToolTip("Lower force limit in constant speed force protocol")
        layout.addWidget(QLabel("Lower force limit"))
        layout.addWidget(self.lowerForceSpinBox)

        self.upperForceSpinBox = QDoubleSpinBox()
        self.upperForceSpinBox.setRange(-120,120)
        self.upperForceSpinBox.setValue(0)
        self.lowerForceSpinBox.setToolTip("Upper force limit in constant speed force protocol")
        layout.addWidget(QLabel("Upper force limit"))
        layout.addWidget(self.upperForceSpinBox)

        self.forceLimitProtocolToggle = QPushButton("Toggle force limit protocol")
        self.forceLimitProtocolToggle.clicked.connect(self.toggleForceProtocol)
        self.forceLimitProtocolToggle.setCheckable(True)
        self.forceLimitProtocolToggle.setChecked(False)
        self.forceLimitProtocolToggle.setToolTip("Toggle the constant speed protocol on/off with specificd force limits. Notes makes use of the force move protocols listed below.")
        layout.addWidget(self.forceLimitProtocolToggle)

        self.setLayout(layout)

    def toggleForceProtocol(self):
        self.force_limit_protocol_running = self.forceLimitProtocolToggle.isChecked()
        if not self.force_limit_protocol_running and self.toggleProtocolButton.isChecked():
            print("De-toggling protocol mover")
            self.toggleProtocolButton.setChecked(False)
            self.c_p['protocol_data'][0] = 0


    def force_limit_protocol(self,axis='Y'):
        """
        This function controls the force limit protocol. It tells the portenta to move along the selected axis (x or y)
        between the force limits set by the user in the forcelimitspinboxes. The movement is performed with the 
        Force A_Y positive (or equivalent) protocol with the limit set very high (e.g 20_000). Once the force limit
        is exceeded (either positive or negative) the direction is switched.
        
        """
        # Adjusted the protocol, now only allows for movement along y and with the bead in the pipette on the bottom, pulling down.
        # Sets the values of the spinboxes for us.
        self.lowerLimitSpinBox.setValue(10_000)
        self.upperLimitSpinBox.setValue(10_000)
        # DO we need to update the parameters?
        self.updateParameters()

        if (not self.toggleProtocolButton.isChecked()) and self.force_limit_protocol_running:
            self.toggleProtocolButton.setChecked(True)
        

        if axis == 'Y':
            self.current_force = np.mean(self.data_channels['F_total_Y'].get_data(100)) # Calculates the current force
            if self.c_p['portenta_command_2'] == 1: # A is being autoaligned
                
                self.current_position = self.data_channels['dac_by'].get_data(1)[0]
                
                if self.force_move_direction == 1:
                    self.axisComboBox.setCurrentIndex(9) # B-Y + protocol
                    self.c_p['protocol_data'][0] = 10
                    print("Setting to BY")
                else:
                    self.axisComboBox.setCurrentIndex(11) # BY - protocol
                    self.c_p['protocol_data'][0] = 12
                    print("Setting to BY")
                

            elif self.c_p['portenta_command_2'] == 2: # B is autoaligned
                self.current_position = self.data_channels['dac_ay'].get_data(1)[0]
                if self.force_move_direction == 1:
                    self.axisComboBox.setCurrentIndex(8) # AY + 
                    self.c_p['protocol_data'][0] = 9
                else:
                    self.axisComboBox.setCurrentIndex(10) # AY -
                    self.c_p['protocol_data'][0] = 11
        else:
            self.current_force = np.mean(self.data_channels['F_total_X'].get_data(100)) # Calculates the current force
            if self.c_p['portenta_command_2'] == 1: # A is being autoaligned
                self.current_position = self.data_channels['dac_bx'].get_data(1)[0]
            elif self.c_p['portenta_command_2'] == 2: # B is autoaligned
                self.current_position = self.data_channels['dac_ax'].get_data(1)[0]

        self.lower_force_limit = self.lowerForceSpinBox.value()
        self.upper_force_limit = self.upperForceSpinBox.value()
        if self.current_force > self.upper_force_limit: # Force increasing
            # Switch direction
            print("Switching move direction",self.current_force , self.previous_force ,self.current_force, self.upper_force_limit)
            self.force_move_direction = -1 # Moving up
        if self.current_force < self.lower_force_limit:
            # Switch direction

            print("Switching move direction",self.current_force , self.previous_force ,self.current_force, self.lower_force_limit)
            self.force_move_direction = 1

        # TODO check if the position exceeds the okay limits.

        if self.current_position > 62_000:
            self.force_move_direction = -1
        elif self.current_position < 2_000:
            self.force_move_direction = 1

        self.previous_force = self.current_force

    def transform_reading2protocol(self):
        lower_force_lim = self.lowerForceSpinBox.value()
        upper_force_lim = self.upperForceSpinBox.value()

        # Convert into psd target values
        pass

    def refresh(self):
        if self.force_limit_protocol_running:
            self.force_limit_protocol()
        
        lower_lim = self.c_p['protocol_data'][3]*256 + self.c_p['protocol_data'][4]
        upper_lim = self.c_p['protocol_data'][1]*256 + self.c_p['protocol_data'][2]
        step_size = self.c_p['protocol_data'][5]*256 + self.c_p['protocol_data'][6]

        # TODO allow to set the lower limit bigger than the upper limit in the GUI, annoying otherwise...
        self.lowerLimitSpinBox.setValue(lower_lim)
        self.upperLimitSpinBox.setValue(upper_lim)
        self.stepSizeSpinBox.setValue(step_size)
        self.toggleProtocolButton.setChecked(self.c_p['protocol_data'][0]>0)

    def getParametersAs8BitArrays(self):

        lower_limit = self.lowerLimitSpinBox.value()
        upper_limit = self.upperLimitSpinBox.value()

        step_size = self.stepSizeSpinBox.value()

        # Function to split a 16-bit number into two 8-bit numbers
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF] 

        return split_16_bit(lower_limit), split_16_bit(upper_limit), split_16_bit(step_size)

    def updateParameters(self):
        lower_limit, upper_limit, step_size = self.getParametersAs8BitArrays()
        # Updates the parameters unless lower lim>upper limit

        if upper_limit < lower_limit:
            print("Upper limit must be larger than lower limit!, not updating parameters")
        else:
            print("Updating parameters", upper_limit, lower_limit)
            self.upper_limit_old_value = upper_limit
            self.lower_limit_old_value = lower_limit
        
        self.c_p['protocol_data'][1:3] = self.upper_limit_old_value # upper_limit
        self.c_p['protocol_data'][3:5] = self.lower_limit_old_value # lower_limit
        self.c_p['protocol_data'][5:7] = step_size
        print(f"Updating parameters Lower limit: {lower_limit}, Upper limit: {upper_limit}, Step size: {step_size}")

    def selectProtocolAxis(self, index):
        
        self.protocol_axis_index = index + 1
        
        
    def toggleProtocol(self):
        if self.toggleProtocolButton.isChecked():
            self.c_p['protocol_data'][0] = self.protocol_axis_index
            print("Setting button to toggled")

        else:
            self.c_p['protocol_data'][0] = 0
       
