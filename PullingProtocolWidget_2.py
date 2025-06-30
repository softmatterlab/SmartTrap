from PyQt6.QtWidgets import QVBoxLayout, QLabel, QSpinBox, QWidget, QApplication, QPushButton, QComboBox,QDoubleSpinBox

from PyQt6.QtCore import Qt

from PyQt6.QtGui import QAction, QIntValidator
from PyQt6.QtCore import QTimer, QTime
from threading import Thread
import numpy as np
import serial
from time import sleep, time
from functools import partial



class force_limits_protocoL_widget(QWidget):
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
        self.protocol_axis_index = 1 # AX=1, AY=2, BX=3, BY=4, BY default
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

        self.axisComboBox.addItem("Constant force (A)")
        self.axisComboBox.addItem("Constant force (B)")


        self.axisComboBox.setCurrentIndex(1)  # Default to A-Y
        self.axisComboBox.currentIndexChanged.connect(self.selectProtocolAxis)
        layout.addWidget(QLabel("Select axis:"))
        layout.addWidget(self.axisComboBox)

        self.protoclDescriptionLabel = QLabel('Procol description')
        layout.addWidget(self.protoclDescriptionLabel)

        self.lowerLimitSpinBox = QDoubleSpinBox() # Change to a double spinbox if we want to allow for decimal values.
        self.lowerLimitSpinBox.setRange(0, 65535)
        self.lowerLimitSpinBox.valueChanged.connect(self.updateParameters)
        self.lowerLimitSpinBox.setValue(self.c_p['protocol_data'][3]*256 + self.c_p['protocol_data'][4])
        self.lowerLimitSpinBox.setToolTip("Lower limit of the protocol, in nm. \n NOTE: The lower limit must be smaller than the upper limit!\n also acts to set the threshold force value in the approach to surface experiments.")
        self.lowerLimitConverter = self.lowerLimitSpinBox.value
        self.lower_lim_label = QLabel("Lower Limit:")
        
        layout.addWidget(self.lower_lim_label)
        layout.addWidget(self.lowerLimitSpinBox)

        self.upperLimitSpinBox = QDoubleSpinBox()
        self.upperLimitSpinBox.setRange(0, 65535)
        self.upperLimitSpinBox.valueChanged.connect(self.updateParameters)
        self.upperLimitSpinBox.setValue(self.c_p['protocol_data'][1]*256 + self.c_p['protocol_data'][2])
        self.upperLimitSpinBox.setToolTip("Upper limit of the protocol, in nm. \n NOTE: The upper limit must be larger than the lower limit!")
        self.upperLimitConverter = self.upperLimitSpinBox.value        
        self.upper_limit_label = QLabel("Upper Limit:")
        layout.addWidget(self.upper_limit_label)
        layout.addWidget(self.upperLimitSpinBox)

        self.stepSizeSpinBox = QDoubleSpinBox()
        self.stepSizeSpinBox.setRange(0, 65332) 
        self.stepSizeSpinBox.valueChanged.connect(self.updateParameters)
        self.stepSizeSpinBox.setValue(self.c_p['protocol_data'][5]*256 + self.c_p['protocol_data'][6])
        self.stepSizeConverter = self.stepSizeSpinBox.value
        self.stepSizeLabel = QLabel("Movement speed (nm/s)")
        layout.addWidget(self.stepSizeLabel)
        layout.addWidget(self.stepSizeSpinBox)

        # Add toggle protocol button
        self.toggleProtocolButton = QPushButton("Toggle protocol")
        self.toggleProtocolButton.clicked.connect(self.toggleProtocol)
        self.toggleProtocolButton.setCheckable(True)
        self.toggleProtocolButton.setChecked(self.c_p['protocol_data'][0])
        self.toggleProtocolButton.setToolTip("Toggles the selected protocol on/off. \n NOTE: You cannot control either piezo manually when a protocol is running!")
        layout.addWidget(self.toggleProtocolButton)

        # FORCE LIMIT PROTOCOL CONTROLS.

        self.lowerForceSpinBox = QDoubleSpinBox()
        self.lowerForceSpinBox.setRange(-120,120)
        self.lowerForceSpinBox.setValue(0)
        self.lowerForceSpinBox.setToolTip("Used for input to the constant force and force limit protocols. " \
        "For constant force this sets the force target along X and for force limit it sets the lower force limit in constant speed force protocol.")
        layout.addWidget(QLabel("Constant Force X (or lower force limit)"))
        layout.addWidget(self.lowerForceSpinBox)

        self.upperForceSpinBox = QDoubleSpinBox()
        self.upperForceSpinBox.setRange(-120,120)
        self.upperForceSpinBox.setValue(0)
        self.upperForceSpinBox.setToolTip("Used for input to the constant force and force limit protocols. " \
        "For constant force this sets the force target along Y and for force limit it sets the upper force limit in constant speed force protocol.")
        layout.addWidget(QLabel("Constant force Y (or upper force limit)"))
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


    def configure_widget(self,
                        lower_lim_name = None,
                        lower_lim_range = None,
                        lower_lim_tooltip = None,
                        lower_lim_converter = None,
                        upper_lim_name = None,
                        upper_lim_range = None,
                        upper_lim_tooltip = None,
                        upper_lim_converter = None,
                        step_size_name = None,
                        step_size_range = None,
                        step_size_tooltip = None,
                        step_size_converter = None,
                        protocol_description = None,
                         ):
        """
        This function is used to configure the widget based on the index of the protocol selected.
        For the spinboxes you can set the name and the range of the spinboxes.
        The converter is used to convert the values from the spinboxes to the values used in the
        protocol. The converter is a function that takes the value from the spinbox and returns the value used in the protocol.
        Defaults to 0-65535 for the lower and upper limits, and the value of the spinbox for the converter.
        """
        if lower_lim_name is not None:
            self.lower_lim_label.setText(lower_lim_name)
        if lower_lim_range is not None:
            self.lowerLimitSpinBox.setRange(lower_lim_range[0], lower_lim_range[1])
        else:
            self.lowerLimitSpinBox.setRange(0, 65535)
        if lower_lim_converter is not None:
            self.lowerLimitConverter = lower_lim_converter
        else:
            self.lowerLimitConverter = self.lowerLimitSpinBox.value
        if lower_lim_tooltip is not None:
            self.lowerLimitSpinBox.setToolTip(lower_lim_tooltip)
        
        if upper_lim_name is not None:
            self.upper_limit_label.setText(upper_lim_name)
        if upper_lim_range is not None:
            self.upperLimitSpinBox.setRange(upper_lim_range[0], upper_lim_range[1])
        else:
            self.upperLimitSpinBox.setRange(0, 65535)
        if upper_lim_converter is not None:
            self.upperLimitConverter = upper_lim_converter
        if upper_lim_tooltip is not None:
            self.upperLimitSpinBox.setToolTip(upper_lim_tooltip)
        
        if step_size_converter is not None:
            self.stepSizeConverter = step_size_converter
        else:
            self.stepSizeConverter = self.stepSizeConverterFunc

        if protocol_description is not None:
            self.protoclDescriptionLabel.setText(protocol_description)
        return


    def force_limit_protocol(self,axis='Y'):
        """
        This function controls the force limit protocol. It tells the portenta to move along the selected axis (x or y)
        between the force limits set by the user in the forcelimitspinboxes. The movement is performed with the 
        Force A_Y positive (or equivalent) protocol with the limit set very high (e.g 20_000). Once the force limit
        is exceeded (either positive or negative) the direction is switched.
        
        """
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
                else:
                    self.axisComboBox.setCurrentIndex(11) # BY - protocol
                    self.c_p['protocol_data'][0] = 12             

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
            self.force_move_direction = -1 # Moving up
        if self.current_force < self.lower_force_limit:
            # Switch direction
            self.force_move_direction = 1

        if self.current_position > 62_000:
            self.force_move_direction = -1
        elif self.current_position < 2_000:
            self.force_move_direction = 1

        self.previous_force = self.current_force
    
    def calcConstForceAX(self):
        X = int(self.lowerLimitSpinBox.value() / (self.c_p['PSD_to_force'][0]*2))
        return X+32768

    def calcConstForceAY(self):
        Y = int(self.upperLimitSpinBox.value() / (self.c_p['PSD_to_force'][1]*2))
        return Y+32768
    
    def calcConstForceBX(self):
        X = int(self.lowerLimitSpinBox.value() / (self.c_p['PSD_to_force'][2]*2))
        return X+32768
 
    def calcConstForceBY(self):
        Y = int(self.upperLimitSpinBox.value() / (self.c_p['PSD_to_force'][3]*2))
        return Y+32768
 
    def stepSizeConverterFunc(self):
        return self.stepSizeSpinBox.value()
    

    def refresh(self):
        if self.force_limit_protocol_running:
            self.force_limit_protocol()
        if self.protocol_axis_index < 20:
            # This is to allow for external command of these protocols.
            lower_lim = self.c_p['protocol_data'][3]*256 + self.c_p['protocol_data'][4]
            upper_lim = self.c_p['protocol_data'][1]*256 + self.c_p['protocol_data'][2]
            step_size = self.c_p['protocol_data'][5]*256 + self.c_p['protocol_data'][6]

            self.lowerLimitSpinBox.setValue(lower_lim)
            self.upperLimitSpinBox.setValue(upper_lim)
            self.stepSizeSpinBox.setValue(step_size)
        self.toggleProtocolButton.setChecked(self.c_p['protocol_data'][0]>0)

    def getParametersAs8BitArrays(self):
        lower_limit = int(self.lowerLimitConverter())
        upper_limit = int(self.upperLimitConverter())
        step_size = int(self.stepSizeConverter())

        # Function to split a 16-bit number into two 8-bit numbers
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF] 

        return split_16_bit(lower_limit), split_16_bit(upper_limit), split_16_bit(step_size)

    def updateParameters(self):

        lower_limit, upper_limit, step_size = self.getParametersAs8BitArrays()
        # Updates the parameters unless lower lim>upper limit

        if upper_limit < lower_limit and self.protocol_axis_index<16:
            print("Upper limit must be larger than lower limit!, not updating parameters")
        else:
            self.upper_limit_old_value = upper_limit
            self.lower_limit_old_value = lower_limit
        
        self.c_p['protocol_data'][1:3] = self.upper_limit_old_value # upper_limit
        self.c_p['protocol_data'][3:5] = self.lower_limit_old_value # lower_limit
        self.c_p['protocol_data'][5:7] = step_size

    def selectProtocolAxis(self, index):
        
        self.protocol_axis_index = index + 1

        match self.protocol_axis_index:
            case 1:
                self.configure_widget(lower_lim_name="Lower Limit (A-X)",                                     
                                      lower_lim_tooltip="Lower limit of the protocol, in nm. \n NOTE: The lower limit must be smaller than the upper limit!",
                                      upper_lim_name="Upper Limit (A-X)",
                                      upper_lim_tooltip="Upper limit of the protocol, in nm. \n NOTE: The upper limit must be larger than the lower limit!",
                                      protocol_description = "Moves laser A at constant speed along x-axis. \n Moves between the two positions specified below"
                                      )
                
            case 2:
                self.configure_widget(lower_lim_name="Lower Limit (A-Y)",
                                      lower_lim_tooltip="Lower limit of the protocol, in nm. \n NOTE: The lower limit must be smaller than the upper limit!",
                                      upper_lim_name="Upper Limit (A-Y)",
                                      upper_lim_tooltip="Upper limit of the protocol, in nm. \n NOTE: The upper limit must be larger than the lower limit!",
                                      protocol_description = "Moves laser A at constant speed along y-axis. \n Moves between the two positions specified below",

                                      )
            case 3:
                self.configure_widget(lower_lim_name="Lower Limit (B-X)",
                                      lower_lim_tooltip="Lower limit of the protocol, in nm. \n NOTE: The lower limit must be smaller than the upper limit!",
                                      upper_lim_name="Upper Limit (B-X)",
                                      upper_lim_tooltip="Upper limit of the protocol, in nm. \n NOTE: The upper limit must be larger than the lower limit!",
                                      protocol_description = "Moves laser B at constant speed along X-axis. \n Moves between the two positions specified below",
                                      )
            case 4:
                self.configure_widget(lower_lim_name="Lower Limit (B-Y)",
                                      lower_lim_tooltip="Lower limit of the protocol, in nm. \n NOTE: The lower limit must be smaller than the upper limit!",
                                      upper_lim_name="Upper Limit (B-Y)",
                                      upper_lim_tooltip="Upper limit of the protocol, in nm. \n NOTE: The upper limit must be larger than the lower limit!",
                                      protocol_description = "Moves laser B at constant speed along y-axis. \n Moves between the two positions specified below",
                                      )
            case 5:
                self.configure_widget(lower_lim_name="Lower Limit (Force A-X positive)",
                                      lower_lim_tooltip="Lower limit of the protocol, in nm. \n NOTE: The lower limit must be smaller than the upper limit!",
                                      upper_lim_name="Upper Limit (Force A-X positive)",
                                      upper_lim_tooltip="Upper limit of the protocol, in nm. \n NOTE: The upper limit must be larger than the lower limit!")
                
            
            case 21:
                # Constant force with B autoaligned
                self.configure_widget(lower_lim_name="Constant Force X value (pN)",
                                      lower_lim_tooltip="Value of the force along the X axis which the protocol will try to maintain. \n NOTE: The value is in pN and must be positive!",
                                      lower_lim_range=(-120, 120),
                                      upper_lim_name="Constant Force Y value (pN)",
                                      upper_lim_tooltip="Constant force along the Y axis which the protocol will try to maintain. \n NOTE: The value is in pN and must be positive!",
                                      upper_lim_range=(-120, 120),    
                                      lower_lim_converter=self.calcConstForceAX,
                                      upper_lim_converter=self.calcConstForceAY)
            case 22:
                # Constant force with A autoaligned
                self.configure_widget(lower_lim_name="Constant Force X value (pN)",
                                      lower_lim_tooltip="Value of the force along the X axis which the protocol will try to maintain. \n NOTE: The value is in pN and must be positive!",
                                      lower_lim_range=(-120, 120),
                                      upper_lim_name="Constant Force Y value (pN)",
                                      upper_lim_tooltip="Constant force along the Y axis which the protocol will try to maintain. \n NOTE: The value is in pN and must be positive!",
                                      upper_lim_range=(-120, 120),    
                                      lower_lim_converter=self.calcConstForceBX,
                                      upper_lim_converter=self.calcConstForceBY)
        
        
    def toggleProtocol(self):
        if self.toggleProtocolButton.isChecked():
            self.c_p['protocol_data'][0] = self.protocol_axis_index
            print("Setting button to toggled")
        else:
            self.c_p['protocol_data'][0] = 0
       
