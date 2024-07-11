from PyQt6.QtWidgets import QVBoxLayout, QLabel, QSpinBox, QWidget, QApplication, QPushButton, QComboBox

from PyQt6.QtCore import Qt

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIntValidator
from PyQt6.QtCore import QTimer, QTime
from threading import Thread
import numpy as np
import serial
from time import sleep, time
from functools import partial




class PullingProtocolWidget(QWidget):
    def __init__(self, c_p, data_channels):

        super().__init__()
        self.c_p = c_p
        self.setWindowTitle("Pulling Protocol")
        self.initUI()
        self.data_channels = data_channels
        # TODO handle the different piezos and how they are shared in a better way.
        # Maybe have all the controls in the same widget?
        self.protocol_axis_index = 4 # AX=1, AY=2, BX=3, BY=4, BY default
        self.timer = QTimer()
        self.timer.setInterval(500) # sets the delay of the timer and thereby how often it should update.
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.updateParameters()  # Initial update

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel("Protocol controller")
        layout.addWidget(self.label)

        self.lowerLimitSpinBox = QSpinBox()
        self.lowerLimitSpinBox.setRange(0, 65535)
        self.lowerLimitSpinBox.valueChanged.connect(self.updateParameters)
        self.lowerLimitSpinBox.setValue(self.c_p['protocol_data'][3]*256 + self.c_p['protocol_data'][4])
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
        self.axisComboBox.setCurrentIndex(4)
        self.axisComboBox.currentIndexChanged.connect(self.selectProtocolAxis)
        layout.addWidget(QLabel("Select axis:"))
        layout.addWidget(self.axisComboBox)

        self.setLayout(layout)

    def refresh(self):
        lower_lim = self.c_p['protocol_data'][3]*256 + self.c_p['protocol_data'][4]
        upper_lim = self.c_p['protocol_data'][1]*256 + self.c_p['protocol_data'][2]
        step_size = self.c_p['protocol_data'][5]*256 + self.c_p['protocol_data'][6]
        self.lowerLimitSpinBox.setValue(lower_lim)
        self.upperLimitSpinBox.setValue(upper_lim)
        self.stepSizeSpinBox.setValue(step_size)
        self.toggleProtocolButton.setChecked(self.c_p['protocol_data'][0]>0)
        # self.axisComboBox.setCurrentIndex(4) # TODO fix so that also the axis is udpadted correctly.

        #print('Refreshing')

    def getParametersAs8BitArrays(self):
        lower_limit = self.lowerLimitSpinBox.value()
        upper_limit = self.upperLimitSpinBox.value()

        if upper_limit < lower_limit:
            print("Upper limit must be larger than lower limit!")
            lower_limit = upper_limit

        step_size = self.stepSizeSpinBox.value()

        # Function to split a 16-bit number into two 8-bit numbers
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF] 

        return split_16_bit(lower_limit), split_16_bit(upper_limit), split_16_bit(step_size)

    def updateParameters(self):
        lower_limit, upper_limit, step_size = self.getParametersAs8BitArrays()
        self.c_p['protocol_data'][1:3] = upper_limit
        self.c_p['protocol_data'][3:5] = lower_limit
        self.c_p['protocol_data'][5:7] = step_size
        print(f"Updating parameters Lower limit: {lower_limit}, Upper limit: {upper_limit}, Step size: {step_size}")

    def selectProtocolAxis(self, index):
        self.protocol_axis_index = index + 1
        

    def toggleProtocol(self):
        if self.toggleProtocolButton.isChecked():
            self.c_p['protocol_data'][0] = self.protocol_axis_index
            print(f"Command 2 is {self.c_p['portenta_command_2']}")
        else:
            self.c_p['protocol_data'][0] = 0
        
        # If we are turning of the protocol then we should update the values of the piezo sliders.
        """
        if self.c_p['protocol_data'][0] == 0:
            self.c_p['piezo_A'] = np.int32([self.data_channels['dac_ax'].get_data_spaced(1)[0],
                                            self.data_channels['dac_ay'].get_data_spaced(1)[0]])
            self.c_p['piezo_B'] = np.int32([self.data_channels['dac_bx'].get_data_spaced(1)[0],
                                            self.data_channels['dac_by'].get_data_spaced(1)[0]])
        """
