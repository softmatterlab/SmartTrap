import sys
from email.header import UTF8

sys.path.append("C:/Users/Martin/Downloads/ESI_V3_08_02/ESI_V3_08_02/ESI_V3_08_02/SDK_V3_08_02/DLL/DLL64")#add the path of the library here
sys.path.append("C:/Users/Martin/Downloads/ESI_V3_08_02/ESI_V3_08_02/ESI_V3_08_02/SDK_V3_08_02/DLL/Python/Python_64")#add the path of the LoadElveflow.py

"""
Note there is a line which needs to be changed in the Elveflow64.py file to get the correct import:
Replaced: ElveflowDLL=CDLL('D:/dev/SDK/DLL64/DLL64/Elveflow64.dll')# change this path 
With: ElveflowDLL=CDLL("C:/Users/Martin/Downloads/ESI_V3_08_02/ESI_V3_08_02/ESI_V3_08_02/SDK_V3_08_02/DLL/DLL64/Elveflow64.dll")# change this path 
Similar corrections may need to be made when installing on a different system.
"""

from array import array
from ctypes import *

import abc

from ctypes import *
from Elveflow64 import *

from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,QHBoxLayout,
    QPushButton, QVBoxLayout, QWidget, QLabel
)
from PyQt6.QtGui import QPalette, QColor

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

import serial
import time
import numpy as np

class MicrofluidicsControllerInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'setPressure') and
                callable(subclass.setPressure) and
                hasattr(subclass, 'setFlowRate') and
                callable(subclass.setFlowRate) and
                hasattr(subclass, 'setValve') and
                callable(subclass.setValve) and
                hasattr(subclass, 'setPump') and
                callable(subclass.setPump) and
                hasattr(subclass, 'getPressure') and
                callable(subclass.getPressure) and
                hasattr(subclass, 'getNumberChannels') and
                callable(subclass.getNumberChannels) and
                hasattr(subclass, 'getValve') and
                callable(subclass.getValve) and
                hasattr(subclass, 'getPump') and
                callable(subclass.getPump) or
                NotImplemented)


class SyringePumpPSU():
    """
    This class is used to control the Tenma 72-2540 PSU that powers the pump which is attached to the pipette.
    The same PSU can also be used for the pipette puller.    
    """
    def __init__(self, com_port, baud_rate=9600):
        self.connected = False
        self.connect_to_psu(com_port, baud_rate)

    def connect_to_psu(self, com_port, baud_rate=9600):
        try:
            # Establish a serial connection to the PSU
            ser = serial.Serial(com_port, baud_rate, timeout=1)
            print(f"Connected to PSU on {com_port} at {baud_rate} baud.")

            # Example command to send (modify as per your PSU's protocol)
            ser.write(b'*IDN?\n') # Example command, replace with your PSU's specific command

            # Wait for response and read it
            time.sleep(1)  # Adjust as necessary
            self.ser = ser

            response = ser.readline().decode().strip()

            print(f"Response from PSU: {response}")
            self.connected = True
            # Close the serial connection
        except Exception as e:
            print(f"Error: {e}")

    def disconnect_from_psu(self):
        self.ser.close()

    # Function to set voltage
    def set_voltage(self, voltage):
        command = f"VSET1:{voltage}\n".encode()  # Adjust the command as per your PSU's protocol
        self.ser.write(command)

    # Function to set current
    def set_current(self,current):
        command = f"ISET1:{current}\n".encode()  # Adjust the command as per your PSU's protocol
        self.ser.write(command)

    def output_on(self):
        command = f"OUT1:\n".encode()  # Adjust the command as per your PSU's protocol
        self.ser.write(command)

    def output_off(self):
        command = f"OUT0:\n".encode()  # Adjust the command as per your PSU's protocol
        self.ser.write(command)

    def read_voltage(self):
        command = f"VOUT1?\n".encode()  # Adjust the command as per your PSU's protocol
        self.ser.write(command)
        voltage = self.ser.readline().decode().strip()
        return voltage

    def read_current(self):
        command = f"IOUT1?\n".encode()  # Adjust the command as per your PSU's protocol
        self.ser.write(command)
        current = self.ser.readline().decode().strip()
        return current

    def get_status(self):
        from time import sleep
        command = f"STATUS?\n".encode()  # Adjust the command as per your PSU's protocol
        self.ser.write(command)
        sleep(0.01)
        status = self.ser.readline().decode().strip()
        if len(status)>0:
            print("status is ", ord(status))
            sixth_bit =  ord(status) & 64
            print("sixth bit is ", sixth_bit)
        return status

class MUXWireValveController():

    def __init__(self):
        self.valve_connected = False
        self.Instr_ID=c_int32()
        self.states = np.zeros(16, dtype=int)

    def connect(self, adress="COM3"):
        # print("Instrument name is hardcoded in the Python script")
        #see User Guide and NIMAX to determine the instrument name 
        error = MUX_Initialization(adress.encode('ascii'),byref(self.Instr_ID))
        if error==0:
            self.valve_connected = True
        return error
    
    def toggle_valve(self, valve_index, open):
        """
        Toggle a valve open or closed. The valve index is the index of the valve in the list of valves. Goes from 0 to 7.
        open is a boolean value which indicates if the valve should be open or closed.
        """
        if not self.valve_connected:
            return
        if open:
            self.states[valve_index] = 1
        else:
            self.states[valve_index] = 0
        self.set_valve_states()

    def get_valve_states(self):
        return self.states[:8]

    def set_valve_states(self):
        valve_state=(c_int32*16)(0)
        for i in range (0 ,16):
            valve_state[i]=c_int32(self.states[i])        
        error=MUX_Set_all_valves(self.Instr_ID.value, valve_state, 16)

class ElvesysMicrofluidicsController(MicrofluidicsControllerInterface):
    def __init__(self):
        super().__init__()
        self.nbr_channels = 3
        self.Calib = (c_double*1000)()

    def connect(self, adress):
        self.Instr_ID=c_int32()
        #print("Instrument name and regulator types are hardcoded in the Python script")
        #see User Guide to determine regulator types and NIMAX to determine the instrument name 
        error = OB1_Initialization(adress.encode('ascii'),0,0,0,0,byref(self.Instr_ID)) # Seems to be either 3 or 6 when looking in NI-MAX
        #all functions will return error codes to help you to debug your code, for further information refer to User Guide
        print('error:%d' % error)
        print("OB1 ID: %d" % self.Instr_ID.value)

    def disconnect(self):
        error = OB1_Close(self.Instr_ID.value)
        return self.check_error(error)

    def setPressure(self, channel, pressure):
        set_channel = int(channel)#convert to int
        set_channel = c_int32(set_channel)#convert to c_int32

        #Pressure
        set_target=float(pressure) 
        set_target=c_double(set_target)#convert to c_double

        error = OB1_Set_Press(self.Instr_ID.value, set_channel, set_target,  byref(self.Calib), 1000)
        return self.check_error(error)

    def check_error(self, error):
        return error == 0
    
    def getNumberChannels(self):
        return self.nbr_channels

    def get_pressure(self, channel):
        """
        Get the pressure of a channel. Automatically updates the control parameters.
        """
        set_channel = int(channel)#convert to int
        set_channel = c_int32(set_channel)#convert to c_int32
        get_pressure = c_double()
        error = OB1_Get_Press(self.Instr_ID.value, set_channel, 1, byref(self.Calib),byref(get_pressure), 1000)#Acquire_data=1 -> read all the analog values
        if self.check_error(error):
            
            return get_pressure.value
        return None


class MicrofluidicsMonitorThread(QThread):
    # Define signals to communicate with the main thread
    finished = pyqtSignal()
    progress = pyqtSignal(list)

    def __init__(self, controller, c_p, pump_psu=None):
        super().__init__()
        self.controller = controller
        self.c_p = c_p
        self.pump_psu = pump_psu
        self.c_p['valve_controller'] = MUXWireValveController()
        self.c_p['valve_controller'].connect(adress=self.c_p['valve_adress'])

    def set_pressures(self):

        for channel in range(self.controller.getNumberChannels()):
            # Indexing starts at 1 in the controller. Also 0 and 1 map to the same channel.
            self.controller.setPressure(channel+1, self.c_p['target_pressures'][channel])

    def get_pressures(self):
        for channel in range(self.controller.getNumberChannels()):
            self.c_p['current_pressures'][channel] = self.controller.get_pressure(channel+1)
    
    def check_pump_psu(self):
        self.pump_psu.set_voltage(self.c_p['pump_PSU_max_voltage'])
        self.pump_psu.set_current(self.c_p['pump_PSU_max_current'])
        if self.c_p['pump_PSU_on']:
            self.pump_psu.output_on()
        else:
            self.pump_psu.output_off()
        self.c_p['pump_PSU_current_voltage'] = self.pump_psu.read_voltage()
        self.c_p['pump_PSU_current_current'] = self.pump_psu.read_current()

    def run(self):
        # Place your background task here
        while self.c_p['program_running']:
            self.set_pressures()
            self.get_pressures()

            # Set the valves to the correct state
            self.c_p['valves_controller_connected'] = self.c_p['valve_controller'].valve_connected
            if self.c_p['valves_controller_connected']:
                for index in self.c_p['valves_used']:
                    self.c_p['valve_controller'].toggle_valve(index, self.c_p['valves_open'][index])
                self.c_p['valve_controller'].set_valve_states()
            
            if self.pump_psu is not None and self.pump_psu.connected:
                self.check_pump_psu()
            self.progress.emit(self.c_p['current_pressures'])
            QThread.msleep(500) # Sleep for specified number of milliseconds
        self.finished.emit()


class ConfigurePumpWidget(QWidget):
    """
    Widget used to change the settings of the fluidics channels. Used to 
    tell the autonomous system which channels contain which particles and what
    a reasonable pressure is to use when flowing them trough the capillaries.
    """

    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        self.setAutoFillBackground(True)
        pal = self.palette()
        # QPalette.ColorRole.Window is the “background” role in Qt6
        pal.setColor(QPalette.ColorRole.Window, QColor(225, 225, 250))#("#f0e68c"))
        self.setPalette(pal)

        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setWindowTitle("Configure Pump")
        self.capillary_1_label = QLabel("Capillary 1")
        self.layout.addWidget(self.capillary_1_label)
        self.capillary_1_channel_spinbox = QSpinBox()
        self.capillary_1_channel_spinbox.setRange(1, 3)
        self.capillary_1_channel_spinbox.setValue(self.c_p['capillary_1_fluidics_channel'][0])
        self.capillary_1_channel_spinbox.valueChanged.connect(lambda value: self.set_capillary_1_channel(value))
        self.layout.addWidget(self.capillary_1_channel_spinbox)

        self.capillary_1_max_pressure_input = QDoubleSpinBox()
        self.capillary_1_max_pressure_input.setRange(0, 2000)
        self.capillary_1_max_pressure_input.setValue(self.c_p['capillary_1_fluidics_channel'][1])
        self.capillary_1_max_pressure_input.valueChanged.connect(lambda value: self.set_capillary_1_flow_pressure(value))
        self.layout.addWidget(self.capillary_1_max_pressure_input)

        self.capillary_1_valve_label = QLabel("Capillary 1 Valve")
        self.layout.addWidget(self.capillary_1_valve_label)
        self.capillary_1_valve_input = QSpinBox()
        self.capillary_1_valve_input.setRange(0, 7)
        self.capillary_1_valve_input.setValue(self.c_p['capillary_1_fluidics_channel'][2])
        self.capillary_1_valve_input.valueChanged.connect(lambda value: self.set_capillary_1_valve(value))
        self.layout.addWidget(self.capillary_1_valve_input)

        self.capillary_2_label = QLabel("Capillary 2")
        self.layout.addWidget(self.capillary_2_label)
        self.capillary_2_channel_spinbox = QSpinBox()
        self.capillary_2_channel_spinbox.setRange(1, 3)
        self.capillary_2_channel_spinbox.setValue(self.c_p['capillary_2_fluidics_channel'][0])
        self.capillary_2_channel_spinbox.valueChanged.connect(lambda value: self.set_capillary_2_channel(value))
        self.layout.addWidget(self.capillary_2_channel_spinbox)

        self.capillary_2_max_pressure_input = QDoubleSpinBox()
        self.capillary_2_max_pressure_input.setRange(0, 2000)
        self.capillary_2_max_pressure_input.setValue(self.c_p['capillary_2_fluidics_channel'][1])
        self.capillary_2_max_pressure_input.valueChanged.connect(lambda value: self.set_capillary_2_flow_pressure(value))
        self.layout.addWidget(self.capillary_2_max_pressure_input)

        self.capillary_2_valve_label = QLabel("Capillary 2 Valve")
        self.layout.addWidget(self.capillary_2_valve_label)
        self.capillary_2_valve_input = QSpinBox()
        self.capillary_2_valve_input.setRange(0, 7)
        self.capillary_2_valve_input.setValue(self.c_p['capillary_2_fluidics_channel'][2])
        self.capillary_2_valve_input.valueChanged.connect(lambda value: self.set_capillary_2_valve(value))
        self.layout.addWidget(self.capillary_2_valve_input)

        self.main_label = QLabel("Central Channel")
        self.layout.addWidget(self.main_label)
        self.main_channel_spinbox = QSpinBox()
        self.main_channel_spinbox.setRange(1, 3)
        self.main_channel_spinbox.setValue(self.c_p['central_fluidics_channel'][0])
        self.main_channel_spinbox.valueChanged.connect(lambda value: self.set_main_channel(value))
        self.layout.addWidget(self.main_channel_spinbox)

        self.main_max_pressure_input = QDoubleSpinBox()
        self.main_max_pressure_input.setRange(0, 2000)
        self.main_max_pressure_input.setValue(self.c_p['central_fluidics_channel'][1])
        self.main_max_pressure_input.valueChanged.connect(lambda value: self.set_main_flow_pressure(value))
        self.layout.addWidget(self.main_max_pressure_input)

        self.central_valve_label = QLabel("Central Valve")
        self.layout.addWidget(self.central_valve_label)
        self.main_valve_input = QSpinBox()
        self.main_valve_input.setRange(0, 7)
        self.main_valve_input.setValue(self.c_p['central_fluidics_channel'][2])
        self.main_valve_input.valueChanged.connect(lambda value: self.set_main_valve(value))
        self.layout.addWidget(self.main_valve_input)
                
        self.setLayout(self.layout)
        self.show()

    def set_capillary_1_channel(self, channel):
        self.c_p['capillary_1_fluidics_channel'][0] = int(channel-1)
    def set_capillary_2_channel(self, channel):
        self.c_p['capillary_2_fluidics_channel'][0] = int(channel-1)
    def set_main_channel(self, channel):
        self.c_p['central_fluidics_channel'][0] = int(channel-1)

    def set_capillary_1_flow_pressure(self, pressure):
        self.c_p['capillary_1_fluidics_channel'][1] = float(pressure)
    def set_capillary_2_flow_pressure(self, pressure):
        self.c_p['capillary_2_fluidics_channel'][1] = float(pressure)
    def set_main_flow_pressure(self, pressure):
        self.c_p['central_fluidics_channel'][1] = float(pressure)

    def set_capillary_1_valve(self, valve):
        self.c_p['capillary_1_fluidics_channel'][2] = int(valve)
    def set_capillary_2_valve(self, valve):
        self.c_p['capillary_2_fluidics_channel'][2] = int(valve)
    def set_main_valve(self, valve):
        self.c_p['central_fluidics_channel'][2] = int(valve)
    
    

class MicrofluidicsControllerWidget(QWidget):
    """
    A widget for controlling the microfluidics system. Will automatically
    create buttons to control each of the channels in the system.
    """

    def __init__(self, c_p, controller=None):
        super().__init__()
        self.c_p = c_p
        self.controller = controller

        self.setAutoFillBackground(True)
        pal = self.palette()
        # QPalette.ColorRole.Window is the “background” role in Qt6
        pal.setColor(QPalette.ColorRole.Window, QColor(225, 225, 250))#("#f0e68c"))
        self.setPalette(pal)


        self.initUI()
        self.pump_PSU = SyringePumpPSU(self.c_p['pump_PSU_adress'])
        self.pumpMonitorThread = MicrofluidicsMonitorThread(self.controller, self.c_p, self.pump_PSU)
        self.pumpMonitorThread.progress.connect(self.updatePressures)
        self.pumpMonitorThread.start()

        self.update_timer = QTimer()
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self.refresh)
        self.update_timer.start()
        print("Pump monitor started")

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setWindowTitle("Microfluidics Controller")
        self.create_channel_controls()

        # Create button for calibrating the pump
        # Also create button for connecting the pump and potentially also for disconnecting it
        # Eventually we will also need to add the valves here.

        # buttons for controlling the pump PSU
        self.pump_PSU_max_voltage_spinbox = QDoubleSpinBox()
        self.pump_PSU_max_voltage_spinbox.setRange(0, 12)
        self.pump_PSU_max_voltage_spinbox.setSingleStep(0.1)
        self.pump_PSU_max_voltage_spinbox.setSuffix(" V")
        self.pump_PSU_max_voltage_spinbox.setValue(self.c_p['pump_PSU_max_voltage'])
        self.pump_PSU_max_voltage_spinbox.valueChanged.connect(lambda value: self.set_pump_PSU_max_voltage(value))
        self.layout.addWidget(self.pump_PSU_max_voltage_spinbox)

        self.toggle_pump_PSU_button = QPushButton("Toggle Pump PSU") # TODO have this react to external events
        self.toggle_pump_PSU_button.clicked.connect(lambda: self.toggle_pump_PSU())
        self.toggle_pump_PSU_button.setCheckable(True)
        self.toggle_pump_PSU_button.setChecked(self.c_p['pump_PSU_on'])
        self.layout.addWidget(self.toggle_pump_PSU_button)

        self.layout.addWidget(QLabel("Valves"))
        self.valve_buttons = []
        for valve_index in self.c_p['valves_used']:
            self.valve_buttons.append(QPushButton(f"Valve {valve_index}"))
            self.valve_buttons[-1].setCheckable(True)
            self.valve_buttons[-1].setChecked(self.c_p['valves_open'][valve_index])
            self.valve_buttons[-1].clicked.connect(lambda checked, valve_index=valve_index: self.toggle_valve(valve_index))
            self.valve_buttons[-1].setStyleSheet("""
                QPushButton {
                    background-color: red;
                }
                QPushButton:checked {
                    background-color: green;
                    color: white;
                }
                """)
            self.layout.addWidget(self.valve_buttons[-1])

        self.setLayout(self.layout)
        self.show()
    
    def toggle_valve(self, valve_index):
        self.c_p['valves_open'][valve_index] = not self.c_p['valves_open'][valve_index]

    def create_channel_controls(self):
        """
        Crates the UI elments needed to control the channels of the pump
        """
        self.pressure_spinboxes = []
        self.pressure_monitor_labels = []

        for channel in range(self.controller.getNumberChannels()):
            # Create a label for the channel
            label = QLabel("Channel " + str(channel+1))
            self.layout.addWidget(label)

            # Create a spinbox for setting the pressure
            self.pressure_spinboxes.append(QDoubleSpinBox())
            self.pressure_spinboxes[-1].setRange(0, 2000)
            self.pressure_spinboxes[-1].setSingleStep(0.1)
            self.pressure_spinboxes[-1].setSuffix(" mbar")
            self.pressure_spinboxes[-1].valueChanged.connect(lambda value, channel=channel: self.setPressure(channel, value))
            self.layout.addWidget(self.pressure_spinboxes[-1])

            # Create a label for monitoring the pressure
            self.pressure_monitor_labels.append(QLabel(f"Pressure {self.c_p['current_pressures'][channel]} mbar"))
            self.layout.addWidget(self.pressure_monitor_labels[-1])

    def updatePressures(self, values):
        for channel in range(self.controller.getNumberChannels()):
            self.pressure_monitor_labels[channel].setText(f"Pressure {self.c_p['current_pressures'][channel]} mbar")
            self.pressure_spinboxes[channel].setValue(self.c_p['target_pressures'][channel])
    def setPressure(self, channel, pressure):
        self.c_p['target_pressures'][channel] = float(pressure)

    def set_pump_PSU_max_voltage(self, value):
        self.c_p['pump_PSU_max_voltage'] = value

    def toggle_pump_PSU(self):
        self.c_p['pump_PSU_on'] = self.toggle_pump_PSU_button.isChecked()

    def refresh(self):
        '''
        for channel in range(self.controller.getNumberChannels()):
            self.pressure_monitor_labels[channel].setText(f"Pressure {self.c_p['current_pressures'][channel]} mbar")
            self.pressure_spinboxes[channel].setValue(self.c_p['target_pressures'][channel])
        '''
        self.pump_PSU_max_voltage_spinbox.setValue(self.c_p['pump_PSU_max_voltage'])
        self.toggle_pump_PSU_button.setChecked(self.c_p['pump_PSU_on'])

        for button, index in zip(self.valve_buttons, self.c_p['valves_used']):
            button.setChecked(self.c_p['valves_open'][index])

    def closeEvent(self, event):
        event.accept()
        self.pump_PSU.disconnect_from_psu()
            
