from PyQt6.QtWidgets import (
 QVBoxLayout, QWidget, QLabel, QPushButton, QSlider, QFormLayout,QLineEdit, QHBoxLayout, QSpinBox,
 QToolBar
)
from PyQt6.QtCore import Qt

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIntValidator
from PyQt6.QtCore import QTimer, QTime
from threading import Thread
import numpy as np
import serial
from time import sleep, time
from functools import partial

class LaserControllerThread(Thread):
    def __init__(self, c_p):
        Thread.__init__(self)
        self.setDaemon(True)
        self.c_p = c_p

# Laser powers to be used in the RBC experiment

# EG currents not powers
laser_powers = [ 
    [86,84,10],
    [98,96,5],
    [86,84,10],
    [116,114,5],
    [86,84,10],
    [155,148,5],
    [86,84,10],
    [195,180,5], # 60
    [86,84,10],
    [237,217,5], # 80
    [86,84,20],
    #[279,250], # 100 mW often damages the cells   
]
"""
laser_powers = [ 
    [86,84,10],
    [237,217,5], # 80
    [86,84,10],   
]
"""
class LaserControllerWidget(QWidget):

    def __init__(self, c_p, OT_GUI):
        super().__init__()
        self.c_p = c_p
        self.OT_GUI = OT_GUI
        self.setWindowTitle("Laser controller")
        
        self.current_A_edit_val = self.c_p['laser_A_current']
        self.current_B_edit_val = self.c_p['laser_B_current']
        self.laser_A_ser = None
        self.laser_B_ser = None
        self.connect_laser('A')
        self.connect_laser('B')
        #self.init_laser_A()
        self.laser_A_on = False
        self.laser_B_on = False

        # create a timer
        self.timer = QTimer(self)

        # set timer timeout callback function
        self.timer.timeout.connect(self.RBC_auto_experiment)

        # set the timer to fire every 100 milliseconds (1 second)
        self.timer.start(100)
        self.experiment_start_time = 0
        self.current_power_start_time = 0
        self.experiment_idx = 0
        self.experiment_running = False
        self.experiment_started = False
        self.snapshot_taken = False
        self.time_interval = 10 # Decrease?


        self.initUI()

    def init_laser_A(self):
        if self.laser_A_ser is None:
            return
        self.set_laser_A_current()
        # Turn on temperature controller
        message = "TCR\r"
        self.laser_A_ser.write(message.encode('utf-8'))

        # TODO query if the laser is on or off

    def get_name(self, idx):
        self.c_p['filename'] = 'RBC_experiment_no-'+str(self.experiment_idx)+ '_A' + str(laser_powers[idx][0]) + '-B' + str(laser_powers[idx][1])
        return

    def start_data_recording(self):
        self.OT_GUI.start_saving()
        if self.c_p['recording']:
            self.OT_GUI.ToggleRecording()
            print("Warning recording was already on, turning off")
            time.sleep(0.05)
        self.OT_GUI.ToggleRecording()
        self.snapshot_taken = False
        # self.OT_GUI.snapshot()

    def stop_data_recording(self):
        self.OT_GUI.stop_saving()
        if not self.c_p['recording']:
            print("Warning recording was turned off")
            return
        self.OT_GUI.ToggleRecording()

    def set_currents_for_RBC(self):
        self.current_A_edit_val = int(laser_powers[self.experiment_idx][0])
        self.current_B_edit_val = int(laser_powers[self.experiment_idx][1])
        self.time_interval = int(laser_powers[self.experiment_idx][2])

        self.laserA_CurrentEdit.setValue(int(self.current_A_edit_val))
        self.laserB_CurrentEdit.setValue(int(self.current_B_edit_val))
        self.set_laser_A_current()
        self.set_laser_B_current()
        sleep(0.05)

    def RBC_auto_experiment(self):
        if not self.experiment_running:
            if self.experiment_started:
                self.stop_data_recording()
                self.toggle_experiment_button.setChecked(False)
            self.experiment_start_time = 0
            self.current_power_start_time = 0
            self.experiment_idx = 0
            self.experiment_started = False
            return

        if self.experiment_idx == 0 and not self.experiment_started:
            self.experiment_started = True
            self.experiment_start_time = time()
            self.current_power_start_time = time()
            self.set_currents_for_RBC()
            self.get_name(self.experiment_idx)
            self.start_data_recording()
            
        if self.c_p['program_running'] and self.experiment_running:
            dt = time()-self.current_power_start_time

            # In the middle of the experiment, take a snapshot
            if dt > self.time_interval/2 and not self.snapshot_taken: 
                self.OT_GUI.snapshot()
                self.snapshot_taken = True

            if dt > self.time_interval:
                print(f"Stopped recording data{dt}\n {self.experiment_idx}")
                self.stop_data_recording()
                self.current_power_start_time = time()
                self.experiment_idx += 1
                if self.experiment_idx >= len(laser_powers):
                    print("Experiment done")
                    self.experiment_running = False
                    self.experiment_started = False
                    self.toggle_experiment_button.setChecked(False)
                    return
                self.set_currents_for_RBC()
                self.get_name(self.experiment_idx)
                self.start_data_recording()

    def initUI(self):
        layout = QVBoxLayout()

        # Laser A
        self.laserA_label = QLabel("Laser A", self)
        layout.addWidget(self.laserA_label)
        self.laserA_layout = QFormLayout()
        self.connectA_btn = QPushButton("Connect", self)
        connect_laserA = partial(self.connect_laser, 'A')
        self.connectA_btn.clicked.connect(connect_laserA)
        self.laserA_layout.addRow("Connect", self.connectA_btn)
        self.onOffA_btn = QPushButton("Turn ON", self)
        self.onOffA_btn.clicked.connect(self.onOff_laserA)
        self.laserA_layout.addRow("Output", self.onOffA_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentA_layout = QHBoxLayout()

        # Create spinbox for current
        self.laserA_CurrentEdit = QSpinBox()
        self.laserA_CurrentEdit.setRange(0, 400)
        self.laserA_CurrentEdit.setValue(self.c_p['laser_A_current'])
        self.laserA_CurrentEdit.valueChanged.connect(self.tempCurrentA)
        self.currentA_layout.addWidget(self.laserA_CurrentEdit)

        self.setCurrentButton = QPushButton("Set current", self)
        self.setCurrentButton.clicked.connect(self.set_laser_A_current)
        self.currentA_layout.addWidget(self.setCurrentButton)
        self.laserA_layout.addRow("Set current", self.currentA_layout)
        layout.addLayout(self.laserA_layout)

        self.laserA_PortSelect = QSpinBox()
        self.laserA_PortSelect.setRange(0, 20)  # Assuming you have 10 COM ports; adjust accordingly
        self.laserA_PortSelect.setValue(int(self.c_p['laser_A_port'][3:]))  # Assuming port is in format 'COMx'
        self.laserA_PortSelect.valueChanged.connect(self.set_laser_A_port)
        self.laserA_layout.addRow("COM Port", self.laserA_PortSelect)

        # Laser B
        self.laserB_label = QLabel("Laser B", self)
        layout.addWidget(self.laserB_label)
        self.laserB_layout = QFormLayout()
        self.connectB_btn = QPushButton("Connect", self)
        connect_laserB = partial(self.connect_laser, 'B')

        self.connectB_btn.clicked.connect(connect_laserB)
        self.laserB_layout.addRow("Connect", self.connectB_btn)
        self.onOffB_btn = QPushButton("Turn ON", self)
        self.onOffB_btn.clicked.connect(self.onOff_laserB)
        self.laserB_layout.addRow("Power", self.onOffB_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentB_layout = QHBoxLayout()

        self.laserB_CurrentEdit = QSpinBox()
        self.laserB_CurrentEdit.setRange(0, 400)
        self.laserB_CurrentEdit.setValue(self.c_p['laser_B_current'])
        self.laserB_CurrentEdit.valueChanged.connect(self.tempCurrentB)
        self.currentB_layout.addWidget(self.laserB_CurrentEdit)
        
        self.setCurrentButton = QPushButton("Set current", self)
        self.setCurrentButton.clicked.connect(self.set_laser_B_current)
        self.currentB_layout.addWidget(self.setCurrentButton)
        self.laserB_layout.addRow("Set current", self.currentB_layout)

        self.laserB_PortSelect = QSpinBox()
        self.laserB_PortSelect.setRange(0, 20)  # Assuming you have 10 COM ports; adjust accordingly
        self.laserB_PortSelect.setValue(int(self.c_p['laser_B_port'][3:]))  # Assuming port is in format 'COMx'
        self.laserB_PortSelect.valueChanged.connect(self.set_laser_B_port)
        self.laserB_layout.addRow("COM Port", self.laserB_PortSelect)

        # Common stuff
        self.set_both_currents_button = QPushButton("Set currents", self)
        self.set_both_currents_button.clicked.connect(self.set_both_currents)
        self.laserB_layout.addRow("Set both currents", self.set_both_currents_button)

        self.toggle_experiment_button = QPushButton("Toggle automatic RBC experiment", self)
        self.toggle_experiment_button.clicked.connect(self.toggle_RBC_experiment)
        self.toggle_experiment_button.setCheckable(True)
        self.laserB_layout.addRow("Toggle experiment", self.toggle_experiment_button)

       
        layout.addLayout(self.laserB_layout)

        self.setLayout(layout)

    def set_laser_A_port(self):
        value = self.laserA_PortSelect.value()
        self.c_p['laser_A_port'] = f"COM{value}"
        print(self.c_p['laser_A_port'])

    def set_laser_B_port(self, value):
        self.c_p['laser_B_port'] = f"COM{value}"


    def connect_laser(self, laser):
        if laser == 'A' and self.laser_A_ser is None:
            port = self.c_p['laser_A_port']
            print("Trying to connect laser A")
            try:
                self.laser_A_ser = serial.Serial(port=port,  # Port to connect to
                    baudrate=9600,  # Baud rate
                    bytesize=serial.EIGHTBITS,  # Data bits
                    parity=serial.PARITY_NONE,  # Parity
                    stopbits=serial.STOPBITS_ONE,  # Stop bits
                    timeout=2,  # Read timeout in seconds
                    xonxoff=False,  # Software flow control
                    rtscts=False,  # Hardware flow control (RTS/CTS)
                    write_timeout=2,  # Write timeout, this is not defined in the config but I assumed the same as the read timeout

                )
                print("Connected laser A")
            except Exception as E:
                print(E)
                print("its okay")

        if laser == 'B' and self.laser_B_ser is None:
            port = self.c_p['laser_B_port']
            try:
                self.laser_B_ser = serial.Serial(port=port,  # Port to connect to
                    baudrate=9600,  # Baud rate
                    bytesize=serial.EIGHTBITS,  # Data bits
                    parity=serial.PARITY_NONE,  # Parity
                    stopbits=serial.STOPBITS_ONE,  # Stop bits
                    timeout=2,  # Read timeout in seconds
                    xonxoff=False,  # Software flow control
                    rtscts=False,  # Hardware flow control (RTS/CTS)
                    write_timeout=2,  # Write timeout, this is not defined in the config but I assumed the same as the read timeout
                )
            except Exception as E:
                print(E)

    def tempCurrentA(self, current):
        self.current_A_edit_val = int(current)
    
    def tempCurrentB(self, current):
        self.current_B_edit_val = int(current)

    def toggle_RBC_experiment(self):
        self.experiment_running = not self.experiment_running

    def set_both_currents(self):
        self.set_laser_A_current()
        self.set_laser_B_current()

    def set_laser_A_current(self):
        # current = int(self.current_A_edit_val) # TODO get the value directly from the edit box 
        current = self.laserA_CurrentEdit.value()

        if int(current)>400 or int(current)<0:
            return
        self.c_p['laser_A_current'] = current
        if self.laser_A_ser is not None:
            message = "LCT" + str(self.current_A_edit_val) + "\r"
            self.laser_A_ser.write(message.encode('utf-8'))

    def set_laser_B_current(self):
        # current = int(self.current_B_edit_val)
        current = self.laserB_CurrentEdit.value()

        if int(current)>400 or int(current)<0:
            return
        self.c_p['laser_B_current'] = current
        if self.laser_B_ser is not None:
            message = "LCT" + str(self.current_B_edit_val) + "\r"
            self.laser_B_ser.write(message.encode('utf-8'))

    def set_laser_currrent(self, laser, current):
        if int(current)>400 or int(current)<0:
            return
        message = "LCT"+str(current)+"\r"
        if laser == 'A' and self.laser_A_ser is not None:
            message = "LCT" + str(self.current_A_edit_val) + "\r"
            self.laser_A_ser.write(message.encode('utf-8'))
        elif laser == 'B' and self.laser_B_ser is not None:
            self.laser_B_ser.write(message.encode('utf-8'))

    def onOff_laserA(self):
        if self.laser_A_on:
            self.laser_A_on = False
            self.onOffA_btn.setText("Turn ON")
            self.laser_A_ser.write(b"LS\r")
            self.c_p['laser_A_on'] = False
        else:
            self.laser_A_on = True
            self.onOffA_btn.setText("Turn OFF")
            self.laser_A_ser.write(b"LR\r")
            self.c_p['laser_A_on'] = True


    def onOff_laserB(self):
        if self.laser_B_on:
            self.laser_B_on = False
            self.onOffB_btn.setText("Turn ON")
            self.laser_B_ser.write(b"LS\r")
            self.c_p['laser_B_on'] = False
        else:
            self.laser_B_on = True
            self.onOffB_btn.setText("Turn OFF")
            self.laser_B_ser.write(b"LR\r")
            self.c_p['laser_B_on'] = True

    def disconnect_laser(self, laser):
        if laser == 'A' and self.laser_A_ser is not None:
            self.laser_A_ser.close()    
            self.laser_A_ser = None
            print("Connection to laser A closed")
        elif laser == 'B' and self.laser_B_ser is not None:
            self.laser_B_ser.close()
            self.laser_B_ser = None
            print("Connection to laser B closed")

    def closeEvent(self, event):
        self.disconnect_laser('A')
        self.disconnect_laser('B')
        event.accept()


class LaserControllerToolbar(QWidget):

    def __init__(self, c_p, OT_GUI):
        super().__init__()
        self.c_p = c_p
        self.OT_GUI = OT_GUI
        self.setWindowTitle("Laser controller")
        
        self.current_A_edit_val = self.c_p['laser_A_current']
        self.current_B_edit_val = self.c_p['laser_B_current']
        self.laser_A_ser = None
        self.laser_B_ser = None
        self.connect_laser('A')
        self.connect_laser('B')
        #self.init_laser_A()
        self.laser_A_on = False
        self.laser_B_on = False

        # create a timer
        self.timer = QTimer(self)

        # set timer timeout callback function
        self.timer.timeout.connect(self.RBC_auto_experiment)

        # set the timer to fire every 100 milliseconds (1 second)
        self.timer.start(100)
        self.experiment_start_time = 0
        self.current_power_start_time = 0
        self.experiment_idx = 0
        self.experiment_running = False
        self.experiment_started = False
        self.snapshot_taken = False
        self.time_interval = 10


        self.initUI()

    def init_laser_A(self):
        if self.laser_A_ser is None:
            return
        self.set_laser_A_current()
        # Turn on temperature controller
        message = "TCR\r"
        self.laser_A_ser.write(message.encode('utf-8'))

        # TODO query if the laser is on or off

    def get_name(self, idx):
        self.c_p['filename'] = 'RBC_experiment_no-'+str(self.experiment_idx)+ '_A' + str(laser_powers[idx][0]) + '-B' + str(laser_powers[idx][1])
        return

    def start_data_recording(self):
        self.OT_GUI.start_saving()
        if self.c_p['recording']:
            self.OT_GUI.ToggleRecording()
            print("Warning recording was already on, turning off")
            time.sleep(0.05)
        self.OT_GUI.ToggleRecording()
        self.snapshot_taken = False
        # self.OT_GUI.snapshot()

    def stop_data_recording(self):
        self.OT_GUI.stop_saving()
        if not self.c_p['recording']:
            print("Warning recording was turned off")
            return
        self.OT_GUI.ToggleRecording()

    def set_currents_for_RBC(self):
        self.current_A_edit_val = int(laser_powers[self.experiment_idx][0])
        self.current_B_edit_val = int(laser_powers[self.experiment_idx][1])
        self.laserA_CurrentEdit.setValue(int(self.current_A_edit_val))
        self.laserB_CurrentEdit.setValue(int(self.current_B_edit_val))
        self.set_laser_A_current()
        self.set_laser_B_current()
        sleep(0.05)

    def RBC_auto_experiment(self):
        #print(f'Timer triggered as it should, time is {round(time()-self.experiment_start_time,1)}')
        if not self.experiment_running:
            if self.experiment_started:
                self.stop_data_recording()
                self.toggle_experiment_button.setChecked(False)
            self.experiment_start_time = 0
            self.current_power_start_time = 0
            self.experiment_idx = 0
            self.experiment_started = False
            return

        if self.experiment_idx == 0 and not self.experiment_started:
            self.experiment_started = True
            self.experiment_start_time = time()
            self.current_power_start_time = time()
            self.set_currents_for_RBC()
            self.get_name(self.experiment_idx)
            self.start_data_recording()
            
        if self.c_p['program_running'] and self.experiment_running:
            dt = time()-self.current_power_start_time

            # In the middle of the experiment, take a snapshot
            if dt > self.time_interval/2 and not self.snapshot_taken: 
                self.OT_GUI.snapshot()
                self.snapshot_taken = True

            if dt > self.time_interval:
                print(f"Stopped recording data {dt}\n {self.experiment_idx} powers {laser_powers[self.experiment_idx]}")
                self.stop_data_recording()
                self.current_power_start_time = time()
                self.experiment_idx += 1
                if self.experiment_idx >= len(laser_powers):
                    print("Experiment done")
                    self.experiment_running = False
                    self.experiment_started = False
                    self.toggle_experiment_button.setChecked(False)
                    return
                self.set_currents_for_RBC()
                self.get_name(self.experiment_idx)
                self.start_data_recording()

    def initUI(self):
        layout = QVBoxLayout()

        # Laser A
        self.laserA_label = QLabel("Laser A", self)
        layout.addWidget(self.laserA_label)
        self.laserA_layout = QFormLayout()
        self.connectA_btn = QPushButton("Connect", self)
        connect_laserA = partial(self.connect_laser, 'A')
        self.connectA_btn.clicked.connect(connect_laserA)
        self.laserA_layout.addRow("Connect", self.connectA_btn)
        self.onOffA_btn = QPushButton("Turn ON", self)
        self.onOffA_btn.clicked.connect(self.onOff_laserA)
        self.laserA_layout.addRow("Output", self.onOffA_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentA_layout = QHBoxLayout()

        # Create spinbox for current
        self.laserA_CurrentEdit = QSpinBox()
        self.laserA_CurrentEdit.setRange(0, 400)
        self.laserA_CurrentEdit.setValue(self.c_p['laser_A_current'])
        self.laserA_CurrentEdit.valueChanged.connect(self.tempCurrentA)

        self.currentA_layout.addWidget(self.laserA_CurrentEdit)

        self.setCurrentButton = QPushButton("Set current", self)
        self.setCurrentButton.clicked.connect(self.set_laser_A_current)
        self.currentA_layout.addWidget(self.setCurrentButton)
        self.laserA_layout.addRow("Set current", self.currentA_layout)
        layout.addLayout(self.laserA_layout)

        self.laserA_PortSelect = QSpinBox()
        self.laserA_PortSelect.setRange(0, 20)  # Assuming you have 10 COM ports; adjust accordingly
        self.laserA_PortSelect.setValue(int(self.c_p['laser_A_port'][3:]))  # Assuming port is in format 'COMx'
        self.laserA_PortSelect.valueChanged.connect(self.set_laser_A_port)
        self.laserA_layout.addRow("COM Port", self.laserA_PortSelect)

        # Laser B
        self.laserB_label = QLabel("Laser B", self)
        layout.addWidget(self.laserB_label)
        self.laserB_layout = QFormLayout()
        self.connectB_btn = QPushButton("Connect", self)
        connect_laserB = partial(self.connect_laser, 'B')

        self.connectB_btn.clicked.connect(connect_laserB)
        self.laserB_layout.addRow("Connect", self.connectB_btn)
        self.onOffB_btn = QPushButton("Turn ON", self)
        self.onOffB_btn.clicked.connect(self.onOff_laserB)
        self.laserB_layout.addRow("Power", self.onOffB_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentB_layout = QHBoxLayout()

        self.laserB_CurrentEdit = QSpinBox()
        self.laserB_CurrentEdit.setRange(0, 400)
        self.laserB_CurrentEdit.setValue(self.c_p['laser_B_current'])
        self.laserB_CurrentEdit.valueChanged.connect(self.tempCurrentB)

        self.currentB_layout.addWidget(self.laserB_CurrentEdit)
        
        self.setCurrentButton = QPushButton("Set current", self)
        self.setCurrentButton.clicked.connect(self.set_laser_B_current)
        self.currentB_layout.addWidget(self.setCurrentButton)
        self.laserB_layout.addRow("Set current", self.currentB_layout)

        self.laserB_PortSelect = QSpinBox()
        self.laserB_PortSelect.setRange(0, 20)  # Assuming you have 10 COM ports; adjust accordingly
        self.laserB_PortSelect.setValue(int(self.c_p['laser_B_port'][3:]))  # Assuming port is in format 'COMx'
        self.laserB_PortSelect.valueChanged.connect(self.set_laser_B_port)
        self.laserB_layout.addRow("COM Port", self.laserB_PortSelect)

        # Common stuff
        self.set_both_currents_button = QPushButton("Set currents", self)
        self.set_both_currents_button.clicked.connect(self.set_both_currents)
        self.laserB_layout.addRow("Set both currents", self.set_both_currents_button)

        self.toggle_experiment_button = QPushButton("Toggle automatic RBC experiment", self)
        self.toggle_experiment_button.clicked.connect(self.toggle_RBC_experiment)
        self.toggle_experiment_button.setCheckable(True)
        self.laserB_layout.addRow("Toggle experiment", self.toggle_experiment_button)

       
        layout.addLayout(self.laserB_layout)

        self.setLayout(layout)

    def set_laser_A_port(self):
        value = self.laserA_PortSelect.value()
        self.c_p['laser_A_port'] = f"COM{value}"
        print(self.c_p['laser_A_port'])

    def set_laser_B_port(self, value):
        self.c_p['laser_B_port'] = f"COM{value}"


    def connect_laser(self, laser):
        if laser == 'A' and self.laser_A_ser is None:
            port = self.c_p['laser_A_port']
            print("Trying to connect laser A")
            try:
                self.laser_A_ser = serial.Serial(port=port,  # Port to connect to
                    baudrate=9600,  # Baud rate
                    bytesize=serial.EIGHTBITS,  # Data bits
                    parity=serial.PARITY_NONE,  # Parity
                    stopbits=serial.STOPBITS_ONE,  # Stop bits
                    timeout=2,  # Read timeout in seconds
                    xonxoff=False,  # Software flow control
                    rtscts=False,  # Hardware flow control (RTS/CTS)
                    write_timeout=2,  # Write timeout, this is not defined in the config but I assumed the same as the read timeout

                )
                print("Connected laser A")
            except Exception as E:
                print(E)
                print("its okay")

        if laser == 'B' and self.laser_B_ser is None:
            port = self.c_p['laser_B_port']
            try:
                self.laser_B_ser = serial.Serial(port=port,  # Port to connect to
                    baudrate=9600,  # Baud rate
                    bytesize=serial.EIGHTBITS,  # Data bits
                    parity=serial.PARITY_NONE,  # Parity
                    stopbits=serial.STOPBITS_ONE,  # Stop bits
                    timeout=2,  # Read timeout in seconds
                    xonxoff=False,  # Software flow control
                    rtscts=False,  # Hardware flow control (RTS/CTS)
                    write_timeout=2,  # Write timeout, this is not defined in the config but I assumed the same as the read timeout
                )
            except Exception as E:
                print(E)

    def tempCurrentA(self, current):
        self.current_A_edit_val = int(current)
    
    def tempCurrentB(self, current):
        self.current_B_edit_val = int(current)

    def toggle_RBC_experiment(self):
        self.experiment_running = not self.experiment_running

    def set_both_currents(self):
        self.set_laser_A_current()
        self.set_laser_B_current()

    def set_laser_A_current(self):
        # current = int(self.current_A_edit_val) # TODO get the value directly from the edit box 
        current = self.laserA_CurrentEdit.value()

        if int(current)>400 or int(current)<0:
            return
        self.c_p['laser_A_current'] = current
        if self.laser_A_ser is not None:
            message = "LCT" + str(self.current_A_edit_val) + "\r"
            self.laser_A_ser.write(message.encode('utf-8'))

    def set_laser_B_current(self):
        # current = int(self.current_B_edit_val)
        current = self.laserB_CurrentEdit.value()

        if int(current)>400 or int(current)<0:
            return
        self.c_p['laser_B_current'] = current
        if self.laser_B_ser is not None:
            message = "LCT" + str(self.current_B_edit_val) + "\r"
            self.laser_B_ser.write(message.encode('utf-8'))

    def set_laser_currrent(self, laser, current):
        if int(current)>400 or int(current)<0:
            return
        message = "LCT"+str(current)+"\r"
        if laser == 'A' and self.laser_A_ser is not None:
            message = "LCT" + str(self.current_A_edit_val) + "\r"
            self.laser_A_ser.write(message.encode('utf-8'))
        elif laser == 'B' and self.laser_B_ser is not None:
            self.laser_B_ser.write(message.encode('utf-8'))

    def onOff_laserA(self):
        if self.laser_A_on:
            self.laser_A_on = False
            self.onOffA_btn.setText("Turn ON")
            self.laser_A_ser.write(b"LS\r")
            self.c_p['laser_A_on'] = False
        else:
            self.laser_A_on = True
            self.onOffA_btn.setText("Turn OFF")
            self.laser_A_ser.write(b"LR\r")
            self.c_p['laser_A_on'] = True


    def onOff_laserB(self):
        if self.laser_B_on:
            self.laser_B_on = False
            self.onOffB_btn.setText("Turn ON")
            self.laser_B_ser.write(b"LS\r")
            self.c_p['laser_B_on'] = False
        else:
            self.laser_B_on = True
            self.onOffB_btn.setText("Turn OFF")
            self.laser_B_ser.write(b"LR\r")
            self.c_p['laser_B_on'] = True

    def disconnect_laser(self, laser):
        if laser == 'A' and self.laser_A_ser is not None:
            self.laser_A_ser.close()    
            self.laser_A_ser = None
            print("Connection to laser A closed")
        elif laser == 'B' and self.laser_B_ser is not None:
            self.laser_B_ser.close()
            self.laser_B_ser = None
            print("Connection to laser B closed")

    def closeEvent(self, event):
        self.disconnect_laser('A')
        self.disconnect_laser('B')
        event.accept()


    