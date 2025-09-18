from PyQt6.QtWidgets import (
 QVBoxLayout, QWidget, QLabel, QPushButton, QFormLayout, QHBoxLayout, QSpinBox,
)

from PyQt6.QtCore import QTimer
from typing import Protocol, runtime_checkable
from time import sleep, time
from functools import partial
import os
import abc

def parse_current_mA(reply: str) -> float:
    # take text before "ma", grab the last whitespace-separated token, convert
    return float(reply.lower().split('ma')[0].split()[-1])




@runtime_checkable
class LaserController(Protocol):
    """Protocol for classes that control a laser source."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...

    def set_current(self, current: float) -> None: ...

    def turn_on_output(self) -> None: ...
    def turn_off_output(self) -> None: ...
    def is_output_on(self) -> bool: ...


class TestLaserController(LaserController):
    """
    A test/mock implementation of the LaserController for testing purposes.
    It simulates a laser controller without actual hardware interaction.
    """

    def __init__(self):
        self.current = 0
        self.output_on = False
        self.laser_ser = None

    def connect(self, adress):
        print(f"Mock connect to laser at {adress}")
        self.laser_ser = True

    def set_current(self, current):
        if int(current)>400 or int(current)<0:
            return
        self.current = current        
        print(f"Mock set current to {self.current} mA")

    def is_connected(self):
        return self.laser_ser is not None

    def is_output_on(self):
        return self.output_on

    def turn_on_output(self):
        print("Mock turn on laser output")
        self.output_on = True
    
    def turn_off_output(self):
        print("Mock turn off laser output")
        self.output_on = False

    def disconnect(self):
        print("Mock disconnect laser")
        self.laser_ser = None

class LaserControllerWidget(QWidget):

    def __init__(self, c_p, OT_GUI, laser_A, laser_B):
        super().__init__()
        self.c_p = c_p
        self.OT_GUI = OT_GUI
        self.setWindowTitle("Laser controller")
        
        self.current_A_edit_val = self.c_p['laser_A_current']
        self.current_B_edit_val = self.c_p['laser_B_current']
        self.laser_A = laser_A
        self.laser_B = laser_B

        if not self.laser_A.is_connected():
            self.laser_A.connect(self.c_p['laser_A_port'])
        if not self.laser_B.is_connected():
            self.laser_B.connect(self.c_p['laser_B_port'])

        # create a timer for timed updates of lasers, used primarily when doing red blood cells experiments
        self.timer = QTimer(self)

        # set timer timeout callback function
        self.timer.timeout.connect(self.laser_power_protocol)

        # set the timer to fire every 100 milliseconds (10 times per second)
        self.timer.start(100)
        
        self.experiment_start_time = 0
        self.current_power_start_time = 0
        self.experiment_idx = 0
        self.experiment_started = False
        self.snapshot_taken = False
        self.time_interval = 10 # Decrease?
        self.particle_no = 0

        self.initiate_interface()        


    def get_name(self, idx):
        folder = '\particle_no-'+str(self.particle_no)
        if not os.path.exists(self.c_p['recording_path']+folder):
            os.mkdir(self.c_p['recording_path']+folder)
        self.c_p['filename'] = (folder + '\particle_experiment_no-' + str(self.experiment_idx) +'_A' +
                                str(self.c_p['power_protocol_currents'][idx][0]) + '-B' + 
                                str(self.c_p['power_protocol_currents'][idx][1]))
        return

    def start_data_recording(self):
        self.OT_GUI.start_saving()
        if self.c_p['recording']:
            self.OT_GUI.toggle_recording()
            print("Warning recording was already on, turning off")
            time.sleep(0.05)
        self.OT_GUI.toggle_recording()
        self.snapshot_taken = False

    def stop_data_recording(self):
        self.OT_GUI.stop_saving()
        if not self.c_p['recording']:
            print("Warning recording was turned off")
            return
        self.OT_GUI.toggle_recording()

    def laser_power_currents(self):
        self.current_A_edit_val = int(self.c_p['power_protocol_currents'][self.experiment_idx][0])
        self.current_B_edit_val = int(self.c_p['power_protocol_currents'][self.experiment_idx][1])
        self.time_interval = int(self.c_p['power_protocol_currents'][self.experiment_idx][2])

        self.laser_A_current_box.setValue(int(self.current_A_edit_val))
        self.laser_B_current_box.setValue(int(self.current_B_edit_val))
        self.set_laser_A_current()
        self.set_laser_B_current()
        sleep(0.05)

    def laser_power_protocol(self):
        self.toggle_experiment_button.setChecked(self.c_p['laser_power_protocol_running'])
        if not self.c_p['laser_power_protocol_running']:
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
            self.laser_power_currents()
            self.get_name(self.experiment_idx)
            self.particle_no += 1
            self.start_data_recording()
            
        if self.c_p['program_running'] and self.c_p['laser_power_protocol_running']:
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
                if self.experiment_idx >= len(self.c_p['power_protocol_currents']):
                    print("Experiment done")
                    self.c_p['laser_power_protocol_running'] = False
                    self.experiment_started = False
                    self.toggle_experiment_button.setChecked(False)
                    return
                self.laser_power_currents()
                self.get_name(self.experiment_idx)
                self.start_data_recording()

    def initiate_interface(self):
        layout = QVBoxLayout()

        # Laser A
        self.laser_A_label = QLabel("Laser A", self)
        layout.addWidget(self.laser_A_label)
        self.laser_A_layout = QFormLayout()
        self.connectA_btn = QPushButton("Connect", self)
        connect_laser_A = partial(self.connect_laser, 'A')
        self.connectA_btn.clicked.connect(connect_laser_A)
        self.laser_A_layout.addRow("Connect", self.connectA_btn)
        self.toggle_A_btn = QPushButton("Turn ON", self)        
        self.toggle_A_btn.clicked.connect(self.toggle_laser_A)
        self.laser_A_layout.addRow("Output", self.toggle_A_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentA_layout = QHBoxLayout()

        # Create spinbox for current
        self.laser_A_current_box = QSpinBox()
        self.laser_A_current_box.setRange(0, 400)
        self.laser_A_current_box.setValue(self.c_p['laser_A_current'])
        self.laser_A_current_box.valueChanged.connect(self.set_current_A)
        self.currentA_layout.addWidget(self.laser_A_current_box)

        self.set_current_button = QPushButton("Set current", self)
        self.set_current_button.clicked.connect(self.set_laser_A_current)
        self.currentA_layout.addWidget(self.set_current_button)

        self.current_now_A_label = QLabel(
            f"Current (mA): {self.c_p['laser_A_current_current']} ma", self)
        self.laser_A_layout.addRow(self.current_now_A_label)
        

        self.laser_A_layout.addRow("Set current", self.currentA_layout)
        
        self.laser_A_port_select = QSpinBox()
        self.laser_A_port_select.setRange(0, 20)  # Assuming you have 10 COM ports; adjust accordingly
        self.laser_A_port_select.setValue(int(self.c_p['laser_A_port'][3:])) 
        self.laser_A_port_select.valueChanged.connect(self.set_laser_A_port)
        self.laser_A_layout.addRow("COM Port", self.laser_A_port_select)

        layout.addLayout(self.laser_A_layout)


        # Laser B
        self.laserB_label = QLabel("Laser B", self)
        layout.addWidget(self.laserB_label)
        self.laserB_layout = QFormLayout()
        self.connectB_btn = QPushButton("Connect", self)
        connect_laserB = partial(self.connect_laser, 'B')

        self.connectB_btn.clicked.connect(connect_laserB)
        self.laserB_layout.addRow("Connect", self.connectB_btn)
        self.toggle_B_btn = QPushButton("Turn ON", self)        
        self.toggle_B_btn.clicked.connect(self.toggle_laser_B)
        self.laserB_layout.addRow("Power", self.toggle_B_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentB_layout = QHBoxLayout()

        self.laser_B_current_box = QSpinBox()
        self.laser_B_current_box.setRange(0, 400)
        self.laser_B_current_box.setValue(self.c_p['laser_B_current'])
        self.laser_B_current_box.valueChanged.connect(self.set_current_B)
        self.currentB_layout.addWidget(self.laser_B_current_box)
        
        self.set_current_button = QPushButton("Set current", self)
        self.set_current_button.clicked.connect(self.set_laser_B_current)
        self.currentB_layout.addWidget(self.set_current_button)
        self.laserB_layout.addRow("Set current", self.currentB_layout)

        self.current_now_B_label = QLabel(
            f"Current (mA): {self.c_p['laser_B_current_current']} ma", self)
        self.laserB_layout.addRow(self.current_now_B_label)

        self.laser_B_port_select = QSpinBox()
        self.laser_B_port_select.setRange(0, 20)
        self.laser_B_port_select.setValue(int(self.c_p['laser_B_port'][3:])) 
        self.laser_B_port_select.valueChanged.connect(self.set_laser_B_port)
        self.laserB_layout.addRow("COM Port", self.laser_B_port_select)

        # Common stuff
        self.set_both_currents_button = QPushButton("Set currents", self)
        self.set_both_currents_button.clicked.connect(self.set_both_currents)
        self.laserB_layout.addRow("Set both currents", self.set_both_currents_button)

        self.toggle_experiment_button = QPushButton("Toggle automatic power changing experiment", self)
        self.toggle_experiment_button.clicked.connect(self.toggle_laser_power_experiment)
        self.toggle_experiment_button.setCheckable(True)
        self.laserB_layout.addRow("Toggle experiment", self.toggle_experiment_button)

       
        layout.addLayout(self.laserB_layout)

        self.setLayout(layout)

    def set_laser_A_port(self):
        value = self.laser_A_port_select.value()
        self.c_p['laser_A_port'] = f"COM{value}"

    def set_laser_B_port(self, value):
        value = self.laser_B_port_select.value()
        self.c_p['laser_B_port'] = f"COM{value}"

    def connect_laser(self, laser):
        if laser == 'A':
            self.laser_A.connect(self.c_p['laser_A_port'])

        elif laser == 'B':
            self.laser_B.connect(self.c_p['laser_B_port'])

    def set_current_A(self, current):
        self.current_A_edit_val = int(current)
    
    def set_current_B(self, current):
        self.current_B_edit_val = int(current)

    def toggle_laser_power_experiment(self):
        self.c_p['laser_power_protocol_running'] = not self.c_p['laser_power_protocol_running']

    def set_both_currents(self):
        self.set_laser_A_current()
        self.set_laser_B_current()

    def set_laser_A_current(self):
        current = self.laser_A_current_box.value()
        self.laser_A.set_current(current)

    def set_laser_B_current(self):
        current = self.laser_B_current_box.value()
        self.laser_B.set_current(current)

    def toggle_laser_A(self):        
        if self.laser_A.is_output_on():
            self.laser_A.turn_off_output()
            self.toggle_A_btn.setText("Turn ON")
            self.c_p['laser_A_on'] = False
        else:
            self.laser_A.turn_on_output()
            self.toggle_A_btn.setText("Turn OFF")
            self.c_p['laser_A_on'] = True

    def toggle_laser_B(self):
        if self.laser_B.is_output_on():
            self.laser_B.turn_off_output()
            self.toggle_B_btn.setText("Turn ON")
            self.c_p['laser_B_on'] = False
        else:
            self.laser_B.turn_on_output()
            self.toggle_B_btn.setText("Turn OFF")
            self.c_p['laser_B_on'] = True

    def disconnect_laser(self, laser):
        if laser == 'A' and self.laser_A.is_connected():
            self.laser_A.disconnect()
        elif laser == 'B' and self.laser_B.is_connected():
            self.laser_B.disconnect()

    def closeEvent(self, event):
        self.disconnect_laser('A')
        self.disconnect_laser('B')
        event.accept()

