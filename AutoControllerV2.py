import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog,
    QHBoxLayout,
)
from PyQt6 import  QtGui
from PyQt6.QtCore import QTimer

from CustomMouseTools import MouseInterface
from threading import Thread
from time import sleep, time
from queue import PriorityQueue
import torch
import torch.nn as nn
from yolov5 import YOLOv5
from ultralytics import YOLO
# from models.experimental import attempt_load  # This is specific to YOLOv5
from collections import deque

import statistics
import cv2

def go2position(position,c_p):
    # TODO use this version of go2position eveywhere go2position is used.
    if c_p['move_to_location']:
        # Stop if already moving.
        c_p['move_to_location'] = False
        c_p['motor_x_target_speed'] = 0
        c_p['motor_y_target_speed'] = 0
        c_p['motor_z_target_speed'] = 0
        return
    if c_p['minitweezers_connected']:
        c_p['minitweezers_target_pos'][0] = int(position[0])
        c_p['minitweezers_target_pos'][1] = int(position[1])
        c_p['minitweezers_target_pos'][2] = int(position[2])
        c_p['move_to_location'] = True


# Initialize a deque with a fixed size of 10
class FixedQueue(deque):
    def __init__(self, size=10):
        super().__init__(maxlen=size)

    def get_average(self):
        # Calculate and return the average of elements
        if self:  # Check if the deque is not empty
            return statistics.mean(self)
        else:
            return 0  # Return 0 if the deque is empty

class ParticleCNN(nn.Module):
    def __init__(self):
        super(ParticleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Assuming grayscale images
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Size now 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Size now 32x32
        )
        # Calculate the size after convolutions and pooling
        # For 128x128 input, after two pooling layers, the size is 32x32
        # And if you have 64 output channels from the last conv layer, then:
        self.size_after_convs = 64 * 32 * 32
        self.fc_layers = nn.Sequential(
            nn.Linear(self.size_after_convs, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Predicting a single value
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # Flatten the output for the fully connected layer
        x = self.fc_layers(x)
        return x


def load_yolov5_model(model_path):
    model = torch.hub.load('.', 'custom', path=model_path, source='local') 
    return model

# A* algorthm can be found at  https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
# Can also just use this version which is a blend of chatgpt and my own code.
directions = [(0, 1), (1, 0), (1, 1), (-1, 1),(1, -1),(-1, -1), (0, -1), (-1, 0)]  # Up, Right, Down, Left

def convert2uint16(num):
    if num<0:
        return np.uint16(0)
    if num>65535:
        return np.uint16(65535)
    return np.uint16(num)


def heuristic(point, end):
    return abs(point[0] - end[0]) + abs(point[1] - end[1])



def a_star(grid, start, end):
    pq = PriorityQueue()
    pq.put((0, start))  # The queue stores tuples (priority, point)
    
    came_from = {start: None}  # Dictionary to store the path
    cost_so_far = {start: 0}  # Dictionary to store current cost

    # Check if target is outside of grid, if so set it to the grid edges instead
    s = grid.shape
    if end[0] >= s[0]:
        end = (s[0]-1,end[1])
    if end[1] >= s[1]:
        end = (end[0],s[1]-1)
    if end[0] < 0:
        end = (0,end[1])
    if end[1] < 0:
        end = (end[0],0)


    while not pq.empty():
        _, current = pq.get()
        
        if current == end:
            break
        
        for direction in directions:
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            
            if (0 <= next_cell[0] < len(grid) and 0 <= next_cell[1] < len(grid[0]) and grid[next_cell[0]][next_cell[1]]):
                new_cost = cost_so_far[current] + 1
                
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + heuristic(next_cell, end)
                    pq.put((priority, next_cell))
                    came_from[next_cell] = current
    
    # Build the path from end to start
    path = []
    current = end
    while current is not None:
        path.append(current)
        try:
            current = came_from[current]
        except KeyError as ke:
            return None
    path.reverse()
    
    return path

def generate_move_map(size, width, height, positions, radii,rect=None, start_pos=None):
    y_size = int(size * height/width)
    area = np.ones((size,y_size))
    # Convert from pixel to grid size
    # norm_positions = positions/width * size
    # radii = radii / width * size 
    # TODO change so that particle positions are entered as grid positions
    for pos in positions:
        x = pos[0]/width * size
        y = pos[1]/height * y_size
        x_min = max(int(x - radii), 0)
        x_max = min(int(x + radii + 1.5), size)
        y_min = max(int(y - radii), 0)
        y_max = min(int(y + radii + 1.5), y_size)
        
        #print("setting area to 1",x_min,x_max,y_min,y_min)
        area[x_min:x_max, y_min:y_max] = 0

    if rect is not None:
        """
        x = rect[0]/width * size
        y = rect[1]/height * y_size
        x_min = max(int(x), 0)
        x_max = min(int(rect[2]/width * size), size)
        y_min = max(int(y), 0)
        y_max = min(int(rect[3]/height * y_size), y_size)
        """
        # ordered as x0,y0,x1,y1
        area[rect[0]:rect[2], rect[1]:rect[3]] = 0 # TODO set the y end to 0
    return area

def simplify_path(path):
    """ Simplifies a given path by removing unnecessary points """
    simplified_path = [path[0]]

    for i in range(1, len(path)-1):
        dx1 = path[i][0] - path[i-1][0]
        dy1 = path[i][1] - path[i-1][1]
        dx2 = path[i+1][0] - path[i][0]
        dy2 = path[i+1][1] - path[i][1]

        # If the direction changed, add this point
        if dx1 != dx2 or dy1 != dy2:
            simplified_path.append(path[i])

    simplified_path.append(path[-1])  # Always add the last point

    return simplified_path



class AutoControlWidget(QWidget):
    def __init__(self, c_p, data_channels):
        # Maybe have the widget poll the different parameters and update the buttons accordingly.
        super().__init__()
        self.c_p = c_p
        self.data_channels = data_channels
        layout = QVBoxLayout()
        self.setWindowTitle("Auto Controller")

        self.toggle_centering_button = QPushButton('Centering on')
        self.toggle_centering_button.pressed.connect(self.center_particle)
        self.toggle_centering_button.setCheckable(True)
        self.toggle_centering_button.setChecked(self.c_p['centering_on'])
        layout.addWidget(self.toggle_centering_button)

        self.toggle_trap_button = QPushButton('Trap particle')
        self.toggle_trap_button.pressed.connect(self.trap_particle)
        self.toggle_trap_button.setCheckable(True)
        self.toggle_trap_button.setChecked(self.c_p['trap_particle'])
        layout.addWidget(self.toggle_trap_button)

        # Add a button for searching and trapping particles as well as a combobox for selecting which capillary to use.
        self.serach_and_trap_layout = QHBoxLayout()
        self.search_and_trap_button = QPushButton('Search and trap')
        self.search_and_trap_button.pressed.connect(self.search_and_trap)
        self.search_and_trap_button.setCheckable(True)
        self.search_and_trap_button.setChecked(self.c_p['search_and_trap'])
        self.serach_and_trap_layout.addWidget(self.search_and_trap_button)
        
        self.capillary_selection = QComboBox()
        self.capillary_selection.addItem("Capillary 1")
        self.capillary_selection.addItem("Capillary 2")
        self.capillary_selection.setCurrentIndex(0)
        self.capillary_selection.currentIndexChanged.connect(self.capillary_selection_changed)
        self.serach_and_trap_layout.addWidget(self.capillary_selection)

        layout.addLayout(self.serach_and_trap_layout)

        self.center_pipette_button = QPushButton('Center pipette')
        self.center_pipette_button.pressed.connect(self.toggle_center_pipette)
        self.center_pipette_button.setCheckable(True)
        self.center_pipette_button.setChecked(self.c_p['center_pipette'])
        layout.addWidget(self.center_pipette_button)



        self.calculate_laser_position_button = QPushButton('Calculate laser position')
        self.calculate_laser_position_button.pressed.connect(self.update_laser_position)
        self.calculate_laser_position_button.setToolTip("Calculates the true laser position based on the predicted particle positions. \n This is done by finding the particle closest to the laser position and setting the laser position to that particle.")
        self.calculate_laser_position_button.setCheckable(False)
        layout.addWidget(self.calculate_laser_position_button)

        """
        self.attach_DNA_button = QPushButton('Attach DNA')
        self.attach_DNA_button.pressed.connect(self.attach_DNA)
        self.attach_DNA_button.setCheckable(True)
        self.attach_DNA_button.setChecked(self.c_p['attach_DNA_automatically'])
        layout.addWidget(self.attach_DNA_button)

        self.set_pippette_location_chamber_button = QPushButton('Set pipette location')
        self.set_pippette_location_chamber_button.pressed.connect(self.set_pippette_location_chamber)
        self.set_pippette_location_chamber_button.setToolTip("Sets the approximate location of the pipette to the current motor position.\n Default is that the pipette is at 0,0 of the motors. Use before trying to locate the pipette.")
        layout.addWidget(self.set_pippette_location_chamber_button)
        
        self.set_tube_AD_tube_position_button = QPushButton('Set tube AD tube position')
        self.set_tube_AD_tube_position_button.pressed.connect(self.set_tube_AD_tube_position)
        self.set_tube_AD_tube_position_button.setToolTip("Sets an approximate position of the AD tube- \n Specifically a position in which it is easy to see beads flowing by.")
        layout.addWidget(self.set_tube_AD_tube_position_button)
        """

        """
        self.move_while_avoiding_beads_button = QPushButton('Move while avoiding beads')
        self.move_while_avoiding_beads_button.pressed.connect(self.move_while_avoiding)
        self.move_while_avoiding_beads_button.setToolTip("Moves the pipette to the specified location while avoiding beads. \n This is done by generating a map of where the beads are and then using A* to find a path to the destination.")
        self.move_while_avoiding_beads_button.setCheckable(True)
        self.move_while_avoiding_beads_button.setChecked(self.c_p['move_avoiding_particles'])
        layout.addWidget(self.move_while_avoiding_beads_button)

        
        # Functions for electrostatic protocol, not used at the moment.
        self.EP_toggled_button = QPushButton('Toggle EP')
        self.EP_toggled_button.pressed.connect(self.toggle_EP)
        self.EP_toggled_button.setCheckable(True)
        self.EP_toggled_button.setChecked(self.c_p['electrostatic_protocol_toggled'])
        layout.addWidget(self.EP_toggled_button)

        self.EP_start_spinbox = QSpinBox()
        self.EP_start_spinbox.setRange(0, 65535)
        self.EP_start_spinbox.valueChanged.connect(self.update_EP_start)
        self.EP_start_spinbox.setValue(self.c_p['electrostatic_protocol_start'])
        layout.addWidget(QLabel("EP start:"))
        layout.addWidget(self.EP_start_spinbox)

        self.EP_end_spinbox = QSpinBox()
        self.EP_end_spinbox.setRange(0, 65535)
        self.EP_end_spinbox.valueChanged.connect(self.update_EP_end)
        self.EP_end_spinbox.setValue(self.c_p['electrostatic_protocol_end'])
        layout.addWidget(QLabel("EP end:"))
        layout.addWidget(self.EP_end_spinbox)

        self.EP_step_spinbox = QSpinBox()
        self.EP_step_spinbox.setRange(0, 65535)
        self.EP_step_spinbox.valueChanged.connect(self.update_EP_step)
        self.EP_step_spinbox.setValue(self.c_p['electrostatic_protocol_steps'])
        layout.addWidget(QLabel("EP step:"))
        layout.addWidget(self.EP_step_spinbox)

        self.EP_duration_spinbox = QSpinBox()
        self.EP_duration_spinbox.setRange(0, 65535)
        self.EP_duration_spinbox.valueChanged.connect(self.update_EP_duration)
        self.EP_duration_spinbox.setValue(self.c_p['electrostatic_protocol_duration'])
        layout.addWidget(QLabel("EP duration:"))
        layout.addWidget(self.EP_duration_spinbox)
        """

        # Add buttons for setting the capillary positions and going to these.
        self.set_capillary_1_position_button = QPushButton('Set capillary 1 position')
        self.set_capillary_1_position_button.pressed.connect(self.set_capillary_1_position)
        self.set_capillary_1_position_button.setToolTip("Sets the approximate location of the capillary 1 to the current motor position.")

        self.set_go2capillary_1_button = QPushButton('Go to capillary 1')
        self.set_go2capillary_1_button.pressed.connect(self.goto_capillary_1)
        self.set_go2capillary_1_button.setToolTip("Moves the pipette to the capillary 1 position.")
        
        self.capillary_1_layout = QHBoxLayout()
        self.capillary_1_layout.addWidget(self.set_capillary_1_position_button)
        self.capillary_1_layout.addWidget(self.set_go2capillary_1_button)
        layout.addLayout(self.capillary_1_layout)

        # Second capillary
        self.set_capillary_2_position_button = QPushButton('Set capillary 2 position')
        self.set_capillary_2_position_button.pressed.connect(self.set_capillary_2_position)
        self.set_capillary_2_position_button.setToolTip("Sets the approximate location of the capillary 2 to the current motor position.")

        self.set_go2capillary_2_button = QPushButton('Go to capillary 2')
        self.set_go2capillary_2_button.pressed.connect(self.goto_capillary_2)
        self.set_go2capillary_2_button.setToolTip("Moves the pipette to the capillary 2 position.")

        self.capillary_2_layout = QHBoxLayout()
        self.capillary_2_layout.addWidget(self.set_capillary_2_position_button)
        self.capillary_2_layout.addWidget(self.set_go2capillary_2_button)
        layout.addLayout(self.capillary_2_layout)

        # Pipette
        self.set_pipette_position_button = QPushButton('Set pipette position')
        self.set_pipette_position_button.pressed.connect(self.set_pipette_position)
        self.set_pipette_position_button.setToolTip("Sets the approximate location of the pipette to the current motor position.")

        self.set_go2pipette_button = QPushButton('Go to pipette')
        self.set_go2pipette_button.pressed.connect(self.goto_pipette)
        self.set_go2pipette_button.setToolTip("Moves the pipette to the pipette position.")

        self.pipette_layout = QHBoxLayout()
        self.pipette_layout.addWidget(self.set_pipette_position_button)
        self.pipette_layout.addWidget(self.set_go2pipette_button)
        layout.addLayout(self.pipette_layout)

        self.z_focus_button = QPushButton('Toggle z focus')
        self.z_focus_button.pressed.connect(self.toggle_z_focus_pipette_trapped)
        self.z_focus_button.setCheckable(True)
        self.z_focus_button.setToolTip("Toggles the z focus on the pipette. \n This is used to match the focus of the trapped particle with that in the pipette.")
        self.z_focus_button.setChecked(self.c_p['focus_z_trap_pipette'])
        layout.addWidget(self.z_focus_button)

        self.pipette_focus_button = QPushButton('Toggle pipette focus')
        self.pipette_focus_button.pressed.connect(self.toggle_pipette_focus)
        self.pipette_focus_button.setCheckable(True)
        self.pipette_focus_button.setToolTip("Toggles the focus on the pipette. \n This is used to match the focus of the pipette with the trapped particle.")
        self.pipette_focus_button.setChecked(self.c_p['focus_pipette'])
        layout.addWidget(self.pipette_focus_button)

        self.move2area_above_button = QPushButton('Move to area above pipette')
        self.move2area_above_button.pressed.connect(self.move2area_above)
        self.move2area_above_button.setToolTip("Moves the pipette to the area above the pipette. \n Position the particle roughly above the pipette, primarily used in testing.")
        self.move2area_above_button.setCheckable(True)
        layout.addWidget(self.move2area_above_button)

        self.move_particle2pipette_button = QPushButton('Move particle to pipette')
        self.move_particle2pipette_button.pressed.connect(self.move2pipette_tip)
        self.move_particle2pipette_button.setToolTip("Moves the trapped particle to the pipette tip. \n Used to trap particles in the pipette.")
        self.move_particle2pipette_button.setCheckable(True)
        layout.addWidget(self.move_particle2pipette_button)

        self.suck_into_pipette_button = QPushButton('Suck into pipette')
        self.suck_into_pipette_button.pressed.connect(self.suck_into_pipette)
        self.suck_into_pipette_button.setToolTip("Sucks the particle into the pipette. \n Used to trap particles in the pipette.")
        self.suck_into_pipette_button.setCheckable(True)
        layout.addWidget(self.suck_into_pipette_button)


        self.move_piezo_2_saved_positions_button = QPushButton('Move piezo to saved positions')
        self.move_piezo_2_saved_positions_button.pressed.connect(self.move_piezo_2_saved_positions)
        self.move_piezo_2_saved_positions_button.setToolTip("Moves the piezos to a saved position. \n Usefull for resetting after having done a trapping.")
        self.move_piezo_2_saved_positions_button.setCheckable(True)
        layout.addWidget(self.move_piezo_2_saved_positions_button)

        self.touch_particles_button = QPushButton('Touch particles')
        self.touch_particles_button.pressed.connect(self.touch_particles)
        self.touch_particles_button.setToolTip("Moves the pipette to the particle closest to the pipette. \n Used to touch particles to see if they are trapped.")
        self.touch_particles_button.setCheckable(True)
        layout.addWidget(self.touch_particles_button)

        self.stretch_molecule_button = QPushButton('Stretch molecule')
        self.stretch_molecule_button.pressed.connect(self.stretch_molecule)
        self.stretch_molecule_button.setToolTip("Moves the piezo to stretch the molecule. \n Used to stretch the molecule after having attached the DNA.")
        self.stretch_molecule_button.setCheckable(True)
        layout.addWidget(self.stretch_molecule_button)

        self.autonomous_button = QPushButton('Autonomous experiment')
        self.autonomous_button.pressed.connect(self.toggle_autonomous_experiment)
        self.autonomous_button.setToolTip("Runs an autonomous experiment. \n This is used to run a full experiment without user intervention.")
        self.autonomous_button.setCheckable(True)
        layout.addWidget(self.autonomous_button)

        # We add this autonomous experiment spinbox to manually change the step we are in while testing the protocol.
        self.autonomous_experiment_stage_box = QComboBox()
        for state in self.c_p['autonomous_experiment_states']:
            self.autonomous_experiment_stage_box.addItem(state)
        self.autonomous_experiment_stage_box.setCurrentIndex(0)
        self.autonomous_experiment_stage_box.currentIndexChanged.connect(self.autonomous_experiment_stage_changed)
        layout.addWidget(self.autonomous_experiment_stage_box)

        self.drop_particles_button = QPushButton('Drop particles')
        self.drop_particles_button.pressed.connect(self.drop_particles)
        self.drop_particles_button.setToolTip("Drops the trapped particle. \n Used to release particles from the pipette.")
        self.drop_particles_button.setCheckable(True)
        layout.addWidget(self.drop_particles_button)

        self.auto_calibrate_button = QPushButton('Auto calibrate')
        self.auto_calibrate_button.pressed.connect(self.auto_calibrate)
        self.auto_calibrate_button.setToolTip("Calibrates by moving the particle in a grid and recording the positions and forces read by the lasers.")
        self.auto_calibrate_button.setCheckable(True)
        layout.addWidget(self.auto_calibrate_button)        

        # Initiate a timeer to alert the user in case one of the protocols finishes.
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

        self.setLayout(layout)

    def refresh(self):
        """
        Refreshes all the indicators of the autocontroller.
        """
        self.z_focus_button.setChecked(self.c_p['focus_z_trap_pipette'])
        self.pipette_focus_button.setChecked(self.c_p['focus_pipette'])
        self.move2area_above_button.setChecked(self.c_p['move2area_above_pipette'])
        self.move_particle2pipette_button.setChecked(self.c_p['move_particle2pipette'])
        self.toggle_centering_button.setChecked(self.c_p['centering_on'])
        self.toggle_trap_button.setChecked(self.c_p['trap_particle'])
        self.search_and_trap_button.setChecked(self.c_p['search_and_trap'])
        self.center_pipette_button.setChecked(self.c_p['center_pipette'])
        self.attach_DNA_button.setChecked(self.c_p['attach_DNA_automatically'])

        # Electrostatic protocol
        # self.move_while_avoiding_beads_button.setChecked(self.c_p['move_avoiding_particles'])

        # self.EP_toggled_button.setChecked(self.c_p['electrostatic_protocol_toggled'])
        # self.EP_start_spinbox.setValue(self.c_p['electrostatic_protocol_start'])
        # self.EP_end_spinbox.setValue(self.c_p['electrostatic_protocol_end'])
        # self.EP_step_spinbox.setValue(self.c_p['electrostatic_protocol_steps'])
        # self.EP_duration_spinbox.setValue(self.c_p['electrostatic_protocol_duration'])

        self.move_piezo_2_saved_positions_button.setChecked(self.c_p['move_piezo_2_target'])
        self.touch_particles_button.setChecked(self.c_p['touch_particles'])
        self.stretch_molecule_button.setChecked(self.c_p['stretch_molecule'])
        self.drop_particles_button.setChecked(self.c_p['drop_particle'])
        self.autonomous_button.setChecked(self.c_p['autonomous_experiment'])
        self.suck_into_pipette_button.setChecked(self.c_p['suck_into_pipette'])
        self.auto_calibrate_button.setChecked(self.c_p['calibration_running'])

    def toggle_EP(self):
        if not self.EP_toggled_button.isChecked() and not self.c_p['electrostatic_protocol_toggled']:
            self.c_p['electrostatic_protocol_toggled'] = not self.c_p['electrostatic_protocol_toggled']
        #print(self.c_p['electrostatic_protocol_toggled'])
        if self.EP_toggled_button.isChecked() and self.c_p['electrostatic_protocol_toggled']:
            self.c_p['electrostatic_protocol_toggled'] = not self.c_p['electrostatic_protocol_toggled']
            self.c_p['electrostatic_protocol_finished'] = False
            self.c_p['electrostatic_protocol_running'] = False

        if self.c_p['electrostatic_protocol_toggled']:
            self.c_p['electrostatic_protocol_finished'] = False
        print("Toggled EP to ", self.c_p['electrostatic_protocol_toggled'], self.EP_toggled_button.isChecked())
    
    def toggle_z_focus_pipette_trapped(self):
        self.c_p['focus_z_trap_pipette'] = not self.z_focus_button.isChecked()

    def toggle_autonomous_experiment(self):
        self.c_p['autonomous_experiment'] = not self.autonomous_button.isChecked()

    def toggle_pipette_focus(self):
        self.c_p['focus_pipette'] = not self.pipette_focus_button.isChecked()
        if self.c_p['focus_pipette']:
            self.c_p['pipette_focus_startup'] = True
            self.c_p['pipette_sharpnesses'] = []
            self.c_p['pipette_sharpness_Z_pos'] = []

    def move2pipette_tip(self):
        self.c_p['move_particle2pipette'] = not self.move_particle2pipette_button.isChecked()

    def move2area_above(self):
        self.c_p['move2area_above_pipette'] = not self.move2area_above_button.isChecked()

    def update_EP_start(self):
        val = int(self.EP_start_spinbox.value())
        self.c_p['electrostatic_protocol_start'] = val

    def update_EP_end(self):
        val = int(self.EP_end_spinbox.value())
        self.c_p['electrostatic_protocol_end'] = val

    def update_EP_step(self):
        val = int(self.EP_step_spinbox.value())
        self.c_p['electrostatic_protocol_steps'] = val

    def update_EP_duration(self):
        val = int(self.EP_duration_spinbox.value())
        self.c_p['electrostatic_protocol_duration'] = val

    def update_laser_position(self):
        self.c_p['find_laser_position'] = True

    def set_capillary_1_position(self):
        x = int(self.data_channels['Motor_x_pos'].get_data(1)[0])
        y = int(self.data_channels['Motor_y_pos'].get_data(1)[0])
        z = int(self.data_channels['Motor_z_pos'].get_data(1)[0])
        self.c_p['capillary_1_position'] = [x, y, z]

    def set_capillary_2_position(self):
        x = int(self.data_channels['Motor_x_pos'].get_data(1)[0])
        y = int(self.data_channels['Motor_y_pos'].get_data(1)[0])
        z = int(self.data_channels['Motor_z_pos'].get_data(1)[0])
        self.c_p['capillary_2_position'] = [x, y, z]

    def set_pipette_position(self):
        x = int(self.data_channels['Motor_x_pos'].get_data(1)[0])
        y = int(self.data_channels['Motor_y_pos'].get_data(1)[0])
        z = int(self.data_channels['Motor_z_pos'].get_data(1)[0])
        self.c_p['pipette_location_chamber'] = [x, y, z]

    def goto_capillary_1(self):
        go2position(self.c_p['capillary_1_position'],self.c_p)

    def goto_capillary_2(self):
        go2position(self.c_p['capillary_2_position'],self.c_p)

    def goto_pipette(self):
        go2position(self.c_p['pipette_location_chamber'],self.c_p)

    def suck_into_pipette(self):
        self.c_p['suck_into_pipette'] = not self.suck_into_pipette_button.isChecked()

    def center_particle(self):
        """
        Function that seeks to keep a particle in the center 
        of the image. 
        """
        self.c_p['centering_on'] = not self.c_p['centering_on']

    def attach_DNA(self):
        self.c_p['attach_DNA_automatically'] = not self.attach_DNA_button.isChecked()
    
    def trap_particle(self):
        """
        Function that seeks to keep a particle in the center 
        of the image. 
        """
        self.c_p['trap_particle'] = not self.c_p['trap_particle']
    
    def drop_particles(self):
        """
        Function that seeks to keep a particle in the center 
        of the image. 
        """
        self.c_p['drop_particle'] = not self.drop_particles_button.isChecked()

    def search_and_trap(self):
        """
        Function that seeks to keep a particle in the center 
        of the image. 
        """
        self.c_p['search_and_trap'] = not self.c_p['search_and_trap']

    def auto_calibrate(self):
        self.c_p['calibration_running'] = not self.c_p['calibration_running']
        if self.c_p['calibration_running']:
            self.c_p['calibration_start'] = True

    def capillary_selection_changed(self):
        self.c_p['particle_type'] = self.capillary_selection.currentIndex()+1
        print(f"Particle type set to {self.c_p['particle_type']}")

    def autonomous_experiment_stage_changed(self):
        state = self.autonomous_experiment_stage_box.currentText()
        self.c_p['autocontroller_current_step'] = state
        print(f"Autonomous experiment stage set to {state}")

    def toggle_center_pipette(self):
        """
        Moves the stage so that the pipette is in the center of the image
        """
        self.c_p['center_pipette'] = not self.center_pipette_button.isChecked()

        # If the button was just pressed then we should find the exact pipette location.
        if self.c_p['center_pipette']:
            self.c_p['pipette_located'] = False
        else:
            self.c_p['locate_pipette'] = False
            self.c_p['move_to_location'] = False
    """
    def set_pippette_location_chamber(self):
        Sets the approximate location of the pipette to the current motor position.
        Default is that the pipette is at 0,0 of the motors.
        Use before trying to locate the pipette.
        x = self.data_channels['Motor_x_pos'].get_data(1)[0]
        y = self.data_channels['Motor_y_pos'].get_data(1)[0]
        z = self.data_channels['Motor_z_pos'].get_data(1)[0]
        self.c_p['pipette_location_chamber'] = [x, y, z]
        print("Approximate pipette location set to: ", self.c_p['pipette_location_chamber'])
    
    def set_tube_AD_tube_position(self):
        x = self.data_channels['Motor_x_pos'].get_data(1)[0]
        y = self.data_channels['Motor_y_pos'].get_data(1)[0]
        z = self.data_channels['Motor_z_pos'].get_data(1)[0]
        self.c_p['AD_tube_position'] = [x, y, z]
    """

    def move_piezo_2_saved_positions(self):
        self.c_p['move_piezo_2_target'] = not self.move_piezo_2_saved_positions_button.isChecked()

    def touch_particles(self):
        self.c_p['touch_particles'] = not self.touch_particles_button.isChecked()
        self.c_p['touch_counter'] = 0
    
    def stretch_molecule(self):
        self.c_p['stretch_molecule'] = not self.stretch_molecule_button.isChecked()

    def move_while_avoiding(self):

        self.c_p['move_avoiding_particles'] = not self.c_p['move_avoiding_particles']



class SelectLaserPosition(MouseInterface):
    def __init__(self, c_p):
        self.c_p = c_p
        self.pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
        self.pen2 = QtGui.QPen(QtGui.QColor(255, 0, 0))

    def draw(self, qp):
        qp.setPen(self.pen)
        r = 10
        x = int((self.c_p['laser_position_A_predicted'][0] - self.c_p['AOI'][0]) / self.c_p['image_scale'] - r/2)
        y = int((self.c_p['laser_position_A_predicted'][1] - self.c_p['AOI'][2]) / self.c_p['image_scale'] - r/2)
        qp.drawEllipse(x,y, r, r)

        qp.setPen(self.pen2)
        x = int((self.c_p['laser_position_B_predicted'][0] - self.c_p['AOI'][0]) / self.c_p['image_scale'] - r/2)
        y = int((self.c_p['laser_position_B_predicted'][1] - self.c_p['AOI'][2]) / self.c_p['image_scale'] - r/2)
        qp.drawEllipse(x,y, r, r)

    def mousePress(self):
        # TODO also set 0 position of the position lasers.
        if self.c_p['mouse_params'][0] == 1:
            self.c_p['laser_position_A'] = np.array(self.c_p['mouse_params'][1:3])*self.c_p['image_scale']
            self.c_p['laser_position_A'][0] += self.c_p['AOI'][0]
            self.c_p['laser_position_A'][1] += self.c_p['AOI'][2]
            print("Laser A position set to: ", self.c_p['laser_position_A'])
        elif self.c_p['mouse_params'][0] == 2:
            self.c_p['laser_position_B'] = np.array(self.c_p['mouse_params'][1:3])*self.c_p['image_scale']
            self.c_p['laser_position_B'][0] += self.c_p['AOI'][0]
            self.c_p['laser_position_B'][1] += self.c_p['AOI'][2]
            print("Laser B position set to: ", self.c_p['laser_position_A'])

    def mouseRelease(self):
        pass
    def mouseDoubleClick(self):
        pass
    def mouseMove(self):
        pass
    def getToolName(self):
        return "Laser position"
    def getToolTip(self):
        return "Click on the screen where the laser is located\n Used to tell the auto controll functions where to expect particles to be trappable."

class autoControllerThread(Thread):
    """
    New and updated version of the autoControllerThraed. The most important change is that now the deep-learning thread is implemented
    in the autoControllerThread. This is done to simplify synchronization between the different threads.

    """
    def __init__(self, c_p, data_channels, main_window=None  ):
        super().__init__()

        self.c_p = c_p
        self.setDaemon(True)
        self.particles_in_view = False
        self.data_channels = data_channels
        self.search_direction = 1 
        self.y_lim_pos = 1 # search limits
        self.x_lim_pos = 0.1
        self.main_window = main_window

        # DNA attachment parameters
        self.DNA_move_direction = 0
        self.bits_per_pixel = 500 / self.c_p['microns_per_pix'] # Number of bits we need to change the piezo dac to move 1 micron, approximate
        self.DNA_length_pix = 160  # approximate particle-particle-distance at which we should see a force ramp in the DNA
        self.closest_distance = 70
        self.sleep_counter = 0
        self.force_limit = 30
        self.last_move_time = 0
        self.protocol_start_time = 0
        # pipette-z focus parameters
        self.sharpnesses_mapped = False
  
        self.touch_counter_limit = 10 # should be 20
        self.pipette_view_counter = 0 # Counts how many frames the pipette has been out of view.
        self.z_move_counter = 0
        self.trapped_bead_z = FixedQueue(10) # Todo replace with the data_channel.
        self.pipette_bead_z = FixedQueue(10)
        
        self.go_trap_timer = 0
        self.in_trapping_area = False
        self.trapping_counter = 0

        # Autocalibration parameters
        self.calibration_timer = 0
        self.calibration_x = 0
        self.calibration_y = 0
        self.calibration_wait_time = 0.6 # How long to wait after moving the piezos before taking recording the data.
        self.calibration_dir = "forwards" # Direction of next x-step, moves back and forth, "forwards" or "backwards"
        self.grid_step = int(60_000 / self.c_p['grid_size'])

        # Deep learning part
        self.confidence_threshold_particle = 0.5 # Deteciton limit for the YOLO model
        self.confidence_threshold_pipette = 0.5 # Detection limit for the pipette(YOLO model)
        self.particle_lower_size_limit = 1.4/self.c_p['microns_per_pix'] # Smaller than 1.5 microns ish is too small to be relevant
        self.particle_upper_size_limit = 7/self.c_p['microns_per_pix'] # Smaller than 1.5 microns ish is too small to be relevant

        self.moving_down = True # Used to determine if we are moving up or down when touching particles,
        self.protocol_started = False
        self.limits_found = [False, False]
        self.moving2trap = False
        self.offset_y = 140 # Offset in pixels when moving to area above pipette

        if self.c_p['yolo_path'] is not None:
            self.c_p['model'] = load_yolov5_model(self.c_p['yolo_path'])
            self.c_p['network'] = "YOLOv5"
            self.c_p['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            print(f"Loaded default network from {self.c_p['yolo_path']}")
        if self.c_p['default_z_model_path'] is not None:
            print(f"Loading z-model")
            self.c_p['z-model'] = torch.load(self.c_p['default_z_model_path'])
            print(f"Loaded default z-model from {self.c_p['default_z_model_path']}")
  
    def YOLO_prediction(self):
        results = self.c_p['model'](self.c_p['image']).xywh[0].cpu().numpy()
        particle_positions = []
        radii = []
        pipette_located = False
        for prediction in results:
            # Check if particle
            if prediction [-1] == 0 and prediction[4] > self.confidence_threshold_particle:
                if prediction[2] > self.particle_lower_size_limit and prediction[3] > self.particle_lower_size_limit and prediction[2] < self.particle_upper_size_limit and prediction[3] < self.particle_upper_size_limit:
                    particle_positions.append(prediction[:2])
                    radii.append((prediction[2]+prediction[3])/4)

            # Check if pipette
            elif prediction[-1] == 1 and prediction[4] > self.confidence_threshold_pipette:
                pipette_located = True
                self.c_p['pipette_location'][0] = prediction[0]
                self.c_p['pipette_location'][1] = prediction[1] - prediction[3]/2
                self.c_p['pipette_location'][2] = prediction[2]
                self.c_p['pipette_location'][3] = prediction[3]
                self.pipette_view_counter = 0
                self.c_p['pipette_tip_location'][0] = prediction[0]
                self.c_p['pipette_tip_location'][1] = prediction[1] - prediction[3]/2
                # Correct for any eventual tilt of the pipette to get an accurate tip position.
                if self.c_p['accurate_tip_detection_needed']:
                    self.YOLO_pipette_tilt_detection()
                else:
                    self.c_p['pipette_tip_location'][0] -= self.c_p['pipette_tilt'] * self.c_p['pipette_location'][3]


        if not pipette_located:
            self.pipette_view_counter += 1
            if self.pipette_view_counter > 4:
                self.c_p['pipette_located'] = False
        else:
            self.c_p['pipette_located'] = True

        self.c_p['predicted_particle_positions'] = np.array(particle_positions)
        self.c_p['predicted_particle_radii'] = np.array(radii)

    def YOLO_pipette_tilt_detection(self, crop_width=600):
        """
        A second call to the YOLO model to detect the tilt of the pipette.
        Used to get a more accurate location of the pipette tip.
        """
        s = np.shape(self.c_p['image'])
        # If there is no pipette in the image or we are zoomed in then we should return.
        if not self.c_p['pipette_located'] or s[0]<crop_width or s[1]<crop_width:
            return
        # exctract a 500x500 pixel area around the pipette tip
        left = int(self.c_p['pipette_location'][1]-crop_width/2)
        right = int(self.c_p['pipette_location'][1]+crop_width/2)
        top = int(self.c_p['pipette_location'][0]-crop_width/2)
        bottom = int(self.c_p['pipette_location'][0]+crop_width/2)
        if left < 0 or top < 0 or right > s[0] or bottom > s[1]:
            return
        crop = self.c_p['image'][left:right, top:bottom]
        results = self.c_p['model'](crop).xywh[0].cpu().numpy()
        for prediction in results:
            if prediction[-1] == 1 and prediction[4] > self.confidence_threshold_pipette:
                new_x = prediction[0] + top
                dx = self.c_p['pipette_location'][0] - new_x
                dy = self.c_p['pipette_location'][3]
                self.c_p['pipette_tilt'] = dx/dy
                self.c_p['accurate_tip_detection_needed'] = False
                return
        print("No pipette tip detected")

    def check_multiple_trapped(self):
        """
        Check if there are multiple particles trapped in the pipette.
        """

        if not self.c_p['particle_trapped'] or not self.c_p['tracking_on']:
            self.c_p['multiple_particles_trapped'] = False
            return

        particle_trapped = self.data_channels['particle_trapped'].get_data(10)
        if False in particle_trapped:
            self.c_p['multiple_particles_trapped'] = False
            return

        xy_force = np.mean(self.data_channels['F_total_X'].get_data(100)[0])**2 + np.mean(self.data_channels['F_total_Y'].get_data(100))**2
        if xy_force > 90: # If there is a large force then this may cause a deviation in z
            self.c_p['multiple_particles_trapped'] = False
            return
        
        # TODO check if we are mooving as well.

        z_predictions = self.data_channels['trapped_particle_z_position'].get_data(10)
        #TODO  Remove prediction outliers
        mean = np.mean(z_predictions)
        #z_predictions = z_predictions[z_predictions < 1]
        self.c_p['multiple_particles_trapped'] = np.abs(mean) > 0.4

    def predict_z_positions(self):
        """
        Function which makes a prediction of the z-positions of the particles located with
        the deep learning model. The z-positions are then stored in the control parameters.

        # TODO add a scaling factor to get something that is similar to true z-positions. Could be model dependent.
        """

        if not self.c_p['tracking_on'] or self.c_p['z-model'] is None:
            return

        # Pre-compute constants
        image_shape = np.shape(self.c_p['image'])
        try:
            # Had problem with Index error here when the camera failed to return an image
            image_width, image_height = image_shape[1], image_shape[0]
        except:
            self.c_p['z-predictions'] = []
            return 
        device = self.c_p['device']
        width = self.c_p['crop_width']

        # List to collect crops
        crops = []

        # TODO It appears as if we mix up the indices which is what is giving the fluctuations in the z-values.
        # Loop through predicted positions to collect crops
        for idx,pos in enumerate(self.c_p['predicted_particle_positions']):
            x, y = int(pos[0]), int(pos[1])
            # Check if the crop is within the image
            if self.c_p['predicted_particle_radii'][idx] > 1.5/self.c_p['microns_per_pix']:
                width = self.c_p['crop_width']
            else:
                width = int(self.c_p['crop_width']-16)
            if 0 <= x - width < x + width <= image_width and 0 <= y - width < y + width <= image_height:
                try:
                    crop = self.c_p['image'][y - width:y + width, x - width:x + width].astype(np.float32)
                    crop /= 2 * np.std(crop)
                    crop -= np.mean(crop)
                    crop = cv2.resize(crop, (128, 128))
                    crop = np.reshape(crop, (128, 128, 1)).astype(np.float32)
                    crops.append(crop)
                except Exception as e:
                    print(e) # Most likely the shape of the image changed during prediction, no worries.
                    pass

        # Convert list of crops to a tensor and prepare for the model
        if crops:  # Check if there are any crops to process
            crops_tensor = torch.tensor(crops, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():  # No gradients needed for inference
                predictions = self.c_p['z-model'](crops_tensor)
            z_vals = predictions.squeeze().tolist()  # Convert predictions to a list
        else:
            z_vals = []
    
        # Convert to list of z-values
        if isinstance(z_vals, float):
            z_vals = np.array([z_vals])
        if len(z_vals) > 0:
            z_vals = np.array(z_vals)
            z_vals -= self.c_p['z-offset']
        self.c_p['z-predictions'] = z_vals

    def compute_gradient_sharpness(self):
        """
        Computes the sharpness of the gradient within a specified region of interest (ROI) in the image.

        Returns:
            sharpness (float): The mean gradient magnitude within the ROI.
                Returns None if there is an error during computation.
        """
        
        pipette_left = int(self.c_p['pipette_location'][0] - self.c_p['pipette_location'][2]/2)
        pipette_right = int(self.c_p['pipette_location'][0] + self.c_p['pipette_location'][2]/2)
        pipette_top = int(self.c_p['pipette_location'][1])
        pipette_bottom = int(self.c_p['pipette_location'][1] + self.c_p['pipette_location'][3])

        # Compute the gradients using the Sobel operator
        # May have problems if the AOI has changed during operation, therefore we need to check the shape of the image.
        # TODO do this with an if statement instead of a try-except
        try:
            image_crop = self.c_p['image'][pipette_top:pipette_bottom, pipette_left:pipette_right]
            grad_x = cv2.Sobel(image_crop, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image_crop, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute the gradient magnitude
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Compute the mean of the gradient magnitudes
            sharpness = np.mean(grad_magnitude)
            
            return sharpness
        except Exception as e:
            return None

    def check_trapped(self, threshold=8_000):
        """
        Check if a particle is trapped in the pipette based on the predicted particle positions.

        Args:
            threshold (int): The threshold distance for determining if a particle is trapped.

        Returns:
            bool: True if a particle is trapped, False otherwise.
        """
        if len(self.c_p['predicted_particle_positions']) < 1:
            self.particles_in_view = False
            self.c_p['particle_trapped'] = False
            return False
        self.particles_in_view = True

        LX = self.c_p['laser_position'][0] - self.c_p['AOI'][0]
        LY = self.c_p['laser_position'][1] - self.c_p['AOI'][2]
        self.c_p['Trapped_particle_position'][0:2], idx = self.find_closest_particle([LX, LY],True)
        self.c_p['Trapped_particle_position'][3] = self.c_p['predicted_particle_radii'][idx]

        # Check if we can get also the z-position, different units tough. Set to None if no z-position found.
        if self.c_p['z-tracking']:
            try:
                self.c_p['Trapped_particle_position'][2] = self.c_p['z-predictions'][idx]
                self.trapped_bead_z.append(self.c_p['Trapped_particle_position'][2])
                self.data_channels['trapped_particle_z_position'].put_data(self.c_p['Trapped_particle_position'][2])
            except IndexError as ie:
                # print("Index error in z-predictions", ie)
                self.c_p['Trapped_particle_position'][2] = None
        else:
            self.c_p['Trapped_particle_position'][2] = None

        trap_dist = (self.c_p['Trapped_particle_position'][0]-LX)**2 + (self.c_p['Trapped_particle_position'][1]-LY)**2
        self.c_p['particle_trapped'] = trap_dist < threshold
    
    def check_in_pipette(self, threshold=2_000, offset=np.array([0, 0])):
        if not self.c_p['pipette_located'] or not self.c_p['tracking_on'] or self.c_p['pipette_location'] is None:
            self.data_channels['particle_in_pipette'].put_data(0)
            return
        try:
             # Take into account that particle is above the pipette
            potential_pos, idx = self.find_closest_particle(np.array(self.c_p['pipette_tip_location']) - offset, True)
        except TypeError as te:
            print("Error in finding closest particle to pipette", te)
            return
        if potential_pos is None:
            self.c_p['particle_in_pipette'] = False
            self.data_channels['particle_in_pipette'].put_data(1)
            return
        
        # Use particle redii to determine position, check that it has not been updated erroneously.
        
        radii = 1.5/self.c_p['microns_per_pix'] # TODO change to radii from the model
        if (self.c_p['pipette_tip_location'][0] - potential_pos[0])**2/4 + ((self.c_p['pipette_tip_location'][1]-radii) - potential_pos[1])**2 < threshold:
            self.c_p['particle_in_pipette'] = True
            self.data_channels['particle_in_pipette'].put_data(2)
            self.c_p['pipette_particle_location'][0:2] = potential_pos
            self.c_p['pipette_particle_location'][3] = self.c_p['predicted_particle_radii'][idx]
            try:
                if self.c_p['z-tracking']:
                    self.c_p['pipette_particle_location'][2] = self.c_p['z-predictions'][idx]
                    self.pipette_bead_z.append(self.c_p['pipette_particle_location'][2])
                else:
                    self.c_p['pipette_particle_location'][2] = None
            except IndexError as ie:
                self.c_p['pipette_particle_location'][2] = None

        else:
            self.data_channels['particle_in_pipette'].put_data(1)

            self.c_p['particle_in_pipette'] = False

    def move_while_avoiding_particlesV2(self, target_position):
        """
        Function that moves the stage while avoiding double trapping particles. 
        """
        # TODO rewrite this to use the predicted particle radii and the predicted pipette shape.

        if not self.data_channels['particle_trapped'].get_data(1)[0]:
            # No particle trapped so no point in avoiding particles.
            print("No particle trapped so no point in avoiding particles.")
            return True
        # Calculate the distance to the target position
        x0 = self.data_channels['Motor_x_pos'].get_data(1)[0]
        y0 = self.data_channels['Motor_y_pos'].get_data(1)[0]
        dx = target_position[0] - x0
        dy = target_position[1] - y0

        move_lim = 100 # Max movemement per "step" if no particles are in view

        if dx**2+dy**2 < 200:
            #self.c_p['move_to_location'] = False
            return True

        if len( self.c_p['predicted_particle_positions']) == 1 and not self.c_p['pipette_located']:
            # Only one particle in view (and no structures) so no point in avoiding particles.
            if dx < -move_lim:
                dx =- move_lim
            elif dx > move_lim:
                dx = move_lim
            if dy < -move_lim:
                dy =- move_lim
            elif dy > move_lim:
                dy = move_lim
            self.c_p['minitweezers_target_pos'][0] = int(x0 + dx)
            self.c_p['minitweezers_target_pos'][1] = int(y0 + dy)
            print("Updating target position to ", self.c_p['minitweezers_target_pos'])
            self.c_p['move_to_location'] = True
            return

        radii = 8 # don't let the particles come closer than 8 microns center to center
        image_shape = np.shape(self.c_p['image']) # TODO may give error if changing size of AOI
        size = int(image_shape[0]*self.c_p['microns_per_pix']) # The size in microns of the image is typically a reasonable size to use in the a-star algorithm

        # Check that we don't do anything stupid with the size
        if size <= 1:
            return False
 
        # Calculate a distance matrix for positions of particles that are not the trapped one
        LX = self.c_p['laser_position'][0]
        LY = self.c_p['laser_position'][1]
        
        # Pop the trapped particle to remove it from the area check.
        distances_squared = [(x-LX)**2+(y-LY)**2 for x,y in self.c_p['predicted_particle_positions']]
        positions = np.copy(self.c_p['predicted_particle_positions'])
        positions = np.delete(positions, np.argmin(distances_squared), axis=0)

        # Also avoid the pipette if there is one in view.
        rect = None
        if self.c_p['pipette_located']:
            # There is a pipette in view, convert from x,y,w,h to x0,y0,x1,y1
            p_x0 = (self.c_p['pipette_location'][0] - self.c_p['pipette_location'][2]/2)*self.c_p['microns_per_pix']
            p_y0 = (self.c_p['pipette_location'][1])*self.c_p['microns_per_pix']
            p_x1 = (self.c_p['pipette_location'][0] + self.c_p['pipette_location'][2]/2)*self.c_p['microns_per_pix']
            p_y1 = -1
            rect = np.array((p_y0, p_x0, p_y1, p_x1)).astype(int)

        area = generate_move_map(size, image_shape[0], image_shape[1], positions, radii, rect, None)

        s = np.shape(area)
        # Assume that we start with particle in middle, changed from middle to trapped particle position
        
        start = (int(LY*self.c_p['microns_per_pix']), int(LX*self.c_p['microns_per_pix']))
        
        # Convert both target position and current laser position to the area coordinates(microns)
        # If the target is outside the FOW(field of view9 it is instead set to the edge of the FOW by the a-star algorithm.
        x_end = int(LX*self.c_p['microns_per_pix'] + self.c_p['microns_per_tick']*dx)
        y_end = int(LY*self.c_p['microns_per_pix'] + self.c_p['microns_per_tick']*dy)

        # Find the path
        start = tuple(start)
        end_pos = tuple((y_end, x_end)) # x and y seem switched in the image
        path = a_star(area, start, end_pos)

        if path is None:
            # TODO this means that we should move back and search for a new path.
            print("No path found"," searched  path from ", start, " to ", end_pos)
            #print(area)
            #print(rect,size, image_shape, positions)
            #import matplotlib.pyplot as plt
            #plt.imshow(area)
            #plt.show()
            return 
        path = simplify_path(path)
        print("Finding path from ", start, " to ", end_pos, "path is ", path)
        # print(rect,size, image_shape, positions)

        # Convert target position to motor ticks
        # factor = image_shape[0] / size * self.c_p['ticks_per_pixel']
        #self.c_p['minitweezers_target_pos'][0] = int(x0 + (path[1][0]-start[0])*self.c_p['ticks_per_micron'])
        #self.c_p['minitweezers_target_pos'][1] = int(y0 + (path[1][1]-start[1])*self.c_p['ticks_per_micron'])

        # Need to handle the case when the particle appears to be inside the pipette(from the algorithms point of view.)
        self.c_p['minitweezers_target_pos'][0] = int(x0 + (path[1][1]-start[1])*self.c_p['ticks_per_micron'])*0.9
        self.c_p['minitweezers_target_pos'][1] = int(y0 + (path[1][0]-start[0])*self.c_p['ticks_per_micron'])
        print("Smart move to ", self.c_p['minitweezers_target_pos'])
        self.c_p['move_to_location'] = True

    def center_particle(self, center, move_limit=10):
        # Find particle closest to the center

        center_particle = self.find_closest_particle(center, False)
        if center_particle is None:
            return

        # Calculate the distance to move the stage
        dx = center[1] - center_particle[0]
        dy = center[0] - center_particle[1]
        if np.abs(dx)<move_limit and np.abs(dy)<move_limit:
            print(f"Limits {dx} {dy}")
            return
        # Tell motors to move
        # NOTE the minus sign here is because the camera is mounted upside down
        self.c_p['stepper_target_position'][0] = self.c_p['stepper_current_position'][0] - dx * self.c_p['microns_per_pix']
        self.c_p['stepper_target_position'][1] = self.c_p['stepper_current_position'][1] - dy * self.c_p['microns_per_pix']

    def move_piezos_2_saved_positions(self, positions, threshold=5e-4):
        """
        Moves the piezos to a saved position.
        Usefull for resetting after having done a trapping
        Returns False if the piezos are not at the target position.
        Returns True if the piezos are at the target position or if they cannot be moved further.
        """

        # Check if we are autoalignin any of the piezos.
        if self.c_p['portenta_command_2'] != 0:
            return True

        current_a_x = np.mean(self.data_channels['PSD_A_P_X'].get_data(1000)[0])
        current_a_y = np.mean(self.data_channels['PSD_A_P_Y'].get_data(1000)[0])
        current_a_sum = np.mean(self.data_channels['PSD_A_P_sum'].get_data(1000)[0])

        current_b_x = np.mean(self.data_channels['PSD_B_P_X'].get_data(1000)[0])
        current_b_y = np.mean(self.data_channels['PSD_B_P_Y'].get_data(1000)[0])
        current_b_sum = np.mean(self.data_channels['PSD_B_P_sum'].get_data(1000)[0])

        # Check that we are not auto-aliging piezo A, then move piezo a towards target position
                # Have had some problems with noise on some of the sum-channels, this is a quick fix for that.
        if current_a_sum < 0:
            current_a_sum = 20_000
        if current_b_sum < 0:
            current_b_sum = 20_000
    
        dax = (positions[0] - current_a_x)/current_a_sum # Will give a number roughly in the range -0.5 to 0.5
        day = (positions[1] - current_a_y)/current_a_sum
        dbx = (positions[2] - current_b_x)/current_b_sum
        dby = (positions[3] - current_b_y)/current_b_sum
        

        if np.abs(dax)<threshold and np.abs(day)<threshold and np.abs(dbx)<threshold and np.abs(dby)<threshold:
            return True
        prefac = 30_000 # Moving the whole distance was unstable at some points.
        ax_n = convert2uint16(self.data_channels['dac_ax'].get_data(1)[0] + dax*prefac)
        if ax_n == 65535 or ax_n == 0:
            return True
        self.c_p['piezo_A'][0] = ax_n
        ay_n = convert2uint16(self.data_channels['dac_ay'].get_data(1)[0] + day*prefac)
        if ay_n == 65535 or ay_n == 0:
            return True
        self.c_p['piezo_A'][1] = ay_n

        bx_n = convert2uint16(self.data_channels['dac_bx'].get_data(1)[0] + dbx*prefac)
        if bx_n == 65535 or bx_n == 0:
            return True
        self.c_p['piezo_B'][0] = bx_n
        by_n = convert2uint16(self.data_channels['dac_by'].get_data(1)[0] + dby*prefac)
        if by_n == 65535 or by_n == 0:
            return True
        self.c_p['piezo_B'][1] = by_n

        return False

    def drop_particle(self):
        """
        Function that drops the trapped particle(s).
        Used when multiple particles are in the trap but you really only want one.

        Moves away one of the lasers so that the particle is no longer trapped.
        Moving both lasers risks leaving the particle in the same position so that it is trapped again when the lasers are reset.
        """
        if not self.c_p['particle_trapped']:
            return True
        # Turn off auto-aligning
        if self.c_p['portenta_command_2'] == 1:
            self.c_p['portenta_command_2'] = 0
            if self.data_channels['dac_ax'].get_data(1)[0] > 32768:
                x = 2000
            else:
                x = 60000
            if self.data_channels['dac_ay'].get_data(1)[0] > 32768:
                y = 2000
            else:
                y = 60000
            self.c_p['piezo_A'] = np.uint16([x, y])
        else:
            self.c_p['portenta_command_2'] = 0
            if self.data_channels['dac_bx'].get_data(1)[0] > 32768:
                x = 2000
            else:
                x = 60000
            if self.data_channels['dac_by'].get_data(1)[0] > 32768:
                y = 2000
            else:
                y = 60000
            self.c_p['piezo_B'] = np.uint16([x, y])

        # move the piezos to a bad position
        
        return False

    def drop_particle_and_reset_trap(self):
        if self.drop_particle():
            self.c_p['move_piezo_2_target'] = True
            return True
        return False

    def is_point_in_rectangle(self, point, top_left, bottom_right):
        """
        Determine if the point (px, py) is inside the rectangle defined by top-left and bottom-right coordinates.
        """
        px, py = point
        tlx, tly = top_left
        brx, bry = bottom_right
        return tlx <= px <= brx and tly <= py <= bry
    
    def check_close2Pipette(self, offset=40):
        """
        Checks the predicted particle position for particles that are far away enough from the pipette to not be stuck on it
        and use this as possible particles to trap.
        """
        if not (self.c_p['locate_pipette'] and self.c_p['pipette_located']):
            #  No pipette in view, we can safely return all particles.
            return self.c_p['predicted_particle_positions']

        # Extract the full extent of the pipette, particles really close to it are not interesting to trap since they are likely to be stuck.
        top_left = self.c_p['pipette_location'][:2]
        top_left[0] -= self.c_p['pipette_location'][2]/2
        top_left[1] -= offset
        bottom_right = self.c_p['pipette_location'][:2]
        bottom_right[0] += self.c_p['pipette_location'][2]/2
        bottom_right[1] = self.c_p['AOI'][3] # Bottom location of AOI
        particle_positions = []

        for pos in self.c_p['predicted_particle_positions']:
            if not self.is_point_in_rectangle(pos, top_left, bottom_right):
                particle_positions.append(pos)
        self.c_p['tmp_predictions'] = np.array(particle_positions)

        return np.array(particle_positions)

    def trap_particle_minitweezers(self, center, move_limit=30):
       
        if not self.c_p['tracking_on']:
            print("Cannot center particle without tracking on")
            return False
        
        center_particle_position, index = self.find_closest_particle(center, True, self.check_close2Pipette())

        # Check that we are not moving and that we are not moving and that there is a particle to trap.
        if center_particle_position is None: #  or self.c_p['move_to_location']: Changed so that it moves immediately and ignores if there is already a move in progress.
            #sleep(0.1)
            return False
        
        dx = (center_particle_position[0] - center[0]) * self.c_p['ticks_per_pixel']
        dy = (center_particle_position[1] - center[1]) * self.c_p['ticks_per_pixel']
        dz = 0
        target_z_pos = self.c_p['minitweezers_target_pos'][2]
        
        # If we are to include the z-position we need to ensure that we have the same index for all the predictions(that it does not change after doing the check-close2 pipette)
        if self.c_p['z-tracking'] and index<len(self.c_p['z-predictions']) and not self.c_p['pipette_located']:
            dz =  -4 * self.c_p['z-predictions'][index]
            target_z_pos = int(self.data_channels['Motor_z_pos'].get_data(1)[0] + dz)
        
        if np.abs(dx) < move_limit and np.abs(dy) < move_limit and np.abs(dz) < move_limit/3:
            print(f"Limits {dx} {dy} {dz}")
            return True

        extra_move = 1 # Moving a little less than needed
        target_x_pos = int(self.data_channels['Motor_x_pos'].get_data(1)[0] + dx*extra_move)
        target_y_pos = int(self.data_channels['Motor_y_pos'].get_data(1)[0] - dy*extra_move) # Offest since we don't want to collide with the pipette.

        self.c_p['minitweezers_target_pos'] = [target_x_pos, target_y_pos, target_z_pos]
        self.c_p['move_to_location'] = True

    def trap_particle_minitweezers_alt(self, center, move_limit=30):
        # Also break when particle is trapped.
        # Center is the laser position on which to center the particle.
       
        if not self.c_p['tracking_on']:
            print("Cannot center particle without tracking on")
            self.c_p['motor_x_target_speed'] = 0
            self.c_p['motor_y_target_speed'] = 0
            self.c_p['motor_z_target_speed'] = 0
            return False
        
        center_particle_position, index = self.find_closest_particle(center, True, self.check_close2Pipette())

        # Check that we are not moving and that we are not moving and that there is a particle to trap.
        if center_particle_position is None: #  or self.c_p['move_to_location']: Changed so that it moves immediately and ignores if there is already a move in progress.
            #sleep(0.1)
            #if self.moving2trap:
            self.c_p['motor_x_target_speed'] = 0
            self.c_p['motor_y_target_speed'] = 0
            self.c_p['motor_z_target_speed'] = 0
            self.moving2trap = False
            return False
        
        dx = (center[0] - center_particle_position[0]) * self.c_p['ticks_per_pixel']
        dy = (center[1] - center_particle_position[1]) * self.c_p['ticks_per_pixel']
        dz = 0
        target_z_pos = self.c_p['minitweezers_target_pos'][2]
        
        # If we are to include the z-position we need to ensure that we have the same index for all the predictions(that it does not change after doing the check-close2 pipette)
        if self.c_p['z-tracking'] and index<len(self.c_p['z-predictions']) and not self.c_p['pipette_located']:
            dz =  10 * self.c_p['z-predictions'][index]
            target_z_pos = int(self.data_channels['Motor_z_pos'].get_data(1)[0] + dz)
        
        if self.c_p['particle_trapped']:
            self.c_p['motor_x_target_speed'] = 0
            self.c_p['motor_y_target_speed'] = 0
            self.c_p['motor_z_target_speed'] = 0
            self.moving2trap = False
            return True
        #speed == [10_000 if ]
        fac = 25_000 / ((dx**2 + dy**2)**0.5)
        if self.c_p['mouse_params'][0] == 0:
            self.c_p['motor_x_target_speed'] = int(dx*fac)
            self.c_p['motor_y_target_speed'] = int(dy*fac)
            self.movin2trap = True
        

    def find_and_trap_particle(self, capillary=1):
        """
        Function that moves to the predefined capillary area and tries to retrieve a particle from there.

        TODO Should ideally also be able to distinguish one particle from many when trapping and also be able to 
        distinguish between the different particle types.
        """

        fludics_channel = 'capillary_1_fluidics_channel' if capillary == 1 else 'capillary_2_fluidics_channel'
        # If already moving, return
        if self.c_p['move_to_location']:
            self.c_p['target_pressures'][self.c_p[fludics_channel][0]] = 0
            return False
        
        x = self.data_channels['Motor_x_pos'].get_data(1)[0]
        y = self.data_channels['Motor_y_pos'].get_data(1)[0]

        # If a particle is trapped, go 2 the pipette
        if self.c_p['particle_trapped']:
            # If we are already at the pipette with a trapped particle, return true
            if (x-self.c_p['pipette_location_chamber'][0])**2+(y-self.c_p['pipette_location_chamber'][1])**2 < 1000:
                self.c_p['target_pressures'][self.c_p[fludics_channel][0]] = 0
                return True
            # We are not at the pipette but we have a particle so we should move 2 the pipette
            go2position(self.c_p['pipette_location_chamber'], self.c_p)
            return False
        
        # No particle trapped and not moving, if we are far from the capillary then we should move to the capillary
        capillary_position = self.c_p['capillary_1_position'] if capillary == 1 else self.c_p['capillary_2_position']
        if (x-capillary_position[0])**2+(y-capillary_position[1])**2 > 15_000:
            # Don't generally want to change the z focus when moving automatically
            capillary_position[2] = self.data_channels['Motor_z_pos'].get_data(1)[0]
            go2position(capillary_position, self.c_p)
            return False
        
        # If we are close to the capillary, we should center the particle, we do this by toggling the correct trapping loop
        self.c_p['centering_on'] = True

        # Should also push out some particles. We are in position and trying to trap
        self.c_p['target_pressures'][self.c_p[fludics_channel][0]] = self.c_p[fludics_channel][1]
        return False

    def find_movement_direction(self, margin=10):
        if not self.c_p['move_to_location']:
            return 0
        dx = self.data_channels['Motor_x_pos'].get_data(1)[0] - self.c_p['minitweezers_target_pos'][0]
        dy = self.data_channels['Motor_y_pos'].get_data(1)[0] - self.c_p['minitweezers_target_pos'][1]

        if dx**2<margin**2 and dy**2<margin**2:
            return 0
        
        if dx>=margin and dy>=margin:
            return 1
        elif dx>=margin and -margin<=dy<=margin:
            return 2
        elif dx>=margin and dy<=-margin:
            return 3
        elif -margin<=dx<=margin and dy<=-margin:
            return 4
        elif dx<=-margin and dy<=-margin:
            return 5
        elif dx<=-margin and -margin<=dy<=margin:
            return 6
        elif dx<=-margin and dy>=margin:
            return 7
        elif -margin<=dx<=margin and dy>=margin:
            return 8

    def check_if_path_clear(self, margin=10):
        dir = self.find_movement_direction(margin=margin)
        if dir == 0 or len(self.c_p['predicted_particle_positions'])<2: # At most one particle (the trapped one) in view
            return True
        
        # Calculate a distance matrix for positions of particles that are not the trapped one
        LX = self.c_p['laser_position'][0]
        LY = self.c_p['laser_position'][1]
        distances = [[(x-LX),(y-LY)] for x,y in self.c_p['predicted_particle_positions']]
        
        # Pop the trapped particle to remove it from the check.
        distances_squared = [(x-LX)**2+(y-LY)**2 for x,y in self.c_p['predicted_particle_positions']]
        distances.pop(np.argmin(distances_squared))

    def initiate_electrostatic_protocol(self):
        self.EP_positions = np.linspace(self.c_p['electrostatic_protocol_start'],
                                        self.c_p['electrostatic_protocol_end'],
                                        self.c_p['electrostatic_protocol_steps'],
                                        dtype=int)
        self.c_p['electrostatic_protocol_running'] = True
        self.c_p['electrostatic_protocol_steps'] = 0
        self.c_p['electrostatic_protocol_finished'] = False
        self.measurement_start_time = time()
 
    def custom_electrostatic_protocol(self):

        if not self.c_p['electrostatic_protocol_running']:
            print("Initiating protocol")
            self.initiate_electrostatic_protocol()

        # Check if measurement needs to be updated
        if time() - self.measurement_start_time < self.c_p['electrostatic_protocol_duration']:
            return
        
        # Update the piezo position
        self.c_p['piezo_B'][1] = self.EP_positions[self.c_p['electrostatic_protocol_steps']]

        print(self.EP_positions[self.c_p['electrostatic_protocol_steps']])

        # Toggle autoalign A for a short time
        self.c_p['portenta_command_2'] = 1
        sleep(0.5)
        # Need to reset the piezo position to the autoaligned one
        self.c_p['piezo_A'] = np.int32([np.mean(self.data_channels['dac_ax'].get_data_spaced(10)),
                                np.mean(self.data_channels['dac_ay'].get_data_spaced(10))])
        self.c_p['portenta_command_2'] = 0

        self.c_p['electrostatic_protocol_steps'] += 1
        if self.c_p['electrostatic_protocol_steps'] >= len(self.EP_positions):
            self.c_p['electrostatic_protocol_running'] = False
            self.c_p['electrostatic_protocol_finished'] = True
            # TODO make it turn of automatically
            return
        # Update measurement start-time
        self.measurement_start_time = time()
        
    def find_true_laser_position(self):

        # TODO update approximation to use also the reading of the position PSDs
        if not self.check_trapped() or not self.c_p['tracking_on']:
            print("No particles detected close enough to current laser position")
            return
        LX = self.c_p['laser_position'][0] - self.c_p['AOI'][0]
        LY = self.c_p['laser_position'][1] - self.c_p['AOI'][2]

        min_pos = self.find_closest_particle([LX, LY])

        self.c_p['laser_position'] = [min_pos[0]+self.c_p['AOI'][0], min_pos[1]+self.c_p['AOI'][2]]
        print("Laser position set to: ", self.c_p['laser_position'])

    def find_closest_particle(self, reference_position, return_idx, particle_positions=None):

        if particle_positions is None:
            particle_positions = self.c_p['predicted_particle_positions']

        if len(particle_positions) == 0:
            if return_idx:
                return None, None
            return None
        try:
            LX = reference_position[0]
            LY = reference_position[1]
            min_x = 1000
            min_dist = 2e10
            min_y = 1000
            idx = 0
            min_idx = 0
            for x,y in particle_positions:

                if (x-LX)**2+(y-LY)**2<min_dist:
                    min_dist = (x-LX)**2+(y-LY)**2
                    min_x = x
                    min_y = y
                    min_idx = idx
                idx += 1
            if return_idx:
                return [min_x, min_y], min_idx
            return [min_x, min_y]
        except Exception as e:
            if return_idx:
                return None, None
            return None
        
    def move2area_above_pipette_check(self, move2particle=True):
        # Calculate the distance to the target position

        if move2particle:
            dx = self.c_p['pipette_particle_location'][0] - self.c_p['Trapped_particle_position'][0]  
            dy = (self.c_p['pipette_particle_location'][1] - self.offset_y) - self.c_p['Trapped_particle_position'][1]    
        else:
            dx = self.c_p['pipette_tip_location'][0] - self.c_p['Trapped_particle_position'][0]  
            dy = (self.c_p['pipette_tip_location'][1] - self.offset_y) - self.c_p['Trapped_particle_position'][1]

        # IF we are close enough to the target position then we return.
        if dx**2 + dy**2 < 1200:
            print("In position")
            self.move_finsih = True
            return True
        return False
        
    def move2area_above_pipette(self, move2particle=True,piezo_y_target=22000):
        """
        Moves the motors to be centered above the particle in the pipette, or alternatively above the pipette tip.
        """
        if not self.c_p['pipette_located']:# or self.c_p['move_to_location']:
            # print("No pipette visible", self.c_p['move_to_location']) # TODO seems like this could be the issue, i.e that it does not actually move even when it thinks it's moving.
            return False
        
        # We put the particle quite far up to make it esier to touch the particles after having moved to the pipette. But only if we are autoaligning.
        #piezo_y_target = 22000 # Inreased a little to get more movement in the piezo.
        piezo_y_target = int(piezo_y_target)
        piezo_x_target = 32768
        if self.c_p['portenta_command_2'] == 0:
            self.c_p['portenta_command_2'] = 1
            return False

        if self.c_p['portenta_command_2'] == 1:
            current_y = self.data_channels['dac_by'].get_data(1)[0]
            if current_y - piezo_y_target > 0:
                self.c_p['piezo_B'] = [int(piezo_x_target), int(current_y - 3000)]
                return False
            self.c_p['piezo_B'] = [32768, piezo_y_target]
        elif self.c_p['portenta_command_2'] == 2:
            current_y = self.data_channels['dac_ay'].get_data(1)[0]
            if current_y - piezo_y_target > 0:
                self.c_p['piezo_A'] = [int(piezo_x_target), int(current_y - 3000)]
                return False
            self.c_p['piezo_A'] = [32768, piezo_y_target]
        
        
        # Calculate the distance to the taret position
        if move2particle:
            dx = self.c_p['pipette_particle_location'][0] - self.c_p['Trapped_particle_position'][0]  
            dy = (self.c_p['pipette_particle_location'][1] - self.offset_y) - self.c_p['Trapped_particle_position'][1]    
        else:
            dx = self.c_p['pipette_tip_location'][0] - self.c_p['Trapped_particle_position'][0]  
            dy = (self.c_p['pipette_tip_location'][1] - self.offset_y) - self.c_p['Trapped_particle_position'][1]
        """
        # IF we are close enough to the target position then we return.
        if dx**2+dy**2<1200:
            print("In position")
            self.move_finsih = True
            return True
        """
        # Had to move this to be able to do the check also from the main experiment thread.
        if self.move2area_above_pipette_check(move2particle):
            return True

        # Give some time between the moves(1 second)
        if time() - self.last_move_time < 0.4:
            return False

        # Move the stage
        num = 0.9 # Factor to not move to far at a time
        current_x = self.data_channels['Motor_x_pos'].get_data(1)[0]
        current_y = self.data_channels['Motor_y_pos'].get_data(1)[0]
        self.c_p['minitweezers_target_pos'][1] = int(current_y - dy*self.c_p['ticks_per_pixel']*num)
        if (dy*self.c_p['ticks_per_pixel'])**2 < 200:
            self.c_p['minitweezers_target_pos'][0] = int(current_x + dx*self.c_p['ticks_per_pixel']*num)
        # print(self.c_p['minitweezers_target_pos'], dx, dy)
        self.last_move_time = time()
        # TODO test the additional check introduced here to see if they fixed the issues with the move2areaabove not turning off properly.
        if np.abs(current_x-self.c_p['minitweezers_target_pos'][0]) >= 4 or np.abs(current_y-self.c_p['minitweezers_target_pos'][1]) >= 4:
            self.c_p['move_to_location'] = True
            return False
        if self.c_p['move_to_location']:
            return False
        return True
    
    def find_max_sharpness(self, starting_guess, max_sharpness, direction,range=200, step=5):
        """
        Function that finds the maximum sharpness in a given direction from a starting position.
        """
        #self.c_p['minitweezers_target_pos'] = starting_guess
        sharpness = self.compute_gradient_sharpness()

        if sharpness is not None and sharpness>0.85*max_sharpness:
            return True
        print(sharpness)


        z_pos =self.data_channels['Motor_z_pos'].get_data(1)[0]

        # Check that we don't move too far
        if direction>0:
            if z_pos > starting_guess + range:
                print("No sharpness found, moving to starting position")
                self.c_p['minitweezers_target_pos'][2] = starting_guess
                self.c_p['move_to_location'] = True
                return True
        else:
            if z_pos < starting_guess - range:
                print("No sharpness found, moving to starting position")
                self.c_p['minitweezers_target_pos'][2] = True
                self.c_p['move_to_location'] = True
                return True

        # Take a step in the correcet direction
        if not self.c_p['move_to_location']:
            self.c_p['minitweezers_target_pos'][2] = int(z_pos + step*direction)
            self.c_p['move_to_location'] = True
        return False



    def z_focus_pipette(self, range=200, step=10, direction=1):

        if self.c_p['pipette_focus_startup']:
            self.start_z = self.data_channels['Motor_z_pos'].get_data(1)[0]
            self.c_p['minitweezers_target_pos'][2] = int(self.start_z + range*direction)
            self.c_p['move_to_location'] = True
            self.c_p['pipette_focus_startup'] = False
            self.c_p['pipette_sharpnesses'] = []
            self.c_p['pipette_sharpness_Z_pos'] = []
            self.sharpnesses_mapped = False
            return False
        if self.c_p['move_to_location']:
            return False
        in_pipette = np.count_nonzero(self.data_channels['particle_in_pipette'].get_data(10) == 2) > 8

        if in_pipette:
            if self.c_p['pipette_particle_location'][2] is not None and np.abs(self.c_p['pipette_particle_location'][2])<0.7:
                self.c_p['pipette_focus_startup'] = True
                print("in good focus")
                return True
        self.c_p['pipette_sharpness_Z_pos'].append(self.data_channels['Motor_z_pos'].get_data(1)[0])
        if not self.c_p['pipette_located']:
            self.c_p['pipette_sharpnesses'].append(-1) # Maybe change so that we get -1 if there is no pipette
        else:
            sharpness = self.compute_gradient_sharpness()
            if sharpness is not None:
                self.c_p['pipette_sharpnesses'].append(sharpness) # Maybe change so that we get -1 if there is no pipette
            else:
                self.c_p['pipette_sharpnesses'].append(-1)
            print(sharpness)

        """
        if len(self.c_p['pipette_sharpnesses'])>5:
            if self.c_p['pipette_sharpnesses'][-1] < self.c_p['pipette_sharpnesses'][-2] and self.c_p['pipette_sharpnesses'][-2] < self.c_p['pipette_sharpnesses'][-3]:
                if np.max(self.c_p['pipette_sharpnesses']) > np.min(self.c_p['pipette_sharpnesses'])*1.2:
                    target_pos = int(self.data_channels['Motor_z_pos'].get_data(1)[0] + 4*step*direction)
                    print(f"Sharpness is decreasing, moving back to {target_pos}")
                    self.c_p['minitweezers_target_pos'][2] = target_pos
                    self.c_p['pipette_focus_startup'] = True
                    self.c_p['move_to_location'] = True
                    return True
        """
        #if len(self.c_p['pipette_sharpnesses'])<range*2/step:
        if direction > 0 and self.c_p['pipette_sharpness_Z_pos'][-1] > self.start_z - range:
            if not self.sharpnesses_mapped:
                self.c_p['minitweezers_target_pos'][2] = int(self.data_channels['Motor_z_pos'].get_data(1)[0] - step)
                self.c_p['move_to_location'] = True
                return False
        if direction < 0 and self.c_p['pipette_sharpness_Z_pos'][-1] < self.start_z + range:
            if not self.sharpnesses_mapped:
                self.c_p['minitweezers_target_pos'][2] = int(self.data_channels['Motor_z_pos'].get_data(1)[0] + step)
                self.c_p['move_to_location'] = True
                return False
            
        if not self.sharpnesses_mapped:
            self.c_p['pipette_sharpnesses'][self.c_p['pipette_sharpnesses']==None] = -1
            best_position = self.c_p['pipette_sharpness_Z_pos'][np.argmax(self.c_p['pipette_sharpnesses'])]
            self.sharpnesses_mapped = True
            self.c_p['minitweezers_target_pos'][2] = int(best_position)
            self.c_p['move_to_location'] = True
            return False
        best_position = self.c_p['pipette_sharpness_Z_pos'][np.argmax(self.c_p['pipette_sharpnesses'])]
        if self.find_max_sharpness(int(best_position+direction*step*5), np.max(self.c_p['pipette_sharpnesses']), direction): # TODO don't move the full distance
            print("Finding max sharpness")
            self.c_p['pipette_focus_startup'] = True
            return True
        #self.c_p['minitweezers_target_pos'][2] = int(best_position)
        #self.c_p['move_to_location'] = True
        #self.c_p['pipette_focus_startup'] = True
        #self.c_p['pipette_sharpnesses'] = []
        #self.c_p['pipette_sharpness_Z_pos'] = []
        return False

    def z_focus_pipette_old(self, range=200, step=10, direction=-1):

        if self.c_p['pipette_focus_startup']:
            self.start_z = self.data_channels['Motor_z_pos'].get_data(1)[0]
            self.c_p['minitweezers_target_pos'][2] = int(self.start_z + range*direction)
            self.c_p['move_to_location'] = True
            self.c_p['pipette_focus_startup'] = False
            self.c_p['pipette_sharpnesses'] = []
            self.c_p['pipette_sharpness_Z_pos'] = []
            return False
        if self.c_p['move_to_location']:
            return False
        
        self.c_p['pipette_sharpness_Z_pos'].append(self.data_channels['Motor_z_pos'].get_data(1)[0])
        if not self.c_p['pipette_located']:
            self.c_p['pipette_sharpnesses'].append(-1) # Maybe change so that we get -1 if there is no pipette
        else:
            sharpness = self.compute_gradient_sharpness()
            if sharpness is not None:
                self.c_p['pipette_sharpnesses'].append(sharpness) # Maybe change so that we get -1 if there is no pipette
            else:
                self.c_p['pipette_sharpnesses'].append(-1)
        
        if direction > 0 and self.c_p['pipette_sharpness_Z_pos'][-1] > self.start_z - range:
            self.c_p['minitweezers_target_pos'][2] = int(self.data_channels['Motor_z_pos'].get_data(1)[0] - step)
            self.c_p['move_to_location'] = True
            return False
        if direction < 0 and self.c_p['pipette_sharpness_Z_pos'][-1] < self.start_z + range:
            self.c_p['minitweezers_target_pos'][2] = int(self.data_channels['Motor_z_pos'].get_data(1)[0] + step)
            self.c_p['move_to_location'] = True
            return False
        self.c_p['pipette_sharpnesses'][self.c_p['pipette_sharpnesses']==None] = -1
        best_position = self.c_p['pipette_sharpness_Z_pos'][np.argmax(self.c_p['pipette_sharpnesses'])]
        self.c_p['minitweezers_target_pos'][2] = int(best_position)
        self.c_p['move_to_location'] = True
        self.c_p['pipette_focus_startup'] = True
        self.c_p['pipette_sharpnesses'] = []
        self.c_p['pipette_sharpness_Z_pos'] = []
        return True

   

    def z_focus(self, limit=3):
        """
        Operational idea:
            The system compares the position of the particle in the trap
            to the particle in the pipette. For this we need to know the approximate laser position
            as well as the position of the pipette. The laser position should be ...
        
        return:
            True if the system has reached the correct z-position
            False if the system needs to move further or no z-position was found.
        """

        # DO ininital checks to ensure we are set-up for this
        if not self.c_p['tracking_on'] and self.c_p['z-tracking']:
            return False
        if len(self.c_p['predicted_particle_positions']) == 0:
            return False
        
        # Find z-position of particle in pipette
        if self.c_p['Trapped_particle_position'][2] is None or self.c_p['pipette_particle_location'][2] is None:
            return False
        dz = -10*(self.c_p['Trapped_particle_position'][2] - self.c_p['pipette_particle_location'][2]) # Puts it to a reasonable scale.
        # Move the stage towards the correct position
        print(f"Time to z-move. {dz},{self.c_p['Trapped_particle_position'][2] },{self.c_p['pipette_particle_location'][2]}")
        if dz**2 < limit:
            print(f"Increasing the z-move counter to: {self.z_move_counter}")
            self.z_move_counter += 1
            if self.z_move_counter>8:
                self.z_move_counter = 0
                print("In good z-position, hooray!")
                return True
            return False
        if dz > 0:
            self.c_p['minitweezers_target_pos'][2] = int(self.data_channels['Motor_z_pos'].get_data(1)[0] + 4) # Cannot move less than this, trying with 3 now intestead of 4.
        if dz < 0:
            self.c_p['minitweezers_target_pos'][2] = int(self.data_channels['Motor_z_pos'].get_data(1)[0] - 4)
        self.z_move_counter = 0
        # Here we risk moving along other axis by accident if we don't set the target position to current position.
        # Does enforce that we only move along one axis(z)
        self.c_p['minitweezers_target_pos'][0] = int(self.data_channels['Motor_x_pos'].get_data(1)[0])
        self.c_p['minitweezers_target_pos'][1] = int(self.data_channels['Motor_y_pos'].get_data(1)[0])

        self.c_p['move_to_location'] = True
        return False        

    def approximate_piezo_moveable_area(self,max_move_microns=12):
        """
        Approximates the area in which the particle can be moved with the piezo.
        """
        max_move_pixels = max_move_microns/self.c_p['microns_per_pix']
        # Extract how far we have moved along the x
        x_piezo = self.data_channels['dac_bx'].get_data(1)[0]
        y_piezo = (self.data_channels['dac_ay'].get_data(1)[0] + self.data_channels['dac_by'].get_data(1)[0])/2
        x_particle = self.c_p['Trapped_particle_position'][0]
        y_particle = self.c_p['Trapped_particle_position'][1]
        x_min = x_particle - ((65536 - x_piezo) / 65536) * max_move_pixels
        x_max = x_particle + (x_piezo / 65536) * max_move_pixels
        y_min = y_particle - (y_piezo / 65536) * max_move_pixels
        y_max = y_particle + ((65536 - y_piezo) / 65536) * max_move_pixels
        return x_min, x_max, y_min, y_max

    def check_limit(self, number):
        if number < 0:
            return 0
        if number > 65535:
            return 65535
        return int(number)

    def move_particle_to_position_with_piezo(self, target_position, piezo="piezo_b", movement_threshold=3, movement_lim=2000):
        """
        Moves the particle with the piezo to the target position.
        Target position should be given in pixels in the image.
        """

        x_min, x_max, y_min, y_max = self.approximate_piezo_moveable_area()
        if not (x_min < target_position[0] < x_max and y_min < target_position[1] < y_max):
            print("Target position outside of moveable area")
            print(target_position, x_min, x_max, y_min, y_max)
            return False

        dx = target_position[0] - self.c_p['Trapped_particle_position'][0]
        dy = target_position[1] - self.c_p['Trapped_particle_position'][1]
        if dx**2+dy**2<(movement_threshold/self.c_p['microns_per_pix'])**2:
            return True

        if piezo == "piezo_b":
            current_piezo_pos_x = self.data_channels['dac_bx'].get_data(1)[0]
            current_piezo_pos_y = self.data_channels['dac_by'].get_data(1)[0]
            if dx > 0:
                new_pos = current_piezo_pos_x - (min(int(dx*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_B'][0] = self.check_limit(new_pos)
            if dx < 0:
                new_pos = current_piezo_pos_x + (min(int(-dx*self.bits_per_pixel), movement_lim)) # Changed to max instead of min
                self.c_p['piezo_B'][0] = self.check_limit(new_pos)
            if dy > 0:
                new_pos = current_piezo_pos_y + (min(int(dy*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_B'][1] = self.check_limit(new_pos)
            if dy < 0:
                new_pos = current_piezo_pos_y - (min(int(-dy*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_B'][1] = self.check_limit(new_pos)
        else:
            current_piezo_pos_x = self.data_channels['dac_ax'].get_data(1)[0]
            current_piezo_pos_y = self.data_channels['dac_ay'].get_data(1)[0]
            if dx > 0:
                new_pos = current_piezo_pos_x + (min(int(dx*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_A'][0] = self.check_limit(new_pos)
            if dx < 0:
                new_pos = current_piezo_pos_x - (min(int(-dx*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_A'][0] = self.check_limit(new_pos)
            if dy > 0:
                new_pos = current_piezo_pos_y + (min(int(dy*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_A'][1] = self.check_limit(new_pos)
            if dy < 0:
                new_pos = current_piezo_pos_y - (min(int(-dy*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_A'][1] = self.check_limit(new_pos)

        return False

    def check_particles_can_touch_with_piezo(self):
        """
        Checks if the particles are close enough to each other to touch when moved with only the piezos(laser).
        """
        if len(self.c_p['predicted_particle_positions']) < 2:
            return False
        x_min, x_max, y_min, y_max = self.approximate_piezo_moveable_area()
        pipette_particle_top = self.c_p['pipette_particle_location'][1] - self.c_p['pipette_particle_location'][3]
        pipette_particle_bottom_reachable = y_max + self.c_p['pipette_particle_location'][3]
        if x_min<self.c_p['pipette_particle_location'][0]<x_max and pipette_particle_bottom_reachable > pipette_particle_top:
            return True
        return False

    def align_trapped_and_pipette_X(self, move_dist=100, limit=0.1):
        """
        Aligns the trapped particle and the pipette in the X direction.

        Args:
            move_dist (int, optional): The distance to move the piezo stage. Defaults to 100.
            limit (float, optional): The limit for the alignment in microns. Defaults to 0.1.

        Returns:
            bool: True if alignment is finished, False otherwise.
        """

        dx = (self.c_p['pipette_particle_location'][0] - self.c_p['Trapped_particle_position'][0]) * self.c_p['microns_per_pix']

        if dx > limit:
            if self.c_p['portenta_command_2'] == 1: # Autoaligning A
                self.c_p['piezo_B'][0] = max(int(self.c_p['piezo_B'][0] - move_dist), 1000)
            else:
                self.c_p['piezo_A'][0] = min(int(self.c_p['piezo_A'][0] + move_dist), 65000)
            return True

        if dx < -limit:
            if self.c_p['portenta_command_2'] == 1:
                self.c_p['piezo_B'][0] = min(int(self.c_p['piezo_B'][0] + move_dist), 65000)
            else:
                self.c_p['piezo_A'][0] = max(int(self.c_p['piezo_A'][0] - move_dist), 1000)
            return True

        return False

    def touch_trapped_2_pieptte_particle(self, force_limit=5, position_limit=0.85, max_dist=6):
        """
        force_limit - the amount of force, in pN which the particles are allowed to be pushed together with.
        position_limit - the distance in particle radii that the particles are allowed to be pushed together with.
        max_dist - the maximum distance in microns that the particles are allowed to be pushed together with.

        returns a desciption of what the function is doing or why it is not doing anything.
        """
        
        if not self.c_p['particle_in_pipette'] or not self.c_p['particle_trapped']:
            print("No particle in pipette or trap")
            return 'particle missing'

        if not self.c_p['z-tracking']:
            print("Z-tracking not on")
            return 'z-tracking not on'

        if not self.check_particles_can_touch_with_piezo():
            print("Particles too far apart")
            return 'Too far'

        # Check that we are autoaligning.
        if self.c_p['portenta_command_2'] == 0:
            print("Autoalign not on")
            return 'no autoalign'

        # Align in x
        if self.align_trapped_and_pipette_X():
            print("Aligning in x")
            return 'aligning in x'

        # Move in y until we get a positive force or the particles are touching.  
        dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] # this will always be positive since the pipette is below the trapped particle
        radii_sum = self.c_p['pipette_particle_location'][3] + self.c_p['Trapped_particle_position'][3]

        # Check DNA prescence
        particle_separation = (dy-radii_sum) * self.c_p['microns_per_pix']
        if self.check_DNA_prescence(particle_separation):
            print("DNA detected")
            self.c_p['molecule_attached'] = True
            self.c_p['touch_counter'] = 0
            return 'DNA present'

        # Check z-alignment, changed to an average to get a more reliable reading.
        dz = 10 * (self.trapped_bead_z.get_average() - self.pipette_bead_z.get_average())
        # Fix z-alginment if the particles are not touching and are missaligned in z
        if np.abs(dz) > 4 and dy > (radii_sum*1.2+2.5/self.c_p['microns_per_pix']):
            return 'need aligning in z'


        # Also if the pipette detectino isn't great it gets confused and stop too far from the bottom particle.
        self.move_up_and_down(max_dist=max_dist, force_limit_lower=force_limit, position_limit=position_limit)
      
        return 'touching particles'

    def move_up_and_down(self, max_dist=5, force_limit_lower=1000, force_limit_upper=1000, position_limit=0.8):
        """
        Moves the trapped particle up and down to touch the pipette particle and look for a molecule strand.
        Force limits refers to the maximum force okay to reach when moving down and touching the particles(limit lower)
        and moving up and pulling the molecule(limit upper). Position limit is the distance in particle radii that the particles are allowed to be pushed together with.
        """
        # TODO may want to change the position limit to something more intuitive.

        dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] # this will always be positive since the pipette is below the trapped particle
        radii_sum = self.c_p['pipette_particle_location'][3] + self.c_p['Trapped_particle_position'][3]
        current_force = np.abs(np.mean(self.data_channels['F_total_Y'].get_data(10))) 
        # TODO this risks going back and forth if there is a large force acting on the particle, return true and initiate pulling instead.
        if self.moving_down:
            # Check if the particles are touching, either force or radii can tell us this
            if current_force > force_limit_lower or dy < radii_sum*position_limit: # TODO make this in microns instead of radiis
                self.moving_down = False
                print("Changing direction of movement to up", current_force > force_limit_lower, dy < radii_sum*position_limit)
                return 'touching particles'
            if self.c_p['portenta_command_2'] == 1:
                self.c_p['piezo_B'][1] = min(64000, int(self.c_p['piezo_B'][1] + 200))
                if self.c_p['piezo_B'][1] >=64000:
                    self.moving_down = False
            else:
                self.c_p['piezo_A'][1] = min(64000, int(self.c_p['piezo_A'][1] + 200))
                if self.c_p['piezo_A'][1] >=64000:
                    self.moving_down = False
        else:
            # We are moving up, check if we have reached the max_distance(measured in particle radii)
            if current_force > force_limit_upper or dy > (max_dist/self.c_p['microns_per_pix'] + radii_sum):
                self.moving_down = True
                print("Changing direction of movement to down", current_force, force_limit_upper, dy, max_dist/self.c_p['microns_per_pix'], radii_sum)
                self.c_p['touch_counter'] += 1
                return 'touching particles'
            if self.c_p['portenta_command_2'] == 1:
                self.c_p['piezo_B'][1] = max(1000, int(self.c_p['piezo_B'][1] - 200))
                if self.c_p['piezo_B'][1] <= 1000:
                    self.moving_down = True
            else:
                self.c_p['piezo_A'][1] = max(1000, int(self.c_p['piezo_A'][1] - 200))
                if self.c_p['piezo_A'][1] <= 1000:
                    self.moving_down = True

    # Simple check to see if there is a molecule being stretched.
    def check_DNA_prescence(self, dy, force_limit=30, min_dist=2, max_dist=6):
        # TODO fix a more robust way of checking for DNA
        force = np.abs(np.mean(self.data_channels['F_total_Y'].get_data(30)) )
        if dy > min_dist and dy < max_dist and force > force_limit:
            print(f"DNA detected {dy} microns surface to surface, force of {force} pN")
            return True
        return False

    def check_molecule_broken(self, molecule_length=6, force_limit=20):
        """
        Checks if a molecule used for stretching has been broken.
        Does so by checking if the force is small at a large particle-particle separation.
        Returns True if the molecule is broken, otherwise false.
        (A small force at a small separation does not imply a broken molecule, as this is expected also in the prescence of a molecule)
        """
        force = np.abs(np.mean(self.data_channels['F_total_Y'].get_data(30)))
        dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] 
        radii_sum = self.c_p['pipette_particle_location'][3] + self.c_p['Trapped_particle_position'][3]
        separation = (dy - radii_sum) * self.c_p['microns_per_pix']

        return force < force_limit and separation > molecule_length

    def put_particle_in_pipette(self):
        """
        Moves the particle from the trap to the pipette.
        """
        if not self.c_p['particle_trapped']:
            return False

        x_min, x_max, y_min, y_max = self.approximate_piezo_moveable_area()
        if not (x_min < self.c_p['pipette_tip_location'][0] < x_max and y_min < self.c_p['pipette_tip_location'][1] - self.c_p['Trapped_particle_position'][3] < y_max):
            print("Pipette location outside of moveable area")
            print(self.c_p['pipette_tip_location'], x_min, x_max, y_min, y_max)


        # Move the particle to the pipette
        target_position = [self.c_p['pipette_tip_location'][0], self.c_p['pipette_tip_location'][1] - self.c_p['Trapped_particle_position'][3]] 
        return self.move_particle_to_position_with_piezo(target_position, piezo='piezo_b')
    
    def terminate_protocol(self):
        self.c_p['protocol_data'][0] = 0
        self.limits_found = [False, False]
        self.protocol_started = False
        # TODO reset the above parameters if we restart the function(turn on/off) Also
        if self.main_window.saving:
            self.main_window.record_data()
        
    def stretch_molecule(self):
        """
        Starts stretching a molecule and saving data from the stretch.
        """
        # TODO do the tracking check in a better way and add it to the check-molecule-broken if statement.
        # Check that we have a molecule attached and particles in position so we are ready to stretch
        if self.check_molecule_broken():
            self.terminate_protocol()
            print("Molecule broke, terminating protocol and attempting to reattach the molecule")
            return "Molecule broken"
        if not (self.c_p['particle_in_pipette'] and self.c_p['particle_trapped']):
            print(f"Particle lost: Particle in pipette: {self.c_p['particle_in_pipette']}, particle trapped: {self.c_p['particle_trapped']}")
            # Terminating protocol and resetting variables
            self.terminate_protocol()
            return "Not ready to stretch"

        # If the protocol is not started then we should create a protocol
        if not self.protocol_started:
            # Check if upper limit of protocol is found, if not move up
            if not self.limits_found[0]:
                self.moving_down = False
                self.move_up_and_down(max_dist=self.c_p['stretching_distance'] * 1.7, force_limit_lower=1000, force_limit_upper=self.c_p['stretch_force'], position_limit=1)
                y_force = np.abs(np.mean(self.data_channels['F_total_Y'].get_data(20)))
                if y_force >= self.c_p['stretch_force'] or self.c_p['piezo_B'][1]<1200 or self.c_p['piezo_A'][1]<1200:
                    self.limits_found[0] = True
                    if self.c_p['portenta_command_2'] == 1:
                        self.c_p['protocol_limits_dac'][0] = self.c_p['piezo_B'][1]
                    else:
                        self.c_p['protocol_limits_dac'][0] = self.c_p['piezo_A'][1]

            if not self.limits_found[1] and self.limits_found[0]:
                self.moving_down = True
                position_limit = 1.2
                self.move_up_and_down(max_dist=self.c_p['stretching_distance'], force_limit_lower=1000, force_limit_upper=self.c_p['stretch_force'], position_limit=position_limit)
                dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] # this will always be positive since the pipette is below the trapped particle
                radii_sum = self.c_p['pipette_particle_location'][3] + self.c_p['Trapped_particle_position'][3]
                if dy < radii_sum*position_limit or self.c_p['piezo_B'][1]>63800 or self.c_p['piezo_A'][1]>63800:
                    self.limits_found[1] = True
                    if self.c_p['portenta_command_2'] == 1:
                        self.c_p['protocol_limits_dac'][1] = self.c_p['piezo_B'][1]
                    else:
                        self.c_p['protocol_limits_dac'][1] = self.c_p['piezo_A'][1]
        if not(self.limits_found[0] and self.limits_found[1]):
            return "Looking for limits"

        if not self.protocol_started:
            self.protocol_started = True
            # Set the protocol
            split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF] 
            self.c_p['protocol_data'][1:3] = split_16_bit(int(self.c_p['protocol_limits_dac'][1])) # TODO check that we get the order here correct.
            self.c_p['protocol_data'][3:5] = split_16_bit(int(self.c_p['protocol_limits_dac'][0]))
            self.c_p['protocol_data'][5:7] = split_16_bit(int(self.c_p['stretching_speed']))
            if self.c_p['portenta_command_2'] == 1:
                self.c_p['protocol_data'][0] = 4 # Protocol activates Ay - B autoaligning
            else:
                self.c_p['protocol_data'][0] = 2 # Protocol activates By
            self.protocol_start_time = time()
            if not self.main_window.saving:
                self.main_window.record_data()

            return "Starting protocol"
        if time() - self.protocol_start_time > self.c_p['measurement_time']:
            self.terminate_protocol()
            # TODO ensure that this stops correctly. Sometimes the recoriding of data does not stop.
            self.main_window.record_data()
            return "Protocol finished"
        return "Protocol underway"

    def update_lasers_position_from_PSDs(self):
        """
        Estimates the laser positions based on the psd readings. Assumes that the laser_position_A and laser_position_B are correct and set to
        while the PSD_position_reading was 0.        
        """
        psd_a_sum = np.mean(self.data_channels['PSD_A_P_sum'].get_data(10))
        psd_b_sum = np.mean(self.data_channels['PSD_B_P_sum'].get_data(10))
        if psd_a_sum == 0 or psd_b_sum == 0:
            return
        psd_a_x = np.mean(self.data_channels['PSD_A_P_X'].get_data(10)) / psd_a_sum
        psd_a_y = np.mean(self.data_channels['PSD_A_P_Y'].get_data(10)) / psd_a_sum
        psd_b_x = np.mean(self.data_channels['PSD_B_P_X'].get_data(10)) / psd_b_sum
        psd_b_y = np.mean(self.data_channels['PSD_B_P_Y'].get_data(10)) / psd_b_sum

        # For some reason this only display something once the tracking is turned on.
        laser_a_x = self.c_p['laser_position_A'][0] + (self.c_p['laser_a_transfer_matrix'][0] * psd_a_x + self.c_p['laser_a_transfer_matrix'][1] * psd_a_y)/self.c_p['microns_per_pix']
        laser_a_y = self.c_p['laser_position_A'][1] + (self.c_p['laser_a_transfer_matrix'][2] * psd_a_x + self.c_p['laser_a_transfer_matrix'][3] * psd_a_y)/self.c_p['microns_per_pix'] # Changed to - here, transfer matrix also different after update of controller.

        laser_b_x = self.c_p['laser_position_B'][0] + (self.c_p['laser_b_transfer_matrix'][0] * psd_b_x + self.c_p['laser_b_transfer_matrix'][1] * psd_b_y)/self.c_p['microns_per_pix']
        laser_b_y = self.c_p['laser_position_B'][1] + (self.c_p['laser_b_transfer_matrix'][2] * psd_b_x + self.c_p['laser_b_transfer_matrix'][3] * psd_b_y)/self.c_p['microns_per_pix']
        self.c_p['laser_position_A_predicted'] = np.array([laser_a_x, laser_a_y])
        self.c_p['laser_position_B_predicted'] = np.array([laser_b_x, laser_b_y])

        self.c_p['laser_position'] = (self.c_p['laser_position_A_predicted'] + self.c_p['laser_position_B_predicted'])/2


    def suck_particle_into_pipette(self):
        """
        Puts the trapped particle inside the pipette.
        """
        if not self.c_p['particle_trapped'] or not self.c_p['pipette_located']:
            return "Particle not trapped or pipette not located"
        
        # estimate if we are supposed to go to the pipette with motors or with the piezo
        x_min, x_max, y_min, y_max = self.approximate_piezo_moveable_area()

        # TODO need to fix this properly so that we can move the particle to the pipette with the piezo.
        """
        if not (x_min < self.c_p['pipette_tip_location'][0] < x_max and y_min < self.c_p['pipette_tip_location'][1] < y_max) and not self.move2area_above_pipette_check(True):
            print("Pipette location outside of moveable area")
            self.c_p['move2area_above_pipette'] = True
            return "Moving with motors to pipette location"
        """
        # Move the particle to the pipette using the piezo
        distance_x = self.c_p['pipette_tip_location'][0] - self.c_p['Trapped_particle_position'][0]
        distance_y = self.c_p['pipette_tip_location'][1] - self.c_p['Trapped_particle_position'][1] - self.c_p['Trapped_particle_position'][3] # Accounting for radii now.
        if distance_x**2 + distance_y**2 < (6/self.c_p['microns_per_pix'])**2:
            print(f"Close enough to suck")
            self.c_p['move2area_above_pipette'] = False

            # TODO have some feedback on these steps.
            self.c_p['move_particle2pipette'] = False
            self.c_p['pump_PSU_on'] = True
            sleep(1)
            self.c_p['pump_PSU_on'] = False
            go2position(self.c_p['pipette_location_chamber'], self.c_p)
            # Turn of auto-aligning
            self.c_p['portenta_command_2'] = 0
            self.c_p['move_piezo_2_target'] = True
            # TODO drop particle, may have more than one trapped.
            # TODO ensure that we succed here, otherwise we need to try again. Also consider flushing main channel in case we got more than one particle stuck.
            return "Finished"

        if not self.move2area_above_pipette_check(False) and not self.c_p['move_particle2pipette']:
            #print("Pipette location outside of moveable area")
            self.c_p['move2area_above_pipette'] = True
            print("Pipette location outside of moveable area", x_max, x_min, y_max, y_min, self.c_p['pipette_tip_location'])
            return "Moving with motors to pipette location"

        # Move the particle to the pipette using the piezo
        distance_x = self.c_p['pipette_tip_location'][0] - self.c_p['Trapped_particle_position'][0]
        distance_y = self.c_p['pipette_tip_location'][1] - self.c_p['Trapped_particle_position'][1]
        print("Piezos moving")
        if distance_x**2 + distance_y**2 > (5/self.c_p['microns_per_pix'])**2:

            self.c_p['move_particle2pipette'] = True
            self.c_p['move2area_above_pipette'] = False
            return "Moving to pipette location"

        if self.c_p['move_particle2pipette']:
            self.c_p['move2area_above_pipette'] = False
            return "Moving to pipette location"
        
    def restart_experiment(self):
        """
        Drops the trapped particle and restarts the experiment with a new particle.
        Does this by moving the piezos away from their current position and then flushing the chamber.
        """
        # TODO ensure that this function does everything necessary to actually restart the experiment. i.e turns of any procedures that should not be on when restarting.
        self.c_p['drop_particle'] = True # TODO this is not executed properly when called from here
        print("Restaring experiment by dropping particle and flushing chamber.")
        # Setting some parameters to false
        self.c_p['search_and_trap'] = False
        self.c_p['centering_on'] = False
        self.c_p['move2area_above_pipette'] = False

        tmp = self.c_p['target_pressures'][self.c_p['central_channel_fluidics'][0]]
        self.c_p['target_pressures'][self.c_p['central_channel_fluidics'][0]] = self.c_p['central_channel_fluidics'][1]

        if self.c_p['valves_controller_connected']:
            for idx,_ in enumerate(self.c_p['valves_open']):
                print("Opening valves for flushing")
                self.c_p['valves_open'][idx] = True

        sleep(2)
        self.c_p['target_pressures'][self.c_p['central_channel_fluidics'][0]] = tmp

        if self.c_p['valves_controller_connected']:
            for idx,_ in enumerate(self.c_p['valves_open']):
                self.c_p['valves_open'][idx] = False
                print("Opening closing valves")
        return "Restarting experiment"
    
    def auto_molecule_experiment(self):
        """
        The full autonomous protocol for doing a molecule stretching experiment.

        1 Checks what is in the trap, if nothing is trapped then we should not autoalign.
        2 If the current step is check pipette then we focus the pipette
        3 Check pipette contents


        """
        particle_trapped = sum(self.data_channels['particle_trapped'].get_data(10)) > 5 # A more robust check if something is trapped.
        # TODO flush chamber before going to trap new particles.
        # TODO take care to turn off stuff that is no longer needed so that only one loop runs at a time.
        if not particle_trapped:
            self.c_p['move_piezo_2_target'] = True
            self.c_p['portenta_command_2'] = 0
        else:
            self.c_p['portenta_command_2'] = 2


        if self.c_p['multiple_particles_trapped']:
            # Drops the trapped particle and restarts the experiment with a new particle.
            self.c_p['autocontroller_current_step'] = 'checking_pipette'
            print("Many particles trapped, restarting experiment")
            self.restart_experiment()
            return

        if self.c_p['autocontroller_current_step'] == 'checking_pipette':
            print("Checking pipette")
            # Ensure that we have not left anything unwanted on, TODO turn these of in the correct locations in the code, not here.
            self.c_p['centering_on'] = False
            self.c_p['touch_particles'] = False
            self.c_p['search_and_trap'] = False
            self.c_p['move2area_above_pipette'] = False

            x = self.data_channels['Motor_x_pos'].get_data(1)[0]
            y = self.data_channels['Motor_y_pos'].get_data(1)[0]

            if (x-self.c_p['pipette_location_chamber'][0])**2 + (y-self.c_p['pipette_location_chamber'][1])**2 > 200:
                if not self.c_p['move_to_location']:
                    go2position(self.c_p['pipette_location_chamber'], self.c_p)
                print(f"Moving to pipette location {self.c_p['pipette_location_chamber']}")
                return
            if self.c_p['move_to_location']:
                return

            if not self.c_p['focus_pipette'] and self.c_p['autocontroller_current_step'] != 'focusing_pipette':
                # TODO fix auto-focus so it is done consistently.
                self.c_p['autocontroller_current_step'] = 'focusing_pipette'
                self.c_p['focus_pipette'] = True
                print("Starting focusing of pipette")
                return

        if self.c_p['focus_pipette'] and self.c_p['autocontroller_current_step'] == 'focusing_pipette':
            print("Focusing pipette")
            self.c_p['touch_particles'] = False
            return

        # If we are close to the pipette then we should check what is in the pipette since now we are focused
        if self.c_p['autocontroller_current_step'] == 'focusing_pipette' and not self.c_p['move_to_location']:
            print("Checking pipette contents and updating z position of pipette")
            if self.c_p['pipette_located']:
                self.c_p['pipette_location_chamber'][2] = self.data_channels['Motor_z_pos'].get_data(1)[0]
            else:
                self.c_p['autonomous_experiment'] = False
                print("Pipette not located, stopping experiment")
            # 4 cases, particle trapped and particle in piptte can both be either true or false
            if not self.c_p['particle_in_pipette'] and not particle_trapped:
                self.c_p['search_and_trap'] = True
                self.c_p['particle_type'] = 1 # Get particle for the pipette
                self.c_p['autocontroller_current_step'] = 'searching_for_particle_1'
                self.c_p['move_piezo_2_target'] = True
                self.c_p['portenta_command_2'] = 1
                return
            if not self.c_p['particle_in_pipette'] and particle_trapped:
                # Put the particle in the pipette, needs manual sucking to continue, i.e the user needs to press a button to confirm that the particle has been sucked after
                print("Pipette suction time, hooray!")
                self.c_p['suck_into_pipette'] = True
                # self.c_p['autocontroller_current_step'] = 'sucking_into_pipette'
                return

            if self.c_p['particle_in_pipette'] and not particle_trapped:
                # Go get particle type number 2
                self.c_p['search_and_trap'] = True
                self.c_p['particle_type'] = 2 # Get particle with DNA
                self.c_p['autocontroller_current_step'] = 'searching_for_particle_2'
                self.c_p['move_piezo_2_target'] = True
                self.c_p['portenta_command_2'] = 0
                return

            if self.c_p['particle_in_pipette'] and particle_trapped:
                # if the trapped particle is of type 1 then we shuld drop it and get a type 2 particle
                if self.c_p['particle_type'] == 1:
                    # TODO drop particle and start searching for particle 2
                    pass
                # We are ready to move on to the step where we attach the DNA now.
                self.c_p['autocontroller_current_step'] = 'move2area_above_pipette'

        
        # We are searching for a new particle to trap, let the other loop do it's job
        if self.c_p['autocontroller_current_step'] == 'searching_for_particle_1' or self.c_p['autocontroller_current_step'] == 'searching_for_particle_2':
            # Once the loop is finished it automatically turns off the search_and_trap and goes to the pipette, use this as a cue to move on by aligning the pipette and the particle in it
            if not self.c_p['search_and_trap']:
                self.c_p['autocontroller_current_step'] = 'checking_pipette'
            return

        if self.c_p['autocontroller_current_step'] == 'move2area_above_pipette':

            # Check trapped, if not go back to checking pipette
            if not particle_trapped:
                self.c_p['move2area_above_pipette'] = False
                self.c_p['autocontroller_current_step'] = 'checking_pipette'
                return    

            # Cgeck if we should move with the motors, if so autoalign and move towards the right area
            if not self.move2area_above_pipette_check(True):
                # todo the conditions for this and the condtion in the move2area_above_pipette function should be the same.
                # Move 2 area above the pipette and toggle autoaligning
                self.c_p['portenta_command_2'] = 1
                self.c_p['move2area_above_pipette'] = True
                self.c_p['autocontroller_current_step'] = 'move2area_above_pipette'
                return
            # If we should not move with the motors then we should move with the piezo and initiate the "touching particles protocol"
            self.c_p['move2area_above_pipette'] = False
            self.c_p['touch_particles'] = True
            self.c_p['autocontroller_current_step'] = 'touching_particles'
            #if not particle_trapped: # Redundant since we check this above
            return

        if self.c_p['autocontroller_current_step'] == 'touching_particles':
            # Here we touch the particles and we also initate the pulling protocol. Haven't tested this part for errors yet.
            # TODO got stuck after getting the Particle missing readout
            if not particle_trapped or not self.c_p['particle_in_pipette']:
                # TODO when loosing view of particle during stretching this does not restart the trapping part correctly, fix this.

                self.c_p['autocontroller_current_step'] = 'checking_pipette'
                self.c_p['touch_particles'] = False
                return

            if self.c_p['touch_counter'] > self.touch_counter_limit:
                # Tested and works well, restarts the experiment if we have touched the particles too many times.
                print("Touch counter limit reached")
                self.c_p['touch_counter'] = 0
                self.restart_experiment()
                self.c_p['autocontroller_current_step'] = 'checking_pipette'

            # Check if we have finished the measurement, if so we should restart the experiment.
            if self.c_p['experiment_finished']:
                self.c_p['autocontroller_current_step'] = 'checking_pipette'
                self.restart_experiment()
                self.c_p['experiment_finished'] = False
                return

            dx = (self.c_p['pipette_particle_location'][0] - self.c_p['Trapped_particle_position'][0])*self.c_p['microns_per_pix']
            dy = (self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1])*self.c_p['microns_per_pix']
            if dx**2 + dy**2 > 220: # More than ca 15 microns between the particles is unrealisitc to have, increase this parameter if really long molecules are expected.
                self.c_p['touch_particles'] = False
                self.c_p['autocontroller_current_step'] = 'checking_pipette'
                return
            return

    def run(self):
        """
        Currently this is primarily used to test different routines for automation.
        
        """
        prev_time = time()
        while self.c_p['program_running']:
            self.c_p['loop_execution_time'] = time() - prev_time 
            prev_time = time()
            if self.c_p['move_piezo_2_target']:
                if self.move_piezos_2_saved_positions(self.c_p['piezo_target_positions']):
                    self.c_p['move_piezo_2_target'] = False
            self.update_lasers_position_from_PSDs()
            # TODO clean up this part of the code.
            if self.c_p['model'] is not None and self.c_p['tracking_on']:
    
                self.YOLO_prediction()
                if self.c_p['z-tracking']:
                    self.predict_z_positions()
                self.check_trapped()
                self.check_in_pipette()
                self.data_channels['particle_trapped'].put_data(self.c_p['particle_trapped']) # TODO consider if this should really be a datachannel as well as a control parameter.
                self.check_multiple_trapped() # Seems to work
                
            else:
                sleep(0.1)
                continue
            if self.c_p['calibration_running']:
                if self.auto_calibration():
                    self.c_p['calibration_running'] = False
                    self.c_p['calibration_start'] = True
                continue

            if self.c_p['centering_on']:
                center = [self.c_p['laser_position'][0] - self.c_p['AOI'][0], self.c_p['laser_position'][1] - self.c_p['AOI'][2]]
                self.trap_particle_minitweezers(center)
                if self.c_p['particle_trapped']:
                    self.c_p['centering_on'] = False
                    if self.moving2trap:
                        self.moving2trap = False
                        self.c_p['motor_x_target_speed'] = 0
                        self.c_p['motor_y_target_speed'] = 0

            elif self.c_p['search_and_trap']:
                if self.find_and_trap_particle(self.c_p['particle_type']):
                    self.c_p['search_and_trap'] = False

            elif self.c_p['move_avoiding_particles']:
                self.move_while_avoiding_particlesV2(self.c_p['saved_positions'][0][1:3])
            elif self.c_p['electrostatic_protocol_toggled'] and not self.c_p['electrostatic_protocol_finished']:
                self.custom_electrostatic_protocol()
            elif self.c_p['find_laser_position']:
                self.find_true_laser_position()
                self.c_p['find_laser_position'] = False


            elif self.c_p['move2area_above_pipette']:
                 if self.move2area_above_pipette(move2particle=self.c_p['particle_in_pipette']):
                    self.c_p['move2area_above_pipette'] = False

            elif self.c_p['move_particle2pipette']:
                if self.put_particle_in_pipette():
                    print("Particle in pipette, or close enough to do a sucking.")
                    self.c_p['move_particle2pipette'] = False

            elif self.c_p['touch_particles']:
                # THis protocol does not have a clear stopping criterion yet, will have once we move to real experiments.
                message = self.touch_trapped_2_pieptte_particle()

                # Resets the aggresive stretching
                self.limits_found = [False, False]
                self.protocol_started = False
                self.c_p['protocol_data'][0] = 0

                if message == 'touching particles':
                    print("Particles touching")

                elif message == 'particle missing':
                    print("Particle missing")
                    self.c_p['touch_particles'] = False
                    self.c_p['touch_counter'] = 0

                elif message == "No molecule found":
                    self.c_p['touch_particles'] = False

                elif message == 'need aligning in z':
                    print("Need aligning in z")
                    self.c_p['focus_z_trap_pipette'] = True

                elif message == 'no autoalign':
                    print("No autoalign, turning it on.")
                    self.c_p['portenta_command_2'] = 1

                elif message == 'DNA present':
                    print("DNA present, Hallelujah!")
                    self.c_p['touch_particles'] = False
                    self.c_p['stretch_molecule'] = True
                    self.c_p['molecule_attached'] = True

            elif self.c_p['stretch_molecule']:
                message = self.stretch_molecule()
                # TODO instead of printing the message have a display in the control widget that displays the message to tell the user what is happening.
                if message == "Looking for limits":
                    print("Looking for limits")
                elif message == "Starting protocol":
                    print("Starting protocol")
                elif message == "Protocol underway":
                    print("Protocol underway")
                elif message == "Molecule broken":
                    print("Molecule broken")
                    self.c_p['stretch_molecule'] = False
                    self.c_p['touch_particles'] = True
                elif message == "Not ready to stretch":
                    print("Not ready to stretch")
                    self.c_p['stretch_molecule'] = False
                elif message == "Protocol finished":
                    print("Protocol finished")
                    # TODO make it turn off the autonomous protocol when finished and restart the experiment.
                    self.c_p['experiment_finished'] = True
                    self.c_p['stretch_molecule'] = False
            elif self.c_p['focus_pipette']:
                if self.z_focus_pipette():
                    self.c_p['focus_pipette'] = False

            if self.c_p['drop_particle']:
                if self.drop_particle_and_reset_trap():
                    self.c_p['drop_particle'] = False
            
            if self.c_p['suck_into_pipette']:
                res = self.suck_particle_into_pipette()
                if res == "Finished":
                    self.c_p['suck_into_pipette'] = False

            if self.c_p['focus_z_trap_pipette']:
                # RESET THE pipette focus
                # TODO do this in a better place in the code
                if self.z_focus():
                    self.c_p['focus_z_trap_pipette'] = False

            if self.c_p['autonomous_experiment']:
                 # TODO add possiblity to set the stage of the experiment we are in, do this in the widget
                 self.auto_molecule_experiment()

    def auto_calibration(self, nbr_averging_points=2000):
        """
        Protocol for automatically moving the trapped particle in a grid pattern recording the laser positions and forces at the different locations.
        uses the range 3000 to 63000 on the piezos to make the grid.

        The data is stored in a 3-array, calibration_points ordered as follows:
        particle position (x,y)
        laser positions (A_x, A_y, B_x, B_y)
        force (A_x, A_y, B_x, B_y, Z)
        DAC values (A_x, A_y, B_x, B_y)
        """

        # If we are at the start of a calibration, reset the grid and set the start position.
        if self.c_p['calibration_start']:
            self.calibration_x = 0
            self.calibration_y = 0
            self.calibration_timer = time()
            self.c_p['calibration_start'] = False
            self.calibration_dir = "forwards"
            self.c_p['portenta_command_2'] = 1 # Autoaliging A
            # TODO may need to change so that it moves to the top left more controlled.
            self.c_p['piezo_B'][0] = int(3000 + self.calibration_x * self.grid_step)
            self.c_p['piezo_B'][1] = int(3000 + self.calibration_y * self.grid_step)
            print(f"Moving to position {self.c_p['piezo_B']}, step {self.calibration_x}, {self.calibration_y}")
            return False

        if time()-self.calibration_timer < self.calibration_wait_time:
            return False
        
        # Update time to current time
        self.calibration_timer = time()

        # Record the data
        # Particle position in image
        x_position = (self.c_p['Trapped_particle_position'][0]+self.c_p['AOI'][0]) * self.c_p['microns_per_pix']
        y_position = (self.c_p['Trapped_particle_position'][1]+self.c_p['AOI'][2]) * self.c_p['microns_per_pix']

        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 0] = x_position
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 1] = y_position

        # Laser positions
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 2] = np.mean(self.data_channels['PSD_A_P_X'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 3] = np.mean(self.data_channels['PSD_A_P_Y'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 4] = np.mean(self.data_channels['PSD_B_P_X'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 5] = np.mean(self.data_channels['PSD_B_P_Y'].get_data(nbr_averging_points))

        # Force readings
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 6] = np.mean(self.data_channels['PSD_A_F_X'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 7] = np.mean(self.data_channels['PSD_A_F_Y'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 8] = np.mean(self.data_channels['PSD_B_F_X'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 9] = np.mean(self.data_channels['PSD_B_F_Y'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 10] = np.mean(self.data_channels['F_total_Z'].get_data(nbr_averging_points))

        # DAC values
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 11] = self.data_channels['dac_ax'].get_data(1)[0]
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 12] = self.data_channels['dac_ay'].get_data(1)[0]
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 13] = self.data_channels['dac_bx'].get_data(1)[0]
        self.c_p['calibration_points'][self.calibration_x, self.calibration_y, 14] = self.data_channels['dac_by'].get_data(1)[0]

        # Move to the next position
        if self.calibration_dir== "forwards" and self.calibration_x < self.c_p['grid_size']-1:
            self.calibration_x += 1

        elif self.calibration_dir == "forwards" and self.calibration_x >= self.c_p['grid_size']-1:
            self.calibration_y += 1
            self.calibration_dir = "backwards"
        elif self.calibration_dir == "backwards" and self.calibration_x > 0:
            self.calibration_x -= 1
        elif self.calibration_dir == "backwards" and self.calibration_x <= 0:
            self.calibration_y += 1
            self.calibration_dir = "forwards"

            # If we have reached the end, save the data and return True
        if self.calibration_y >= self.c_p['grid_size']:
            # Save the calibration array
            calibration_file = self.c_p['recording_path']+'/'+self.c_p['filename']+'auto_calibration'
            print(f"Calibration finished, saving to file{calibration_file}")
            np.save(calibration_file, self.c_p['calibration_points'])
            self.c_p['calibration_performed'] = True
            return True

        # Set the position of the piezos
        self.c_p['portenta_command_2'] = 1 # Autoaliging A
        self.c_p['piezo_B'][0] = int(3000 + self.calibration_x * self.grid_step)
        self.c_p['piezo_B'][1] = int(3000 + self.calibration_y * self.grid_step)
        print(f"Moving to position {self.c_p['piezo_B']}, step {self.calibration_x}, {self.calibration_y}")
        return False
    

"""

    def attach_DNA(self):
        # This function is essentially made obsolete by the touch_trapped_2_pipette_particle function.
        # Check that the two necessary particles are in view, someitmes the bead in the pipette is misstaken for part of the pipette, thereof the offset.
        # pipette_particle, _ = self.find_particle_in_pippette(offset=0)
        if not (self.c_p['particle_in_pipette'] and self.c_p['particle_trapped']):
            print("Particle missing in either pipette or trap")
            return

        if not self.c_p['portenta_command_2'] == 1:
            print("Not autoaligned")
            return 

        touchable = self.check_particles_can_touch_with_piezo()
        if not touchable:
            print("Particles too far apart")
            return

        pipette_particle = self.c_p['pipette_particle_location']
        dx = ((self.c_p['laser_position'][0]-self.c_p['AOI'][0]) - pipette_particle[0])
        dy = -((self.c_p['laser_position'][1]-self.c_p['AOI'][2]) - pipette_particle[1])
        dz = self.c_p['Trapped_particle_position'][2] - pipette_particle[2]

        # Adjust DX
        print("DX",self.c_p['laser_position'][0],self.c_p['AOI'][0],pipette_particle[0])
        print("DY",self.c_p['laser_position'][1],self.c_p['AOI'][2],pipette_particle[1],dy)

        if np.abs(dy)< 0.1 and np.abs(dx)< 1:
            return

        if dy<self.closest_distance-20:
            self.c_p['piezo_B'][1] = max(0,self.c_p['piezo_B'][1] - 200)
            if self.c_p['piezo_B'][1] == 0:
                # Error which we cannot fix from here

                return False
            return
        # TODO do a proper check if we can reasonably move to the correct location
        if dx>3:
            self.c_p['piezo_B'][0] = min(2**16-1,self.c_p['piezo_B'][0]+min(dx*self.bits_per_pixel, 400))
            return
        if dx<-3:
            self.c_p['piezo_B'][0] = max(0,self.c_p['piezo_B'][0]+max(dx*self.bits_per_pixel, -400))
            return
        
        
        if dy>self.DNA_length_pix and self.data_channels['F_total_Y'].get_data(1)[0] > self.force_limit:
            print("DNA found and attached")
            return True
        # Need to handle situation in which we start out with the bead in the trap being lower than that of the pipette.
        if self.DNA_move_direction == 0:
            # Move towards the pipette bead
            if dy<self.closest_distance:

                # Take a break of ca 10 seconds to let the DNA attach before separating the beads
                if self.sleep_counter < 100:
                    self.sleep_counter += 1
                    return
                self.sleep_counter = 0
                self.DNA_move_direction = 1
                return
            self.c_p['piezo_B'][1] = min(2**16-1,self.c_p['piezo_B'][1]+ 200)

        elif self.DNA_move_direction == 1:
            # Move away from the pipette bead            
            self.c_p['piezo_B'][1] = max(0,self.c_p['piezo_B'][1] - 200)
            if dy>self.DNA_length_pix+10:
                # Moved to far let's turn down
                self.DNA_move_direction = 0
                return
            


"""