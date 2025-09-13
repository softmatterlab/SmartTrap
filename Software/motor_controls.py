from __future__ import annotations
from typing import Protocol, runtime_checkable, Sequence

from PyQt6.QtWidgets import (    
    QLineEdit, QCheckBox, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QGridLayout,
    QToolBar
)

from PyQt6.QtGui import QDoubleValidator, QKeySequence, QAction
from mouse_tool import MouseTool
from time import time
import numpy as np


@runtime_checkable
class Motor(Protocol):
    """
    Generic N-axis motor stage.

    Conventions:
    - Positions are sequences of floats: len(position) == number of axes.
    - Velocities are passed as varargs: move_at_speed(vx, vy, [vz, ...]).
    """
    # TODO add connect function
    def set_speed(self, mm_per_s: float) -> None: ...
    def get_speed(self) -> float: ...

    def move_to_location(self, position: Sequence[float]) -> None: ...
    def move_at_speed(self, *v: float) -> None: ...

    def get_position(self) -> Sequence[float]: ...
    def stop_moving(self) -> None: ...
    def is_moving(self) -> bool: ...


@runtime_checkable
class ObjectiveMovement(Protocol):
    def connect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def slow_towards_sample(self) -> None: ...
    def slow_away_from_sample(self) -> None: ...
    def fast_towards_sample(self) -> None: ...
    def fast_away_from_sample(self) -> None: ...



class TestMotorController(Motor):
    def __init__(self):
        self.speed = 10.0
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0

    def set_speed(self, speed):
        if np.abs(float(speed)) < 100:
            self.speed = float(speed)
        else:
            self.speed = float(100)

    def get_speed(self):
        return self.speed

    def move_to_location(self, position):
        print(f"Moving to location {position}")
        self.x_pos = position[0]
        self.y_pos = position[1]
        self.z_pos = position[2]

    def get_x_position(self):
        return self.x_pos
    
    def get_y_position(self):
        return self.y_pos
    
    def get_z_position(self):
        return self.z_pos

    def stop_moving(self):
        print("Stopping movement")

    def is_moving(self):
        return False

    def move_at_speed(self, x_speed, y_speed, z_speed=0):
        print(f"Moving at speed x: {x_speed}, y: {y_speed}, z: {z_speed}")
        self.x_pos += x_speed * 1
        self.y_pos += y_speed * 1
        self.z_pos += z_speed * 1

class TestObjectiveMovement(ObjectiveMovement):
    def __init__(self):
        self.connected = False

    def connect(self, adress):
        self.connected = True
        print(f"Connected to objective at {adress}")

    def is_connected(self):
        return self.connected

    def slow_towards_sample(self):
        print("Moving slow towards sample")

    def slow_away_from_sample(self):
        print("Moving slow away from sample")

    def fast_towards_sample(self):
        print("Moving fast towards sample")

    def fast_away_from_sample(self):
        print("Moving fast away from sample")

class MotorControllerWindow(QWidget):
    """
    Widget for controlling the motors moving the sample stage. Allows for changing the 
    speed of the motors and moving them in the x and y direction.
    """
        
    def __init__(self, c_p, motor_controller: "Motor"):
        super().__init__()
        self.c_p = c_p
        self.motor_controller = motor_controller
        self.setWindowTitle("Motor controller")

        main = QVBoxLayout()

        # --- Speed label + edit ---
        self.label = QLabel("Set motor speed (0–100) microns/s")
        main.addWidget(self.label)

        self.SpeedLineEdit = QLineEdit()
        self.SpeedLineEdit.setValidator(QDoubleValidator(0.0, 100.0, 1, self))
        self.SpeedLineEdit.setText(str(self.motor_controller.get_speed()))
        self.SpeedLineEdit.setToolTip("""Sets the motor speed. Units are in microns/second.\n
                                      The mose movement is independent of the speed set here.""")
        self.SpeedLineEdit.textChanged.connect(self.motor_controller.set_speed)
        main.addWidget(self.SpeedLineEdit)

        # --- Presets row ---
        presets = QHBoxLayout()
        self.slow_speed_button = QPushButton('1 micron/s')
        self.slow_speed_button.clicked.connect(lambda: self.SpeedLineEdit.setText("1"))
        presets.addWidget(self.slow_speed_button)
        self.medium_speed_button = QPushButton('10 microns/s')
        self.medium_speed_button.clicked.connect(lambda: self.SpeedLineEdit.setText("10"))
        presets.addWidget(self.medium_speed_button)
        self.fast_speed_button = QPushButton('100 microns/s')
        self.fast_speed_button.clicked.connect(lambda: self.SpeedLineEdit.setText("100"))
        presets.addWidget(self.fast_speed_button)
        main.addLayout(presets)

        # --- Arrows (grid) ---
        arrows = QGridLayout()
        self.up_button = QPushButton("↑ Up")
        self.up_button.pressed.connect(self.move_up)
        self.up_button.released.connect(self.stop_y)
        self.up_button.setShortcut(QKeySequence("Up"))
        self.up_button.setToolTip("Moves left in the sample. Also accessible via the left arrow key.")
        arrows.addWidget(self.up_button, 0, 1)

        self.left_button = QPushButton("← Left")
        self.left_button.setShortcut(QKeySequence("Left"))
        self.left_button.pressed.connect(self.move_left)
        self.left_button.released.connect(self.stop_x)
        self.left_button.setToolTip("Moves down in the sample. Also accessible via the down arrow key.")
        arrows.addWidget(self.left_button, 1, 0)

        self.down_button = QPushButton("↓ Down")
        self.down_button.setShortcut(QKeySequence("Down"))
        self.down_button.pressed.connect(self.move_down)
        self.down_button.released.connect(self.stop_y)
        self.down_button.setToolTip("Moves right in the sample. Also accessible via the right arrow key.")
        arrows.addWidget(self.down_button, 1, 1)

        self.right_button = QPushButton("→ Right")
        self.right_button.setShortcut(QKeySequence("Right"))
        self.right_button.pressed.connect(self.move_right)
        self.right_button.setToolTip("Moves up in the sample. Also accessible via the up arrow key.")
        self.right_button.released.connect(self.stop_x)
        arrows.addWidget(self.right_button, 1, 2)

        main.addLayout(arrows)

        # --- Sample forward/backward ---
        sample = QHBoxLayout()
        self.objective_forward_button = QPushButton("Sample forward")
        self.objective_forward_button.setShortcut(QKeySequence("PgUp"))
        self.objective_forward_button.pressed.connect(self.objective_forward)
        self.objective_forward_button.released.connect(self.objective_stop)
        self.objective_forward_button.setToolTip("Moves the sample forwards towards the imaging objective. Also accessible via the page up key.")
        sample.addWidget(self.objective_forward_button)

        self.objective_backward_button = QPushButton("Sample backward")
        self.objective_backward_button.setShortcut(QKeySequence("PgDown"))
        self.objective_backward_button.pressed.connect(self.objective_backward)
        self.objective_backward_button.released.connect(self.objective_stop)
        self.objective_backward_button.setToolTip("Moves the sample backwards towards the imaging objective. Also accessible via the page down key.")
        sample.addWidget(self.objective_backward_button)

        main.addLayout(sample)

        # --- LED toggle ---
        main.addWidget(QLabel("Sample LED ON/OFF"))
        self.led_button = QCheckBox()
        self.led_button.setChecked(self.c_p['blue_led'] == 0)
        self.led_button.setStyleSheet("""
            QCheckBox::indicator {
                width: 30px;
                height: 30px;
            }
        """)
        main.addWidget(self.led_button)

        self.setLayout(main)

    def toggle_led(self, state):
        """
        Toggle the saving_toggled property of the DataChannel when the checkbox is toggled.
        """
        self.c_p['blue_led'] = 0 if bool(state) else 1
        if self.c_p['blue_led'] == 0:
            print("LED on")
        else:
            print("LED off")

    def move_up(self):
        self.motor_controller.move_at_speed(self.motor_controller.get_speed(),0,0)

    def stop_y(self):
        self.motor_controller.stop_moving()

    def move_down(self):
        self.motor_controller.move_at_speed(-self.motor_controller.get_speed(),0,0)

    def move_right(self):
        self.motor_controller.move_at_speed(0,-self.motor_controller.get_speed(),0)

    def stop_x(self):
        self.motor_controller.stop_moving()

    def move_left(self):
        self.motor_controller.move_at_speed(0,self.motor_controller.get_speed(),0)

    def objective_forward(self):
        self.motor_controller.move_at_speed(0,0,self.motor_controller.get_speed())

    def objective_stop(self):
        self.motor_controller.stop_moving()

    def objective_backward(self):
        self.motor_controller.move_at_speed(0,0,-self.motor_controller.get_speed())

class MotorMouseMove(MouseTool):
    
    def __init__(self, c_p, data_channels, MotorController):
        self.c_p = c_p
        self.data_channels = data_channels # Needed to know the position and speed
        self.x_0 = 0
        self.y_0 = 0
        self.z_0 = 0
        self.x_prev = 0
        self.y_prev = 0
        self.z_prev = 0
        self.prev_t = time()
        self.speed_factor = 4
        self.MotorController = MotorController

    def mousePress(self):
        # left click
        if self.c_p['mouse_params'][0] == 1:
            center_x = int((self.c_p['camera_width']/2 - self.c_p['AOI'][0])/self.c_p['image_scale'])
            center_y = int((self.c_p['camera_height']/2 - self.c_p['AOI'][2])/self.c_p['image_scale'])

            # Checking that the click was made in an ok position
            width = (self.c_p['AOI'][1] - self.c_p['AOI'][0])/self.c_p['image_scale'] / 2
            height = (self.c_p['AOI'][3] - self.c_p['AOI'][2])/self.c_p['image_scale'] / 2
            if self.c_p['mouse_params'][1] < center_x - width or self.c_p['mouse_params'][1] > center_x + width:
                return
            if self.c_p['mouse_params'][2] < center_y - height or self.c_p['mouse_params'][2] > center_y + height:
                return

            dx_ticks = (self.c_p['mouse_params'][1] - center_x) * self.c_p['image_scale']*self.c_p['ticks_per_pixel']
            dy_ticks = (self.c_p['mouse_params'][2] - center_y) * self.c_p['image_scale']*self.c_p['ticks_per_pixel']

            x = int(self.MotorController.get_x_position() + dx_ticks)
            y = int(self.MotorController.get_y_position() - dy_ticks)

            self.MotorController.move_to_location((x,y,self.MotorController.get_z_position()))
            
  
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            if self.x_prev == self.c_p['mouse_params'][1] and self.y_prev == self.c_p['mouse_params'][2]:
                self.MotorController.stop_moving()
                return
            self.x_prev = self.c_p['mouse_params'][1]
            self.y_prev = self.c_p['mouse_params'][2]

        if self.c_p['mouse_params'][0] == 3:
            self.z_0 = self.c_p['mouse_params'][2]
            self.z_prev = self.c_p['mouse_params'][2]

    def mouseRelease(self):
        self.MotorController.stop_moving()

    def mouseDoubleClick(self):
        pass
    
    def draw(self, qp):
        pass
    
    def check_speed(self, speed):
        if 0 < speed < 2:
            return 2
        if 0 > speed > -2:
            return -2
        if speed > 32767:
            return 32767
        if speed < -32767:
            return  -32767
        return speed

    def mouseMove(self):
        """
        This function is called when the mouse is moved. It is used to move the motors
        in real time.
        """
        if self.c_p['mouse_params'][0] == 2:

            dx = (self.c_p['mouse_params'][3] - self.x_prev)
            dy = (self.c_p['mouse_params'][4] - self.y_prev)
            x_speed = self.check_speed(dx * self.speed_factor)
            y_speed = self.check_speed(dy * self.speed_factor)
            
            self.MotorController.move_at_speed(int(y_speed), int(x_speed), 0) # x and y are mirrored
            self.x_prev = self.c_p['mouse_params'][3]
            self.y_prev = self.c_p['mouse_params'][4]
            
        elif self.c_p['mouse_params'][0] == 3:
            dz = (self.c_p['mouse_params'][4] - self.z_prev)
            z_speed = self.check_speed(dz * self.speed_factor/5)            
            self.MotorController.move_at_speed(0,0,int(z_speed))
            self.z_prev = self.c_p['mouse_params'][4]

    def getToolName(self):
        return "Minitweezers motor"

    def getToolTip(self):
        return "Move the motors by clicking or dragging on the screen"


class ObjectiveStepperControllerToolbar(QToolBar):
    """
    Simple toolbar to control the objective stepper motor. Can move the stepper motor
    either towards or away from the sample. Can move slowly or fast (big or small steps).
    """
    def __init__(self, objective_mover, parent):
        super().__init__("Objective Controller", parent)
        self.objective_mover = objective_mover
        
        # Add move towards action
        self.slow_towards_action = QAction('Slow towards sample')
        self.slow_towards_action.triggered.connect(self.objective_mover.slow_towards_sample)
        self.addAction(self.slow_towards_action)

        # Add move away action
        self.slow_away_action = QAction('Slow away from sample')
        self.slow_away_action.triggered.connect(self.objective_mover.slow_away_from_sample)   
        self.addAction(self.slow_away_action)

        # Add move fast towards action
        self.fast_towards_action = QAction('Fast towards sample')
        self.fast_towards_action.triggered.connect(self.objective_mover.fast_towards_sample)
        self.addAction(self.fast_towards_action)

        # Add move fast away action
        self.fast_away_sample_action = QAction('Fast away from sample')
        self.fast_away_sample_action.triggered.connect(self.objective_mover.fast_away_from_sample)
        self.addAction(self.fast_away_sample_action)