from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,QCheckBox,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

from PyQt6.QtGui import QAction, QIntValidator

import numpy as np
from CustomMouseTools import MouseInterface
from time import time, sleep


class MotorControllerToolbar(QToolBar):
    """
    Widget for controlling the motors of the minitweezers setup. Allows for changing the 
    speed of the motors and moving them in the x and y direction.
    """
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        self.motor_speed = self.c_p['minitweezers_goto_speed'] 
        self.y_movement = 0
        self.setWindowTitle("Motor controller")

        self.label = QLabel("Set motor speed (0-32767) a.u")
        self.addWidget(self.label)

        self.SpeedLineEdit = QLineEdit()
        self.SpeedLineEdit.setValidator(QIntValidator(0,32767))
        self.SpeedLineEdit.setText(str(self.c_p['minitweezers_goto_speed']))
        self.SpeedLineEdit.setToolTip("Sets the motor speed. Units are arbitrary and do not apply to movement with the mouse.")
        self.SpeedLineEdit.textChanged.connect(self.set_motor_speed)
        self.addWidget(self.SpeedLineEdit) 
        
        self.up_button = QPushButton('LEFT')
        self.up_button.pressed.connect(self.move_up)
        self.up_button.released.connect(self.stop_y)
        self.up_button.setToolTip("Moves left in the sample. Also accessible via the left arrow key.")
        self.up_button.setCheckable(False)
        self.up_button.setShortcut("left")
        self.addWidget(self.up_button)

        self.down_button = QPushButton('RIGHT')
        self.down_button.pressed.connect(self.move_down)
        self.down_button.released.connect(self.stop_y)
        self.down_button.setToolTip("Moves right in the sample. Also accessible via the right arrow key.")
        self.down_button.setCheckable(False)
        self.down_button.setShortcut("right")
        self.addWidget(self.down_button)


        self.right_button = QPushButton('UP')
        self.right_button.pressed.connect(self.move_right)
        self.right_button.released.connect(self.stop_x)
        self.right_button.setToolTip("Moves up in the sample. Also accessible via the up arrow key.")   
        self.right_button.setShortcut("up")
        self.right_button.setCheckable(False)
        self.addWidget(self.right_button)

        self.left_button = QPushButton('DOWN')
        self.left_button.pressed.connect(self.move_left)
        self.left_button.setShortcut("down")
        self.left_button.setToolTip("Moves down in the sample. Also accessible via the down arrow key.")
        self.left_button.released.connect(self.stop_x)
        self.left_button.setCheckable(False)
        self.addWidget(self.left_button)

        self.objective_forward_button = QPushButton('Sample forward')
        self.objective_forward_button.pressed.connect(self.objective_forward)
        self.objective_forward_button.released.connect(self.objective_stop)
        self.objective_forward_button.setShortcut("pgup")
        self.objective_forward_button.setToolTip("Moves the sample forwards towards the imaging objective. Also accessible via the page up key.")
        self.objective_forward_button.setCheckable(False)
        self.addWidget(self.objective_forward_button)

        self.objective_backward_button = QPushButton('Sample backward')
        self.objective_backward_button.pressed.connect(self.objective_backward)
        self.objective_backward_button.setToolTip("Moves the sample backwards towards the imaging objective. Also accessible via the page down key.")
        self.objective_backward_button.released.connect(self.objective_stop)
        self.objective_backward_button.setShortcut("pgdown")
        self.objective_backward_button.setCheckable(False)
        self.addWidget(self.objective_backward_button)

        self.led_button = QCheckBox()
        self.led_button.stateChanged.connect(self.toggle_led)
        self.led_button.setChecked((self.c_p['blue_led']==0))

        self.led_button.setStyleSheet("""
            QCheckBox::indicator {
                width: 30px;
                height: 30px;
            }
            """)
        led_label = QLabel("Sample LED ON/OFF")
        self.addWidget(led_label) 
        self.addWidget(self.led_button)
        

    def toggle_led(self, state):
        """
        Toggle the saving_toggled property of the DataChannel when the checkbox is toggled.
        """
        self.c_p['blue_led'] = 0 if bool(state) else 1
        if self.c_p['blue_led'] == 0:
            print("LED on")
        else:
            print("LED off")

    def set_motor_speed(self, speed):
        if np.abs(int(speed)) < 32760:
            self.motor_speed = int(speed)
        else:
            self.motor_speed = int(32760)

    def move_up(self):
        self.c_p['motor_x_target_speed'] = self.motor_speed
    def stop_y(self):
        self.c_p['motor_x_target_speed'] = 0
    def move_down(self):
        self.c_p['motor_x_target_speed'] = - self.motor_speed


    def move_right(self):
        self.c_p['motor_y_target_speed'] = self.motor_speed
    def stop_x(self):
        self.c_p['motor_y_target_speed'] = 0
    def move_left(self):
        self.c_p['motor_y_target_speed'] = - self.motor_speed

    def objective_forward(self):
        self.c_p['motor_z_target_speed'] = self.motor_speed
    def objective_stop(self):
        self.c_p['motor_z_target_speed'] = 0
    def objective_backward(self):
        self.c_p['motor_z_target_speed'] = - self.motor_speed

    


class MotorControllerWindow(QWidget):
    """
    Widget for controlling the motors of the minitweezers setup. Allows for changing the 
    speed of the motors and moving them in the x and y direction.
    """
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.motor_speed = self.c_p['minitweezers_goto_speed']
        self.y_movement = 0
        self.setWindowTitle("Motor controller")

        self.label = QLabel("Set motor speed (0-32767) a.u")
        layout.addWidget(self.label)

        self.SpeedLineEdit = QLineEdit()
        self.SpeedLineEdit.setValidator(QIntValidator(0,32767))
        self.SpeedLineEdit.setText(str(self.motor_speed))
        self.SpeedLineEdit.setToolTip("Sets the motor speed. Units are arbitrary and do not apply to movement with the mouse.")
        self.SpeedLineEdit.textChanged.connect(self.set_motor_speed)
        layout.addWidget(self.SpeedLineEdit) 
        
        self.up_button = QPushButton('LEFT')
        self.up_button.pressed.connect(self.move_up)
        self.up_button.released.connect(self.stop_y)
        self.up_button.setToolTip("Moves left in the sample. Also accessible via the left arrow key.")
        self.up_button.setCheckable(False)
        self.up_button.setShortcut("left")
        layout.addWidget(self.up_button)

        self.down_button = QPushButton('RIGHT')
        self.down_button.pressed.connect(self.move_down)
        self.down_button.released.connect(self.stop_y)
        self.down_button.setToolTip("Moves right in the sample. Also accessible via the right arrow key.")
        self.down_button.setCheckable(False)
        self.down_button.setShortcut("right")
        layout.addWidget(self.down_button)


        self.right_button = QPushButton('UP')
        self.right_button.pressed.connect(self.move_right)
        self.right_button.released.connect(self.stop_x)
        self.right_button.setToolTip("Moves up in the sample. Also accessible via the up arrow key.")   
        self.right_button.setShortcut("up")
        self.right_button.setCheckable(False)
        layout.addWidget(self.right_button)

        self.left_button = QPushButton('DOWN')
        self.left_button.pressed.connect(self.move_left)
        self.left_button.setShortcut("down")
        self.left_button.setToolTip("Moves down in the sample. Also accessible via the down arrow key.")
        self.left_button.released.connect(self.stop_x)
        self.left_button.setCheckable(False)
        layout.addWidget(self.left_button)

        self.objective_forward_button = QPushButton('Sample forward')
        self.objective_forward_button.pressed.connect(self.objective_forward)
        self.objective_forward_button.released.connect(self.objective_stop)
        self.objective_forward_button.setShortcut("pgup")
        self.objective_forward_button.setToolTip("Moves the sample forwards towards the imaging objective. Also accessible via the page up key.")
        self.objective_forward_button.setCheckable(False)
        layout.addWidget(self.objective_forward_button)

        self.objective_backward_button = QPushButton('Sample backward')
        self.objective_backward_button.pressed.connect(self.objective_backward)
        self.objective_backward_button.setToolTip("Moves the sample backwards towards the imaging objective. Also accessible via the page down key.")
        self.objective_backward_button.released.connect(self.objective_stop)
        self.objective_backward_button.setShortcut("pgdown")
        self.objective_backward_button.setCheckable(False)
        layout.addWidget(self.objective_backward_button)

        self.led_button = QCheckBox()
        self.led_button.stateChanged.connect(self.toggle_led)
        self.led_button.setChecked((self.c_p['blue_led']==0))

        self.led_button.setStyleSheet("""
            QCheckBox::indicator {
                width: 30px;
                height: 30px;
            }
            """)
        led_label = QLabel("Sample LED ON/OFF")
        layout.addWidget(led_label) 
        layout.addWidget(self.led_button)
        self.setLayout(layout)
        

    def toggle_led(self, state):
        """
        Toggle the saving_toggled property of the DataChannel when the checkbox is toggled.
        """
        self.c_p['blue_led'] = 0 if bool(state) else 1
        if self.c_p['blue_led'] == 0:
            print("LED on")
        else:
            print("LED off")

    def set_motor_speed(self, speed):
        if np.abs(int(speed)) < 32760:
            self.motor_speed = int(speed)
        else:
            self.motor_speed = int(32760)

        if self.motor_speed > 300:
            self.c_p['minitweezers_goto_speed'] = self.motor_speed

    def move_up(self):
        self.c_p['motor_x_target_speed'] = self.motor_speed
    def stop_y(self):
        self.c_p['motor_x_target_speed'] = 0
    def move_down(self):
        self.c_p['motor_x_target_speed'] = - self.motor_speed


    def move_right(self):
        self.c_p['motor_y_target_speed'] = self.motor_speed
    def stop_x(self):
        self.c_p['motor_y_target_speed'] = 0
    def move_left(self):
        self.c_p['motor_y_target_speed'] = - self.motor_speed

    def objective_forward(self):
        self.c_p['motor_z_target_speed'] = self.motor_speed
    def objective_stop(self):
        self.c_p['motor_z_target_speed'] = 0
    def objective_backward(self):
        self.c_p['motor_z_target_speed'] = - self.motor_speed

    
class ThorlabsMotorWindow(QWidget):
    # Allow for connecting and disconnecting motors from this function
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.motor_speed = 1
        self.y_movement = 0
        
        self.label = QLabel("Motor controller")
        layout.addWidget(self.label)
        
        self.SpeedLineEdit = QLineEdit()
        self.SpeedLineEdit.setValidator(QIntValidator(0,200))
        self.SpeedLineEdit.setText(str(self.motor_speed))
        self.SpeedLineEdit.textChanged.connect(self.set_motor_speed)
        layout.addWidget(self.SpeedLineEdit) 
        

        self.start_x = QPushButton('start x thread')
        self.start_x.pressed.connect(self.start_motor_x)
        self.start_x.setCheckable(False)
        col_x = 'green' if self.check_connected(0) else 'red'
        self.start_x.setStyleSheet(f"background-color : {col_x}")
        layout.addWidget(self.start_x)

        self.start_y = QPushButton('start y thread')
        self.start_y.pressed.connect(self.start_motor_y)
        self.start_y.setCheckable(False)
        col_y = 'green' if self.check_connected(1) else 'red'
        self.start_y.setStyleSheet(f"background-color : {col_y}")
        layout.addWidget(self.start_y)
        
        self.start_z = QPushButton('start z thread')
        self.start_z.pressed.connect(self.start_piezo_z)
        self.start_z.setCheckable(False)

        self.print_position_button = QPushButton('Print position')
        self.print_position_button.pressed.connect(self.print_current_pos)
        self.print_position_button.setCheckable(False)
        layout.addWidget(self.print_position_button)

        self.move_up_button = QPushButton('Move up')
        self.move_up_button.pressed.connect(self.move_up)
        self.move_up_button.setCheckable(False)
        layout.addWidget(self.move_up_button)

        self.move_down_button = QPushButton('Move down')
        self.move_down_button.pressed.connect(self.move_down)
        self.move_down_button.setCheckable(False)
        layout.addWidget(self.move_down_button)

        self.move_right_button = QPushButton('Move right')
        self.move_right_button.pressed.connect(self.move_right)
        self.move_right_button.setCheckable(False)
        layout.addWidget(self.move_right_button)

        self.move_left_button = QPushButton('Move left')
        self.move_left_button.pressed.connect(self.move_left)
        self.move_left_button.setCheckable(False)
        layout.addWidget(self.move_left_button)

        self.disconnect_motor_x_b = QPushButton('Disconnect x')
        self.disconnect_motor_x_b.pressed.connect(self.disconnect_motor_x)
        self.disconnect_motor_x_b.setCheckable(False)
        layout.addWidget(self.disconnect_motor_x_b)

        self.disconnect_motor_y_b = QPushButton('Disconnect y')
        self.disconnect_motor_y_b.pressed.connect(self.disconnect_motor_y)
        self.disconnect_motor_y_b.setCheckable(False)
        layout.addWidget(self.disconnect_motor_y_b)

        self.setLayout(layout)
        
    def start_motor_y(self):
        if self.c_p['thorlabs_threads'][1] is None:
            from ThorlabsMotor import MotorThreadV2

            self.c_p['thorlabs_threads'][1] = MotorThreadV2(channel=1, axis=1, c_p=self.c_p)
            self.c_p['thorlabs_threads'][1].start()
        elif self.c_p['steppers_connected'][1] is False:
            self.c_p['connect_steppers'][1] = True
        sleep(0.2)
        col_y = 'green' if self.check_connected(1) else 'red'
        self.start_y.setStyleSheet(f"background-color : {col_y}")
            
    def start_motor_x(self):
        if self.c_p['thorlabs_threads'][0] is None:
            from ThorlabsMotor import MotorThreadV2

            self.c_p['thorlabs_threads'][0] = MotorThreadV2(channel=0, axis=0, c_p=self.c_p)
            self.c_p['thorlabs_threads'][0].start()
            
        elif self.c_p['steppers_connected'][0] is False:
            self.c_p['connect_steppers'][0] = True
        sleep(0.2)
        col_x = 'green' if self.check_connected(0) else 'red'
        self.start_x.setStyleSheet(f"background-color : {col_x}")
        print("Color set to ", col_x)

    def disconnect_motor_x(self):
        if self.c_p['thorlabs_threads'][0] is not None:
            self.c_p['disconnect_motor'][0] = True
            sleep(0.2)
            col_x = 'green' if self.check_connected(0) else 'red'
            self.start_x.setStyleSheet(f"background-color : {col_x}")

    def disconnect_motor_y(self):
            if self.c_p['thorlabs_threads'][1] is not None:
                self.c_p['disconnect_motor'][1] = True
                sleep(0.2)
                col_y = 'green' if self.check_connected(1) else 'red'
                self.start_y.setStyleSheet(f"background-color : {col_y}")

    def check_connected(self, axis):
        if self.c_p['thorlabs_threads'][axis] is None:
            return False
        return self.c_p['steppers_connected'][axis]

    def set_motor_speed(self, speed):
        try:
            speed = float(speed) * 1e-3
        except:
            return
        if speed > 1:
            print("Speed too high")
            return
        self.c_p['stepper_max_speed'][0] = speed
        self.c_p['stepper_max_speed'][1] = speed
        self.c_p['stepper_acc'][0] = speed * 2 
        self.c_p['stepper_acc'][1] = speed * 2
        self.c_p['new_stepper_velocity_params'][0] = True
        self.c_p['new_stepper_velocity_params'][1] = True        

    def start_piezo_z(self):
        if self.c_p['thorlabs_threads'][2] is None:
            serial_no = "97100532"
            channel = 1
            from ThorlabsMotor import  PiezoThread
            self.c_p['thorlabs_threads'][2] = PiezoThread(serial_no, channel, self.c_p)
            self.c_p['thorlabs_threads'][2].start()
            

    def print_current_pos(self):
        print(f"Motors are at {self.c_p['stepper_current_position']}")

    def motor_y_pos(self):
        print(self.c_p['stepper_current_position'][1])
        
    def move_up(self):
        #if self.motor_y is not None:
        self.c_p['stepper_target_position'][1] += 0.05

    def move_down(self):
        #if self.motor_y is not None:
        self.c_p['stepper_target_position'][1] -= 0.05

    def move_left(self):
        self.c_p['stepper_target_position'][0] += 0.05

    def move_right(self):
        self.c_p['stepper_target_position'][0] -= 0.05

    def close(self):
        self.motor_y.join()
        super().close()


class MotorClickMove(MouseInterface):
    
    def __init__(self, c_p):
        self.c_p = c_p
        self.x_0 = 0
        self.y_0 = 0
        self.x_0_motor = 0
        self.y_0_motor = 0

    def mousePress(self):
        # left click
        if self.c_p['mouse_params'][0] == 1:
            center_x = int((self.c_p['camera_width']/2 - self.c_p['AOI'][0])/self.c_p['image_scale'])
            center_y = int((self.c_p['camera_height']/2 - self.c_p['AOI'][2])/self.c_p['image_scale'])
            dx_pix = (self.c_p['mouse_params'][1] - center_x) * self.c_p['image_scale']
            dy_pix = (self.c_p['mouse_params'][2] - center_y) * self.c_p['image_scale'] 
            dy_pix = (self.c_p['mouse_params'][2] - center_y) * self.c_p['image_scale'] 
            
            
            self.c_p['stepper_target_position'][0] += dx_pix * self.c_p['microns_per_pix']
            self.c_p['stepper_target_position'][1] += dy_pix * self.c_p['microns_per_pix']
            
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            self.x_0 = self.c_p['mouse_params'][1]
            self.y_0 = self.c_p['mouse_params'][2]
            self.x_0_motor = self.c_p['stepper_current_position'][0]
            self.y_0_motor = self.c_p['stepper_current_position'][1]

        # Scroll wheel
        elif self.c_p['mouse_params'][0] == 3:
            self.y_0 = self.c_p['mouse_params'][2]
        
    def mouseRelease(self):
        if self.c_p['mouse_params'][0] == 2:
            self.c_p['stepper_target_position'][0] = self.c_p['stepper_current_position'][0]
            self.c_p['stepper_target_position'][1] = self.c_p['stepper_current_position'][1]
        
    def mouseDoubleClick(self):
        pass
    
    def draw(self, qp):
        pass
    
    def mouseMove(self):
        if self.c_p['mouse_params'][0] == 2:
            dx = (self.c_p['mouse_params'][3] - self.x_0) * self.c_p['image_scale'] 
            dy = (self.c_p['mouse_params'][4] - self.y_0) * self.c_p['image_scale'] 
            self.c_p['stepper_target_position'][0] = self.x_0_motor - dx * self.c_p['microns_per_pix']
            self.c_p['stepper_target_position'][1] = self.y_0_motor - dy * self.c_p['microns_per_pix']
            print(dx* self.c_p['microns_per_pix'], dy* self.c_p['microns_per_pix'])

        elif self.c_p['mouse_params'][0] == 3:
            dy = (self.y_0 - self.c_p['mouse_params'][4])
            self.y_0 = self.c_p['mouse_params'][4]
            if np.abs(dy) > 2:
                self.c_p['z_movement'] += dy * 4
            self.c_p['z_movement'] += dy
            print(dy, self.c_p['z_current_position'])

    def getToolName(self):
        return "Thorlabs motor"

    def getToolTip(self):
        return "Move the motors by clicking or dragging on the screen"
        
class MinitweezersMouseMove(MouseInterface):
    
    def __init__(self, c_p, data_channels):
        self.c_p = c_p
        self.data_channels = data_channels # Needed to know the position and speed
        self.x_0 = 0
        self.y_0 = 0
        self.z_0 = 0
        self.x_prev = 0
        self.y_prev = 0
        self.z_prev = 0
        self.prev_t = time()
        self.speed_factor = 1000 

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

            self.c_p['minitweezers_target_pos'][0] = int(self.data_channels['Motor_x_pos'].get_data(1)[0]+dx_ticks)
            self.c_p['minitweezers_target_pos'][1] = int(self.data_channels['Motor_y_pos'].get_data(1)[0]-dy_ticks)
            print("Setting minitiwezers target pos to ", self.c_p['minitweezers_target_pos'])
            self.c_p['move_to_location'] = True
  
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            if self.x_prev == self.c_p['mouse_params'][1] and self.y_prev == self.c_p['mouse_params'][2]:
                self.c_p['motor_x_target_speed'] = 0
                self.c_p['motor_y_target_speed'] = 0
                return
            self.x_prev = self.c_p['mouse_params'][1]
            self.y_prev = self.c_p['mouse_params'][2]

        if self.c_p['mouse_params'][0] == 3:
            self.z_0 = self.c_p['mouse_params'][2]
            self.z_prev = self.c_p['mouse_params'][2]

    def mouseRelease(self):
        self.c_p['motor_x_target_speed'] = 0
        self.c_p['motor_y_target_speed'] = 0
        self.c_p['motor_z_target_speed'] = 0
        
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
            
            self.c_p['motor_x_target_speed'] = int(x_speed)
            self.c_p['motor_y_target_speed'] = int(y_speed) # Changed sign here
            self.x_prev = self.c_p['mouse_params'][3]
            self.y_prev = self.c_p['mouse_params'][4]
            
        elif self.c_p['mouse_params'][0] == 3:
            dz = (self.c_p['mouse_params'][4] - self.z_prev)
            z_speed = self.check_speed(dz * self.speed_factor/5)
            
            self.c_p['motor_z_target_speed'] = int(z_speed)
            self.z_prev = self.c_p['mouse_params'][4]

    def getToolName(self):
        return "Minitweezers motor"

    def getToolTip(self):
        return "Move the motors by clicking or dragging on the screen"
