
from PyQt6.QtWidgets import (
    QPushButton, QVBoxLayout, QWidget, QLabel,QToolBar
)
from PyQt6.QtWidgets import  QMenu
from PyQt6.QtGui import QAction

class ObjectiveStepperController(QWidget):



    
    """
    Simple widget to control the objective stepper motor. Can move the stepper motor
    either towards or away from the sample. Can move slowly or fast (big or small steps).
    """
    def __init__(self, c_p, ArduinoUnoSerial):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        self.last_write = "Q"
        self.ArduinoUnoSerial = ArduinoUnoSerial

        #self.label = QLabel("Objective controller")
        #layout.addWidget(self.label)
        self.setWindowTitle("Objective controller")
        
        # Add move towards button
        self.slow_towards_button = QPushButton('Slow towards sample')
        self.slow_towards_button.pressed.connect(self.slow_towards_sample)
        self.slow_towards_button.setCheckable(False)
        layout.addWidget(self.slow_towards_button)

        # Add move away button
        self.slow_away_button = QPushButton('Slow away from sample')
        self.slow_away_button.pressed.connect(self.slow_away_from_sample)   
        self.slow_away_button.setCheckable(False)
        layout.addWidget(self.slow_away_button)

        # Add move fast towards button
        self.fast_towards_button = QPushButton('Fast towards sample')
        self.fast_towards_button.pressed.connect(self.fast_towards_sample)
        self.fast_towards_button.setCheckable(False)
        layout.addWidget(self.fast_towards_button)

        # Add move fast away button
        self.fast_away_sample_button = QPushButton('Fast away from sample')
        self.fast_away_sample_button.pressed.connect(self.fast_away_from_sample)
        self.fast_away_sample_button.setCheckable(False)
        layout.addWidget(self.fast_away_sample_button)

        self.setLayout(layout)
    
    def slow_towards_sample(self):
        self.last_write = 'Q'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)
        #print('Moving slowly towards sample.')

    def slow_away_from_sample(self):
        self.last_write = 'W'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)
        #print('Moving slowly away from sample.')

    def fast_towards_sample(self):
        self.last_write = 'E'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)
        #print('Moving fast towards sample.')

    def fast_away_from_sample(self):
        self.last_write = 'R'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)
        #print('Moving fast away from sample.')

class ObjectiveStepperControllerToolbar(QToolBar):
    """
    Simple toolbar to control the objective stepper motor. Can move the stepper motor
    either towards or away from the sample. Can move slowly or fast (big or small steps).
    """
    def __init__(self, c_p, ArduinoUnoSerial, parent):
        super().__init__("Objective Controller", parent)
        self.c_p = c_p
        self.last_write = "Q"
        self.ArduinoUnoSerial = ArduinoUnoSerial
        
        # Add move towards action
        self.slow_towards_action = QAction('Slow towards sample')
        self.slow_towards_action.triggered.connect(self.slow_towards_sample)
        self.addAction(self.slow_towards_action)

        # Add move away action
        self.slow_away_action = QAction('Slow away from sample')
        self.slow_away_action.triggered.connect(self.slow_away_from_sample)   
        self.addAction(self.slow_away_action)

        # Add move fast towards action
        self.fast_towards_action = QAction('Fast towards sample')
        self.fast_towards_action.triggered.connect(self.fast_towards_sample)
        self.addAction(self.fast_towards_action)

        # Add move fast away action
        self.fast_away_sample_action = QAction('Fast away from sample')
        self.fast_away_sample_action.triggered.connect(self.fast_away_from_sample)
        self.addAction(self.fast_away_sample_action)

        #parent.addToolBar(self)
    
    def slow_towards_sample(self):
        self.last_write = 'Q'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)

    def slow_away_from_sample(self):
        self.last_write = 'W'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)

    def fast_towards_sample(self):
        self.last_write = 'E'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)

    def fast_away_from_sample(self):
        self.last_write = 'R'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)