# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:00:04 2022

@author: marti
"""


import sys
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QDoubleValidator, QIntValidator
sys.path.append("..")

import numpy as np
from functools import partial

class SaveDataWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, c_p, data):
        super().__init__()
        self.c_p = c_p
        self.data = data
        layout = QVBoxLayout()
        self.label = QLabel("Data selector")
        layout.addWidget(self.label)

        self.save_channels = {}
        
        for channel in self.data:
            self.output_button = QPushButton(channel)
            data_command = partial(self.ToggleOutput, channel)
            self.output_button.clicked.connect(data_command)
            self.output_button.setCheckable(True)
            self.output_button.setChecked(False)            
            self.save_channels[channel] = False
            layout.addWidget(self.output_button)

        self.setLayout(layout)
    def ToggleOutput(self, channel,_):
        print("output yeah")
        self.save_channels[channel] = not self.save_channels[channel]
        print(f"Channel {channel} set to {self.save_channels[channel]}.")
        
    def save_data(self):
        pass
