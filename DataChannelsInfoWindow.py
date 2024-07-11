
from PyQt6.QtWidgets import (
 QCheckBox, QVBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem
)

# from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtCore import QTimer

import numpy as np

class CurrentValueWindow(QWidget):
    """
    Window for displaying the current values of the data channels.
    """
    def __init__(self, c_p, data_channels):
        # TODO add checkbox for saving data(continously) and a button to save data snapshot
        super().__init__()
        self.c_p = c_p
        self.data_channels = data_channels
        self.setWindowTitle("Data channels viewer")
        self.resize(800, 800)

        self.vBox = QVBoxLayout()
        self.CreateTable()
        self.vBox.addWidget(self.table)        

        self.timer = QTimer()
        self.timer.setInterval(500) # sets the fps of the timer
        self.timer.timeout.connect(self.set_data)

        self.timer.start()
        self.setLayout(self.vBox)   
    '''
    def CreateTable(self):

        self.table = QTableWidget(len(self.data_channels)+1, 6)
        self.table.setHorizontalHeaderLabels(["Channel", "Value", "Mean", "Standard dev", "Unit", "Save"])

        for idx, channel in enumerate(self.data_channels):
            self.table.setItem(idx, 0, QTableWidgetItem(f"{self.data_channels[channel].name}"))
            self.table.setItem(idx, 1, QTableWidgetItem(f"{self.data_channels[channel].get_data(1)}"))
            data = self.data_channels[channel].get_data(self.c_p['averaging_interval'])
            if data is not None and len(data) > 1:
                self.table.setItem(idx, 2, QTableWidgetItem(f"{np.mean(data)}"))
            else:
                self.table.setItem(idx, 2, QTableWidgetItem(""))

            if data is not None and len(data) > 1:
                self.table.setItem(idx, 3, QTableWidgetItem(f"{np.std(data)}"))
            else:
                self.table.setItem(idx, 3, QTableWidgetItem(""))
            self.table.setItem(idx,4, QTableWidgetItem(f"{self.data_channels[channel].unit}"))

            # Create a QCheckBox, connect it to the toggle_save method and add it to the table
            save_checkbox = QCheckBox()
            save_checkbox.stateChanged.connect(lambda state, ch=channel: self.toggle_save(state, ch))
            save_checkbox.setChecked(self.data_channels[channel].saving_toggled)

            save_checkbox.setStyleSheet("""
                QCheckBox::indicator {
                    width: 30px;
                    height: 30px;
                }
                """)
            self.table.setCellWidget(idx, 5, save_checkbox)
    '''
    def CreateTable(self):

        self.table = QTableWidget(len(self.data_channels)+2, 6)  # +2 now for the fps row and a buffer
        self.table.setHorizontalHeaderLabels(["Channel", "Value", "Mean", "Standard dev", "Unit", "Save"])

        for idx, channel in enumerate(self.data_channels):
            self.table.setItem(idx, 0, QTableWidgetItem(f"{self.data_channels[channel].name}"))
            self.table.setItem(idx, 1, QTableWidgetItem(f"{self.data_channels[channel].get_data(1)}"))
            data = self.data_channels[channel].get_data(self.c_p['averaging_interval'])
            if data is not None and len(data) > 1:
                self.table.setItem(idx, 2, QTableWidgetItem(f"{np.mean(data)}"))
            else:
                self.table.setItem(idx, 2, QTableWidgetItem(""))

            if data is not None and len(data) > 1:
                self.table.setItem(idx, 3, QTableWidgetItem(f"{np.std(data)}"))
            else:
                self.table.setItem(idx, 3, QTableWidgetItem(""))
            self.table.setItem(idx,4, QTableWidgetItem(f"{self.data_channels[channel].unit}"))

            # Create a QCheckBox, connect it to the toggle_save method and add it to the table
            save_checkbox = QCheckBox()
            save_checkbox.stateChanged.connect(lambda state, ch=channel: self.toggle_save(state, ch))
            save_checkbox.setChecked(self.data_channels[channel].saving_toggled)

            save_checkbox.setStyleSheet("""
                QCheckBox::indicator {
                    width: 30px;
                    height: 30px;
                }
                """)
            self.table.setCellWidget(idx, 5, save_checkbox)

        # Set the 'fps' value in the last row
        fps_idx = len(self.data_channels)
        self.table.setItem(fps_idx, 0, QTableWidgetItem("FPS"))
        self.table.setItem(fps_idx, 1, QTableWidgetItem(f"{self.c_p['fps']}"))
        for i in range(2, 6):  # Clear the rest of the columns for the FPS row
            self.table.setItem(fps_idx, i, QTableWidgetItem(""))

    def toggle_save(self, state, channel):
        """
        Toggle the saving_toggled property of the DataChannel when the checkbox is toggled.
        """
        self.data_channels[channel].saving_toggled = bool(state)
    '''
    def set_data(self):
        
        for idx, channel in enumerate(self.data_channels):
            data = self.data_channels[channel].get_data(self.c_p['averaging_interval'])
            self.table.setItem(idx,1, QTableWidgetItem(f"{self.data_channels[channel].get_data(1)}"))
            if data is not None and len(data) > 1:
                self.table.setItem(idx,2, QTableWidgetItem(f"{round(np.mean(data),6)}"))
            else:
                self.table.setItem(idx,2, QTableWidgetItem(f"{data}"))
            if data is not None and len(data) > 1:
                self.table.setItem(idx,3, QTableWidgetItem(f"{round(np.std(data),6)}"))
            else:
                self.table.setItem(idx,3, QTableWidgetItem(f"{data}"))
    '''
    def set_data(self):
        
        for idx, channel in enumerate(self.data_channels):
            data = self.data_channels[channel].get_data(self.c_p['averaging_interval'])
            self.table.setItem(idx,1, QTableWidgetItem(f"{self.data_channels[channel].get_data(1)}"))
            if data is not None and len(data) > 1:
                self.table.setItem(idx,2, QTableWidgetItem(f"{round(np.mean(data),6)}"))
            else:
                self.table.setItem(idx,2, QTableWidgetItem(f"{data}"))
            if data is not None and len(data) > 1:
                self.table.setItem(idx,3, QTableWidgetItem(f"{round(np.std(data),6)}"))
            else:
                self.table.setItem(idx,3, QTableWidgetItem(f"{data}"))

        # Refresh the 'fps' value in the last row
        fps_idx = len(self.data_channels)
        self.table.setItem(fps_idx, 1, QTableWidgetItem(f"{self.c_p['fps']}"))