import serial
import time
import sys

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel,QDoubleSpinBox, QGridLayout
from PyQt6.QtCore import QEvent, QThread, pyqtSignal
from time import sleep
import matplotlib.pyplot as plt
import numpy as np


COM_PORT = "COM6"
BAUD_RATE = 9600   # The baud rate of the PSU

def connect_to_psu(com_port, baud_rate):
    try:
        # Establish a serial connection to the PSU
        ser = serial.Serial(com_port, baud_rate, timeout=1)
        print(f"Connected to PSU on {com_port} at {baud_rate} baud.")

        # Example command to send (modify as per your PSU's protocol)
        ser.write(b'*IDN?\n') # Example command, replace with your PSU's specific command

        # Wait for response and read it
        time.sleep(1)  # Adjust as necessary
        response = ser.readline().decode().strip()
        print(f"Response from PSU: {response}")

        # Close the serial connection
        # ser.close()
        return ser
    except Exception as e:
        print(f"Error: {e}")

def disconnect_from_psu(ser):
    ser.close()

# Function to set voltage
def set_voltage(ser, voltage):
    command = f"VSET1:{voltage}\n".encode()  # Adjust the command as per your PSU's protocol
    ser.write(command)

# Function to set current
def set_current(ser, current):
    command = f"ISET1:{current}\n".encode()  # Adjust the command as per your PSU's protocol
    ser.write(command)

def output_on(ser):
    command = f"OUT1:\n".encode()  # Adjust the command as per your PSU's protocol
    ser.write(command)

def output_off(ser):
    command = f"OUT0:\n".encode()  # Adjust the command as per your PSU's protocol
    ser.write(command)

def read_voltage(ser):
    command = f"VOUT1?\n".encode()  # Adjust the command as per your PSU's protocol
    ser.write(command)
    voltage = ser.readline().decode().strip()
    return voltage

def read_current(ser):
    command = f"IOUT1?\n".encode()  # Adjust the command as per your PSU's protocol
    ser.write(command)
    current = ser.readline().decode().strip()
    return current

def get_status(ser):
    from time import sleep
    command = f"STATUS?\n".encode()  # Adjust the command as per your PSU's protocol
    ser.write(command)
    sleep(0.01)
    status = ser.readline().decode().strip()
    if len(status)>0:
        print("status is ", ord(status))
        sixth_bit =  ord(status) & 64
        print("sixth bit is ", sixth_bit)
    return status


class PSUControlPanel(QWidget):
    def __init__(self, COM_PORT='COM5', BAUD_RATE=9600): # Used to be COM5
        super().__init__()
        self.PSU = connect_to_psu(COM_PORT, BAUD_RATE)
        self.output_on = False
        self.protocol = [
            [0.1, 0.5],
            [0.2, 0.5],
            [0.3, 0.5],
            [0.4, 0.5],
            [0.5, 0.5],
            [0.6, 0.5],
            [0.7, 0.5],
            [0.8, 0.5],
            [0.9, 0.5],
            [1.0, 0.5],
        ]
        self.ramp_duration = 8.5
        self.max_time = 12
        self.max_current = 2.84 # 2.9 used to be default but sometimes gave closed pipettes
        self.ramp_frequency = 20
        self.voltage = 5
        self.initUI()
        self.create_protocol()
        try:
            set_voltage(self.PSU, self.voltage)
        except:
            print("Error setting voltage")

    def initUI(self):
        # Layout
        #layout = QVBoxLayout()
        layout = QGridLayout()


          # Voltage control
        self.voltage_input = QDoubleSpinBox(self)
        self.voltage_input.setRange(0, 10)  # Set voltage limits (0V to 30V, adjust as needed)
        self.voltage_input.setSingleStep(0.1)
        self.voltage_input.setValue(self.voltage)
        self.set_voltage_button = QPushButton('Set Voltage', self)
        self.set_voltage_button.clicked.connect(self.set_voltage)
        layout.addWidget(QLabel('Voltage (V):'), 0, 0)
        layout.addWidget(self.voltage_input, 0, 1)
        layout.addWidget(self.set_voltage_button, 0, 2)


        # Current control
        self.current_input = QDoubleSpinBox(self)
        self.current_input.setRange(0, 3.5)  # Set current limits (0A to 5A, adjust as needed)
        self.current_input.setSingleStep(0.1)
        self.set_current_button = QPushButton('Set Current', self)
        self.set_current_button.clicked.connect(self.set_current)
        layout.addWidget(QLabel('Current (A):'), 1, 0)
        layout.addWidget(self.current_input, 1, 1)
        layout.addWidget(self.set_current_button, 1, 2)

        # Output control
        self.output_button = QPushButton('Toggle Output', self)
        self.output_button.clicked.connect(self.toggle_output)
        self.output_button.setCheckable(True)
        self.output_button.setChecked(self.output_on)
        layout.addWidget(self.output_button, 2, 0)

        # Disconnect
        self.disconnect_button = QPushButton('Disconnect', self)
        self.disconnect_button.clicked.connect(self.disconnect)
        layout.addWidget(self.disconnect_button, 2, 1)

        # Status
        self.status_button = QPushButton('Status', self)
        self.status_button.clicked.connect(self.check_PSU_Status)
        layout.addWidget(self.status_button, 2, 2)

        self.create_protocol_button = QPushButton('Create Protocol', self)
        self.create_protocol_button.clicked.connect(self.create_protocol)
        layout.addWidget(self.create_protocol_button, 3, 0)

        # Button to start the protocol
        self.start_protocol_button = QPushButton('Start Protocol', self)
        self.start_protocol_button.clicked.connect(self.start_protocol)
        layout.addWidget(self.start_protocol_button, 3, 1)

        # Button to stop the protocol
        self.stop_protocol_button = QPushButton('Stop Protocol', self)
        self.stop_protocol_button.clicked.connect(self.stop_protocol)
        layout.addWidget(self.stop_protocol_button, 3, 2)


        layout.addWidget(QLabel('Ramp duration (S):'), 4, 0)
        self.ramp_duration_input = QDoubleSpinBox(self)
        self.ramp_duration_input.setRange(0, 100)
        self.ramp_duration_input.setSingleStep(0.1)
        self.ramp_duration_input.setValue(self.ramp_duration)
        layout.addWidget(self.ramp_duration_input, 4, 1)
        self.set_ramp_duration_button = QPushButton('Set Ramp Duration', self)
        self.set_ramp_duration_button.clicked.connect(self.set_ramp_duration)
        layout.addWidget(self.set_ramp_duration_button, 4, 2)        


        layout.addWidget(QLabel('Max time (S):'), 5, 0)
        self.max_time_input = QDoubleSpinBox(self)
        self.max_time_input.setRange(0, 100)
        self.max_time_input.setSingleStep(0.1)
        self.max_time_input.setValue(self.max_time)
        layout.addWidget(self.max_time_input, 5, 1)
        self.set_max_time_button = QPushButton('Set Max Time', self)
        self.set_max_time_button.clicked.connect(self.set_max_time)
        layout.addWidget(self.set_max_time_button, 5, 2)


        self.max_current_input = QDoubleSpinBox(self)
        self.max_current_input.setRange(0, 3.5)
        self.max_current_input.setSingleStep(0.1)
        self.max_current_input.setValue(self.max_current)
        layout.addWidget(QLabel('Max current (A):'), 6, 0)
        layout.addWidget(self.max_current_input, 6, 1)
        self.set_max_current_button = QPushButton('Set Max Current', self)
        self.set_max_current_button.clicked.connect(self.set_max_current)
        layout.addWidget(self.set_max_current_button, 6, 2)

        # Set the layout
        self.setLayout(layout)
        self.setWindowTitle('PSU Control Panel')

    def check_PSU_Status(self):
        status = get_status(self.PSU)
        

    def set_voltage(self):
        voltage = self.voltage_input.value()
        set_voltage(self.PSU , voltage)  # Assuming set_voltage function and PSU object

    def set_current(self):
        current = self.current_input.value()
        set_current(self.PSU , current)  # Assuming set_current function and PSU object

    def set_ramp_duration(self):
        ramp_duration = self.ramp_duration_input.value()
        self.ramp_duration = ramp_duration

    def set_max_time(self):
        max_time = self.max_time_input.value()
        self.max_time = max(max_time, self.ramp_duration)
    
    def set_max_current(self):
        max_current = self.max_current_input.value()
        self.max_current = max_current

    def toggle_output(self):
        # This function should toggle the PSU output on or off
        # You need to keep track of the state of the PSU output
        if self.output_on == False:
            output_on(self.PSU )
            self.output_on = True
        else:
            output_off(self.PSU )
            self.output_on = False

    def create_protocol(self):
        self.protocol = np.ones((int(self.max_time*self.ramp_frequency),2))
        self.protocol[:,1] = np.linspace(0,self.max_time, int(self.max_time*self.ramp_frequency))
        self.protocol[:,0] *= self.max_current
        ramp_voltages = np.linspace(0,self.max_current, int(self.ramp_duration*self.ramp_frequency))
        self.protocol[:len(ramp_voltages),0] = ramp_voltages
        """
        plt.plot(np.linspace(0,self.max_time,len(self.protocol[:,0])), self.protocol[:,0], label="Voltages")
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.show()
        """
    def start_protocol(self):
        if not self.output_on:
            self.toggle_output()
        self.protocol_thread = CurrentProtocolThread(self.PSU, self.protocol)
        self.protocol_thread.update_signal.connect(self.handle_protocol_update)
        self.protocol_thread.start()

    def stop_protocol(self):
        if self.protocol_thread:
            output_off(self.PSU)
            if self.output_on:
                self.toggle_output()

    def handle_protocol_update(self, message):
        # Handle updates from the protocol thread (e.g., update a status label)
       #print(message)
        pass

    def disconnect(self):
        disconnect_from_psu(self.PSU )

    def closeEvent(self, event: QEvent):
        """
        Reimplemented close event to handle PSU disconnection
        when the GUI window is closed.
        """
        # Add your PSU disconnect logic here
        disconnect_from_psu(self.PSU)
        print("PSU disconnected successfully.")
        event.accept()  # Accept the close event

class CurrentProtocolThread(QThread):
    # Signal to update the GUI or status
    update_signal = pyqtSignal(str)

    def __init__(self, PSU, protocol):
        super().__init__()
        self.PSU = PSU
        self.protocol = protocol
        self.running = False

    def run(self):
        self.running = True
        start = time.time()
        for current, timing in self.protocol:
            while time.time() - start < timing:
                sleep(0.005)
            print(time.time() - start, current)
            if not self.running:
                break
            set_current(self.PSU, current)  # Assuming set_voltage function
            self.update_signal.emit(f"Current set to {current} A")
        self.update_signal.emit("Protocol completed")
        output_off(self.PSU)

    def stop(self):
        self.running = False
        self.update_signal.emit("Protocol stopped")
        output_off(self.PSU)


def main():
    app = QApplication(sys.argv)
    ex = PSUControlPanel()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()