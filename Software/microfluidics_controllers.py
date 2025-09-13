from typing import Protocol, runtime_checkable, Mapping

from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from typing import Mapping
from enum import Enum


class ValveState(Enum):
    CLOSED = 0
    OPEN = 1

@runtime_checkable
class MicrofluidicsController(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def set_pressure(self, channel: str, value_kpa: float) -> None: ...
    def get_pressure(self, channel: str) -> float: ...
    def get_number_channels(self) -> int: ...


@runtime_checkable
class ValveController(Protocol):
    def connect(self, address: str | None = None) -> None: ...
    def is_connected(self) -> bool: ...
    def toggle_valve(self, valve_id: str, state: "ValveState") -> None: ...
    def get_valve_states(self) -> Mapping[str, "ValveState"]: ...


@runtime_checkable
class PipettePump(Protocol):
    def connect(self, address: str | None = None) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def set_power(self, power: float) -> None: ...
    def get_power(self) -> float: ...
    def activate_suction(self) -> float: ...
    def deactivate_suction(self) -> float: ...
    def suction_active(self) -> bool: ...

class TestMicrofluidicsController(MicrofluidicsController):
    def __init__(self):
        self.connected = False
        self.num_channels = 3
        self.pressures = [0]*self.num_channels
        self.verbose = False

    def connect(self, adress=None):
        self.connected = True
        print("Connected to test pump controller")

    def disconnect(self):
        self.connected = False
        print("Disconnected from test pump controller")

    def set_pressure(self, channel, pressure):
        if not self.connected:
            raise Exception("Not connected to pump controller")
        if channel < 1 or channel > self.num_channels:
            raise Exception("Channel out of range")
        if pressure < 0 or pressure > 2000:
            raise Exception("Pressure out of range")
        self.pressures[channel-1] = pressure

        if self.verbose:
            print(f"Set pressure of channel {channel} to {pressure} mbar")

    def get_pressure(self, channel):
        if not self.connected:
            raise Exception("Not connected to pump controller")
        if channel < 1 or channel > self.num_channels:
            raise Exception("Channel out of range")
        return self.pressures[channel-1]

    def get_number_channels(self):
        return self.num_channels


class TestValveController(ValveController):

    def __init__(self):
        self.connected = False
        self.valve_states = {}
        self.verbose = False

    def connect(self, address=None):
        self.connected = True
        print("Connected to test valve controller")

    def is_connected(self):
        return self.connected

    def toggle_valve(self, valve_id, state: ValveState):
        if not self.connected:
            raise Exception("Not connected to valve controller")
        self.valve_states[valve_id] = state
        if self.verbose:
            print(f"Set valve {valve_id} to state {state}")

    def get_valve_states(self):
        if not self.connected:
            raise Exception("Not connected to valve controller")
        return self.valve_states
    

class TestPipettePump(PipettePump):

    def __init__(self):
        self.connected = False
        self.power = 0
        self.suction_active_state = False
        self.verbose = False
    
    def connect(self, address=None):
        self.connected = True
        print("Connected to test pipette pump")

    def disconnect(self):
        self.connected = False
        print("Disconnected from test pipette pump")

    def is_connected(self):
        return self.connected

    def set_power(self, power:float):
        if not self.connected:
            raise Exception("Not connected to pipette pump")
        if power < 0 or power > 100:
            raise Exception("Power out of range")
        self.power = power
        if self.verbose:
            print(f"Set power to {power}%")

    def get_power(self) -> float:
        if not self.connected:
            raise Exception("Not connected to pipette pump")
        return self.power
    
    def activate_suction(self):

        if not self.connected:
            raise Exception("Not connected to pipette pump")
        self.suction_active_state = True
        print(f"Activated suction")
    
    def suction_active(self):
        return self.suction_active_state

    def deactivate_suction(self):
        if not self.connected:
            raise Exception("Not connected to pipette pump")
        self.suction_active_state = False
        if self.verbose:
            print(f"Deactivated suction")

class MicrofluidicsMonitorThread(QThread):
    # Define signals to communicate with the main thread
    finished = pyqtSignal()
    progress = pyqtSignal(list)

    def __init__(self, microfluidicsController, valve_controller, c_p, pipette_pump=None):
        super().__init__()
        self.microfluidicsController = microfluidicsController
        self.c_p = c_p
        self.pipette_pump = pipette_pump
        self.valve_controller = valve_controller
        # self.valve_controller.connect(adress=self.c_p['valve_adress'])

    def set_pressures(self):

        for channel in range(self.microfluidicsController.get_number_channels()):
            # Indexing starts at 1 in the controller. Also 0 and 1 map to the same channel.
            self.microfluidicsController.set_pressure(
                channel+1, self.c_p['target_pressures'][channel])

    def get_pressures(self):
        for channel in range(self.microfluidicsController.get_number_channels()):
            self.c_p['current_pressures'][channel] = self.microfluidicsController.get_pressure(channel+1)
    
    def check_pipette_pump(self):
        self.pipette_pump.set_power(self.c_p['pipette_pump_target_power'])
        if self.c_p['pipette_pump_on']:
            self.pipette_pump.activate_suction()
        else:
            self.pipette_pump.deactivate_suction()
        self.c_p['pipette_pump_current_power'] = self.pipette_pump.get_power()

    def run(self):
        # Place your background task here
        while self.c_p['program_running']:
            self.set_pressures()
            self.get_pressures()

            # Set the valves to the correct state
            # self.c_p['valves_controller_connected'] = self.valve_controller.valve_connected
            if self.valve_controller.is_connected():
                for index in self.c_p['valves_used']:
                    self.valve_controller.toggle_valve(index, self.c_p['valves_open'][index])
                # self.c_p['valve_controller'].set_valve_states() # Not needed?
            
            if self.pipette_pump is not None and self.pipette_pump.is_connected():
                self.check_pipette_pump()
            self.progress.emit(self.c_p['current_pressures'])
            QThread.msleep(500) # Sleep for specified number of milliseconds
        self.finished.emit()


class ConfigurePumpWidget(QWidget):
    """
    Widget used to change the settings of the fluidics channels. Specifically to
    configure for the autonomous system which channels contain which particles and what
    is a reasonable pressure to use when flowing particles trough the capillaries.
    """

    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(225, 225, 250))
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
        self.capillary_1_channel_spinbox.valueChanged.connect(
            lambda value: self.set_capillary_1_channel(value))
        self.layout.addWidget(self.capillary_1_channel_spinbox)

        self.capillary_1_max_pressure_input = QDoubleSpinBox()
        self.capillary_1_max_pressure_input.setRange(0, 2000)
        self.capillary_1_max_pressure_input.setValue(self.c_p['capillary_1_fluidics_channel'][1])
        self.capillary_1_max_pressure_input.valueChanged.connect(
            lambda value: self.set_capillary_1_flow_pressure(value))
        self.layout.addWidget(self.capillary_1_max_pressure_input)

        self.capillary_1_valve_label = QLabel("Capillary 1 Valve")
        self.layout.addWidget(self.capillary_1_valve_label)
        self.capillary_1_valve_input = QSpinBox()
        self.capillary_1_valve_input.setRange(0, 7)
        self.capillary_1_valve_input.setValue(self.c_p['capillary_1_fluidics_channel'][2])
        self.capillary_1_valve_input.valueChanged.connect(
            lambda value: self.set_capillary_1_valve(value))
        self.layout.addWidget(self.capillary_1_valve_input)

        self.capillary_2_label = QLabel("Capillary 2")
        self.layout.addWidget(self.capillary_2_label)
        self.capillary_2_channel_spinbox = QSpinBox()
        self.capillary_2_channel_spinbox.setRange(1, 3)
        self.capillary_2_channel_spinbox.setValue(self.c_p['capillary_2_fluidics_channel'][0])
        self.capillary_2_channel_spinbox.valueChanged.connect(
            lambda value: self.set_capillary_2_channel(value))
        self.layout.addWidget(self.capillary_2_channel_spinbox)

        self.capillary_2_max_pressure_input = QDoubleSpinBox()
        self.capillary_2_max_pressure_input.setRange(0, 2000)
        self.capillary_2_max_pressure_input.setValue(self.c_p['capillary_2_fluidics_channel'][1])
        self.capillary_2_max_pressure_input.valueChanged.connect(
            lambda value: self.set_capillary_2_flow_pressure(value))
        self.layout.addWidget(self.capillary_2_max_pressure_input)

        self.capillary_2_valve_label = QLabel("Capillary 2 Valve")
        self.layout.addWidget(self.capillary_2_valve_label)
        self.capillary_2_valve_input = QSpinBox()
        self.capillary_2_valve_input.setRange(0, 7)
        self.capillary_2_valve_input.setValue(self.c_p['capillary_2_fluidics_channel'][2])
        self.capillary_2_valve_input.valueChanged.connect(
            lambda value: self.set_capillary_2_valve(value))
        self.layout.addWidget(self.capillary_2_valve_input)

        self.main_label = QLabel("Central Channel")
        self.layout.addWidget(self.main_label)
        self.main_channel_spinbox = QSpinBox()
        self.main_channel_spinbox.setRange(1, 3)
        self.main_channel_spinbox.setValue(self.c_p['central_fluidics_channel'][0])
        self.main_channel_spinbox.valueChanged.connect(
            lambda value: self.set_main_channel(value))
        self.layout.addWidget(self.main_channel_spinbox)

        self.main_max_pressure_input = QDoubleSpinBox()
        self.main_max_pressure_input.setRange(0, 2000)
        self.main_max_pressure_input.setValue(self.c_p['central_fluidics_channel'][1])
        self.main_max_pressure_input.valueChanged.connect(
            lambda value: self.set_main_flow_pressure(value))
        self.layout.addWidget(self.main_max_pressure_input)

        self.central_valve_label = QLabel("Central Valve")
        self.layout.addWidget(self.central_valve_label)
        self.main_valve_input = QSpinBox()
        self.main_valve_input.setRange(0, 7)
        self.main_valve_input.setValue(self.c_p['central_fluidics_channel'][2])
        self.main_valve_input.valueChanged.connect(
            lambda value: self.set_main_valve(value))
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

    def __init__(self, c_p, microfluidicsController=None, valve_controller=None, pipette_pump=None):
        super().__init__()
        self.c_p = c_p
        self.microfluidicsController = microfluidicsController
        self.valve_controller = valve_controller        

        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(225, 225, 250))
        self.setPalette(pal)


        self.initUI()
        self.pipette_pump = pipette_pump 
        self.pumpMonitorThread = MicrofluidicsMonitorThread(
            self.microfluidicsController, self.valve_controller, self.c_p, self.pipette_pump)
        self.pumpMonitorThread.progress.connect(self.update_pressures)
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
        self.pipette_pump_max_power_spinbox = QDoubleSpinBox()
        self.pipette_pump_max_power_spinbox.setRange(0, 12)
        self.pipette_pump_max_power_spinbox.setSingleStep(0.1)
        self.pipette_pump_max_power_spinbox.setSuffix(" V")
        self.pipette_pump_max_power_spinbox.setValue(self.c_p['pipette_pump_current_power'])
        self.pipette_pump_max_power_spinbox.valueChanged.connect(
            lambda value: self.set_pipette_pump_power(value))
        self.layout.addWidget(self.pipette_pump_max_power_spinbox)

         # TODO have this react to external events
        self.toggle_pipette_pump_button = QPushButton("Toggle Pump PSU")
        self.toggle_pipette_pump_button.clicked.connect(lambda: self.toggle_pipette_pump())
        self.toggle_pipette_pump_button.setCheckable(True)
        self.toggle_pipette_pump_button.setChecked(self.c_p['pipette_pump_on'])
        self.layout.addWidget(self.toggle_pipette_pump_button)

        self.layout.addWidget(QLabel("Valves"))
        self.valve_buttons = []
        for valve_index in self.c_p['valves_used']:
            self.valve_buttons.append(QPushButton(f"Valve {valve_index}"))
            self.valve_buttons[-1].setCheckable(True)
            self.valve_buttons[-1].setChecked(self.c_p['valves_open'][valve_index])
            self.valve_buttons[-1].clicked.connect(
                lambda checked, valve_index=valve_index: self.toggle_valve(valve_index))
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

        for channel in range(self.microfluidicsController.get_number_channels()):
            # Create a label for the channel
            label = QLabel("Channel " + str(channel+1))
            self.layout.addWidget(label)

            # Create a spinbox for setting the pressure
            self.pressure_spinboxes.append(QDoubleSpinBox())
            self.pressure_spinboxes[-1].setRange(0, 2000)
            self.pressure_spinboxes[-1].setSingleStep(0.1)
            self.pressure_spinboxes[-1].setSuffix(" mbar")
            self.pressure_spinboxes[-1].valueChanged.connect(
                lambda value, channel=channel: self.set_pressure(channel, value))
            self.layout.addWidget(self.pressure_spinboxes[-1])

            # Create a label for monitoring the pressure
            self.pressure_monitor_labels.append(
                QLabel(f"Pressure {self.c_p['current_pressures'][channel]} mbar"))
            self.layout.addWidget(self.pressure_monitor_labels[-1])

    def update_pressures(self, values):
        for channel in range(self.microfluidicsController.get_number_channels()):
            self.pressure_monitor_labels[channel].setText(
                f"Pressure {self.c_p['current_pressures'][channel]} mbar")
            self.pressure_spinboxes[channel].setValue(self.c_p['target_pressures'][channel])

    def set_pressure(self, channel, pressure):
        self.c_p['target_pressures'][channel] = float(pressure)

    def set_pipette_pump_power(self, value):
        self.c_p['pipette_pump_target_power'] = value

    def toggle_pipette_pump(self):
        self.c_p['pipette_pump_on'] = self.toggle_pipette_pump_button.isChecked()

    def refresh(self):
        
        self.pipette_pump_max_power_spinbox.setValue(self.c_p['pipette_pump_current_power'])
        self.toggle_pipette_pump_button.setChecked(self.c_p['pipette_pump_on'])

        for button, index in zip(self.valve_buttons, self.c_p['valves_used']):
            button.setChecked(self.c_p['valves_open'][index])

    def closeEvent(self, event):
        event.accept()
        self.pipette_pump.disconnect_from_psu()
            
