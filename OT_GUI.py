# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:50:13 2022

@author: Martin Selin
"""
import sys
import cv2 # Certain versions of this won't work
import pickle

from PyQt6.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QCheckBox, QComboBox, QListWidget, QLineEdit, QSpinBox,
    QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QFileDialog, QInputDialog
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QRunnable, QObject, QPoint, QRect, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QAction, QDoubleValidator, QPen, QIntValidator, QKeySequence, QFont

# from pyqtgraph import PlotWidget, plot
# import pyqtgraph as pg
from random import randint
import numpy as np
from time import sleep, time
from functools import partial
# TODO remove the things that are not used for the minitweezers such as the thorlabs motors!
import BaslerCameras
#import ThorlabsCameras # Chrashed upon opening the filemenu when this was not here.
#import clr
import win32com.client as win32  # For reasons this needs to be here for the filmenu to work (as tested on windows 11).

import asyncio # TODO finish the move to asyncio saving. May require quite a bit of work to get it to work properly.
#from concurrent.futures import ThreadPoolExecutor

from CameraControlsNew import CameraThread, VideoWriterThread, CameraClicks
from ControlParameters import default_c_p, get_data_dicitonary_new, ControlParametersViewer
from TemperatureControllerTED4015 import TemperatureThread
from TemperatureControllerWidget import TempereatureControllerWindow
from LivePlots import PlotWindow
from SaveDataWidget import SaveDataWindow
import MotorControlWidget
from QWidgetDockContainer import QWidgetWindowDocker
from LaserPiezosControlWidget import LaserPiezoWidget, MinitweezersLaserMove
from CameraMeasurementTool import CameraMeasurements
from DeepLearningThread import DeepLearningControlWidget , ParticleCNN#, DeepLearningAnalyserLDS #,  # TODO load it in a nicer way
from PlanktonViewWidget import PlanktonViewer
from DataChannelsInfoWindow import CurrentValueWindow
from ReadArduinoPortenta import PortentaComms #For some reason we need to import this to be able to open the file explorer.
from PortentaMultiprocess import PortentaComms

from PullingProtocolWidget import PullingProtocolWidget
from StepperObjective import ObjectiveStepperController, ObjectiveStepperControllerToolbar
from MicrofluidicsPumpController import MicrofluidicsControllerWidget, ElvesysMicrofluidicsController, ConfigurePumpWidget

import AutoControllerV2 as AutoController
import LaserController


# TODO put the camera window into the main window in the same manner as the other widgets. Also add the force and postiion
# windows below the main window as there is often space there to do stuff.

class Worker(QThread):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    Used to update the screen continoulsy with the images of the camera
    '''
    changePixmap = pyqtSignal(QImage)

    def __init__(self, c_p, data, test_mode=False, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.c_p = c_p
        self.data_channels = data
        self.args = args
        self.kwargs = kwargs
        self.test_mode = test_mode

        # Create different colored pens for drawing on the screen
        self.green_pen = QPen()
        self.green_pen.setColor(QColor('green'))
        self.green_pen.setWidth(3)
        self.blue_pen = QPen()
        self.blue_pen.setColor(QColor('blue'))
        self.blue_pen.setWidth(3)
        self.red_pen = QPen()
        self.red_pen.setColor(QColor('red'))
        self.red_pen.setWidth(3)
        # self.signals = WorkerSignals()

    def testDataUpdate(self, max_length=10_000):
        # Fill data dicitonary with fake data to test the interface.
        # Used only for testing
        self.dt = 1000/max_length

        if len(self.data_channels['Time'].data) < max_length:
            self.data_channels['Time'].put_data(np.linspace(0, 1000, num=max_length))
            self.data_channels['Y-force'].put_data(np.sin(self.data_channels['Time'].data / 10))
            self.data_channels['X-force'].put_data(np.cos(self.data_channels['Time'].data / 10))
            self.data_channels['Z-force'].put_data(np.cos(self.data_channels['Time'].data / 10)**2)
            self.data_channels['X-position'].put_data(self.c_p['stepper_current_position'][0])
            self.data_channels['Y-position'].put_data(self.c_p['stepper_current_position'][1])
            self.data_channels['Z-position'].put_data(np.random.rand(max_length) * 2 - 1)
            self.data_channels['Motor_position'].put_data(np.sin(self.data_channels['Time'].data / 10))
        else:
            # Shift the data
            # Update last element
            self.data_channels['Time'].put_data(self.data_channels['Time'].get_data(1) + self.dt)

            self.data_channels['Y-force'].put_data(np.sin(self.data_channels['Time'].get_data(1) / 10))
            self.data_channels['X-force'].put_data(np.cos(self.data_channels['Time'].get_data(1) / 10))
            self.data_channels['Z-force'].put_data(np.cos(self.data_channels['Time'].get_data(1) / 10)**2)

            self.data_channels['X-position'].put_data(self.c_p['stepper_current_position'][0])
            self.data_channels['Y-position'].put_data(self.c_p['stepper_current_position'][1])
            self.data_channels['Z-position'].put_data(np.random.rand() * 2 - 1)
            self.data_channels['Motor_position'].put_data((self.data_channels['Time'].get_data(1) / 10) + np.random.rand())

    def draw_particle_positions(self, centers, pen=None, radius=250, show_z_info=False):
        # TODO add function also for crosshair to help with alignment.
        rx = int(radius/self.c_p['image_scale'])
        ry = rx
        font_size = int(rx/4)

        # Create a QFont object with the desired font size
        self.qp.setFont(QFont("Arial", font_size))
        if pen is None:
            self.qp.setPen(self.red_pen)
        else:
            self.qp.setPen(pen)
        for idx, pos in enumerate(centers):
            # Adopting what we draw to the predicted radii

            try:
                # TODO add radii as a separate array as input
                rx = int(2*self.c_p['predicted_particle_radii'][idx]/self.c_p['image_scale'])
                ry = rx
            except Exception as E:
                rx = int(radius/self.c_p['image_scale'])
                ry = rx
                pass

            x = int(pos[0]/ self.c_p['image_scale'])
            y = int(pos[1]/ self.c_p['image_scale'])

            self.qp.drawEllipse(x-int(rx/2)-1, y-int(ry/2)-1, rx, ry)
                    # Check if information display is enabled
            if show_z_info:
                # You can customize this part to show whatever information you want
                # For example, showing the index of the particle or custom info passed in a list

                try:
                    if idx > len(self.c_p['z-predictions']) or len(self.c_p['z-predictions']) == 0:
                        info_text = str(idx)
                        continue
                except TypeError:
                    continue
                try:
                    info_text = 'z: ' + str(round(self.c_p['z-predictions'][idx],1))

                    # Position for the text: adjust the x, y as needed for text to not overlap the circle
                    text_x = int(x +1.1*rx)
                    text_y = int(y)

                    # Draw the text
                    self.qp.drawText(text_x, text_y, info_text)
                except Exception as E:
                    # There is an index errror here which is harmless, caused by a missing detection in the deep learning thread and a 
                    # the image being updated too quickly for the predictions to catch on. 
                    pass



    def draw_particle_positions_new(self,centers, pen=None, radii=None, info=None, info_labels=None):
        if radii is None:
            radii = [250]*len(centers)
        if len(radii)>0:
            radius = radii[0]
        else:
            radius = 250
        rx = int(100/self.c_p['image_scale'])
        ry = rx
        font_size = int(rx/4)

        # Create a QFont object with the desired font size
        self.qp.setFont(QFont("Arial", font_size))
        if pen is None:
            self.qp.setPen(self.red_pen)
        else:
            self.qp.setPen(pen)
        for idx, pos in enumerate(centers):
            # Adopting what we draw to the predicted radii

            try:
                # TODO add radii as a separate array as input
                rx = int(2*radii[idx]/self.c_p['image_scale'])
                ry = rx
            except Exception as E:
                rx = int(radius/self.c_p['image_scale'])
                ry = rx
                pass

            x = int(pos[0]/ self.c_p['image_scale'])
            y = int(pos[1]/ self.c_p['image_scale'])

            self.qp.drawEllipse(x-int(rx/2)-1, y-int(ry/2)-1, rx, ry)
                    # Check if information display is enabled
            if info is not None:
                # You can customize this part to show whatever information you want
                # TODO make it possible to show more than just numbers

                try:
                    if idx > len(info) or len(info) == 0:
                        info_text = str(idx)
                        continue
                except TypeError:
                    continue
                try:
                    info_text = info_labels + str(round(info[idx],1))

                    # Position for the text: adjust the x, y as needed for text to not overlap the circle
                    text_x = int(x +1.1*rx)
                    text_y = int(y)

                    # Draw the text
                    self.qp.drawText(text_x, text_y, info_text)
                except Exception as E:
                    # There is an index errror here which is harmless, caused by a missing detection in the deep learning thread and a 
                    # the image being updated too quickly for the predictions to catch on. 
                    pass

    def draw_pipette(self, pen=None):
        """
        Draws the bounding box of the pipette
        """
        if pen is None:
            self.qp.setPen(self.green_pen)
        else:
            self.qp.setPen(pen)
        x = int((self.c_p['pipette_location'][0] - self.c_p['pipette_location'][2] / 2) / self.c_p['image_scale'])
        y = int((self.c_p['pipette_location'][1]) / self.c_p['image_scale'])
        w = int(self.c_p['pipette_location'][2] / self.c_p['image_scale'])
        h = int(self.c_p['pipette_location'][3] / self.c_p['image_scale'])
        self.qp.drawRect(x, y, w, h)

    def preprocess_image(self):

        # Check if offset and gain should be applied.
        if self.c_p['image_offset'] != 0:
            self.image += int(self.c_p['image_offset'])
            self.image = np.uint8(self.image)
            
        if self.c_p['image_gain'] != 1:
            # TODO unacceptably slow
            pass
            self.image = np.uint8((self.image*self.c_p['image_gain']))

        #self.image = np.uint8(self.image)

    def draw_central_circle(self):
        self.blue_pen.setColor(QColor('blue'))
        cx = int((self.c_p['camera_width']/2 - self.c_p['AOI'][0])/self.c_p['image_scale'])
        cy = int((self.c_p['camera_height']/2 - self.c_p['AOI'][2])/self.c_p['image_scale'])
        rx=50
        ry=50
        self.qp.drawEllipse(cx-int(rx/2)-1, cy-int(ry/2)-1, rx, ry)

    def run(self):

        # Initialize pens to draw on the images
        """
        self.blue_pen = QPen()
        self.blue_pen.setColor(QColor('blue'))
        self.blue_pen.setWidth(2)
        self.red_pen = QPen()
        self.red_pen.setColor(QColor('red'))
        self.red_pen.setWidth(2)
        """
        while True:
            if self.test_mode:
                # TODO have this add data channels if they are not already created.
                self.testDataUpdate()

            if self.c_p['image'] is not None:
                self.image = np.array(self.c_p['image'])
            #else:
            #    print("Frame missed!")
            W, H = self.c_p['frame_size']
            self.c_p['image_scale'] = max(self.image.shape[1]/W, self.image.shape[0]/H)
            self.preprocess_image()            

            # It is quite sensitive to the format here, won't accept any missmatch
            if len(np.shape(self.image)) < 3:
                QT_Image = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_Grayscale8)
                QT_Image = QT_Image.convertToFormat(QImage.Format.Format_RGB888)
            else:                
                QT_Image = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_RGB888)
                
            picture = QT_Image.scaled(
                W,H,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            # Give other things time to work, roughly 40-50 fps default.
            sleep(0.04) # Sets the FPS
            
            # Paint extra items on the screen
            self.qp = QPainter(picture)
            # Draw zoom in rectangle
            try:
                self.c_p['click_tools'][self.c_p['mouse_params'][5]].draw(self.qp)
            except Exception as E:
                print(E)
                print(len(self.c_p['click_tools']))
                
            self.qp.setPen(self.blue_pen)
            if self.c_p['central_circle_on']:
                self.draw_central_circle()

                
            if self.c_p['draw_pipette'] and self.c_p['pipette_location'][0] is not None and self.c_p['pipette_located']:

                self.draw_pipette()

            # Draw the particles if the parameters tell us to do so
            if self.c_p['tracking_on'] and self.c_p['draw_particles']:       
                info = None
                info_labels = None
                               
                if not self.c_p['draw_TnP_particles']:

                    if self.c_p['draw_z_text']:
                        info = self.c_p['z-predictions']
                        info_labels = 'z: '
                    self.draw_particle_positions_new(
                        centers=self.c_p['predicted_particle_positions'],
                        radii=self.c_p['predicted_particle_radii'],
                        info=info,
                        info_labels=info_labels)
                else:
                    positions = []
                    radii = []
                    z_positions = []
                    # Extract positions that are not close to the trapped particle
                    for idx, pos in enumerate(self.c_p['predicted_particle_positions']):
                        dx1 = pos[0]-self.c_p['Trapped_particle_position'][0]
                        dy1 = pos[1]-self.c_p['Trapped_particle_position'][1]
                        dx2 = pos[0]-self.c_p['pipette_particle_location'][0]
                        dy2 = pos[1]-self.c_p['pipette_particle_location'][1]
                        if dx1**2 + dy1**2 < 10 and dx2**2+dy2**2 < 10:
                            positions.append(pos)
                            radii.append(self.c_p['predicted_particle_radii'][idx])
                            z_positions.append(self.c_p['z-predictions'][idx])
                    if self.c_p['draw_z_text']:
                        info = z_positions
                        info_labels = 'z: '
                    self.draw_particle_positions_new(
                        centers=positions,
                        radii=radii,
                        info=info,
                        info_labels=info_labels)

            if self.c_p['draw_TnP_particles']:

                if self.data_channels['particle_trapped'].get_data(1)[0]:
                    # gp = QPen()
                    # gp.setColor(QColor('green'))
                    # gp.setWidth(3)
                    info = None
                    info_labels = None
                    if self.c_p['draw_z_text']:
                        info = [self.c_p['Trapped_particle_position'][2]]
                        info_labels = 'z: '
                    self.draw_particle_positions_new(
                        [self.c_p['Trapped_particle_position'][0:2]],
                        radii=[self.c_p['Trapped_particle_position'][3]],
                        pen=self.green_pen,
                        info=info,
                        info_labels=info_labels,)
                
                if self.c_p['particle_in_pipette'] and self.c_p['locate_pipette'] and self.c_p['pipette_located']:
                    # tmp_pen = QPen()
                    # tmp_pen.setColor(QColor('blue'))
                    # tmp_pen.setWidth(3)
                    info = None
                    info_labels = None
                    if self.c_p['draw_z_text']:
                        info = [self.c_p['Trapped_particle_position'][2]]
                        info_labels = 'z: '
                    self.draw_particle_positions_new(
                        [self.c_p['pipette_particle_location'][0:2]],
                        radii=[self.c_p['pipette_particle_location'][3]],
                        pen=self.blue_pen,
                        info=info,
                        info_labels= info_labels,
                        )
                
            self.qp.end()
            self.changePixmap.emit(picture)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Optical tweezers: Main window")
        self.c_p = default_c_p()
        self.data_channels = get_data_dicitonary_new()
        self.video_idx = 0
        self.data_idx = 0 # Index of data save
        self.loop = asyncio.get_event_loop()

        self.saving = False
        # Start camera threads
        self.CameraThread = None
        try:
            camera = None
            camera = BaslerCameras.BaslerCamera()
            # camera if there is no camera connected.
            # camera = ThorlabsCameras.ThorlabsCamera()
            
            if camera is not None:
                self.CameraThread = CameraThread(self.c_p, camera)
                self.CameraThread.start()
        except Exception as E:
            print(f"Camera error!\n{E}")
        self.TemperatureThread = None

        self.channelView = None
        self.PortentaReaderT = None
        try:
            
            self.PortentaReaderT = PortentaComms(self.c_p, self.data_channels) #portentaReaderThread(self.c_p, self.data_channels) #portentaComms(self.c_p, self.data_channels)
            self.PortentaReaderT.start()
            sleep(0.1)
            
        except Exception as E:
            print(E)

        try:
            # TODO fix a better way to save then to send the main window to the thread.
            self.AotuControllerThread = AutoController.autoControllerThread(self.c_p, self.data_channels, main_window=self)
            self.AotuControllerThread.start()
            print("Auto controller started")
        except Exception as E:
            print(E)

        self.ArduinoUnoSerial = None
        try:
            import serial
            port = self.c_p['objective_stepper_port']
            self.ArduinoUnoSerial = serial.Serial(port, 9600)
            print("Connected to Arduino Uno.")
        except Exception as E:
            print(E)
            print("Could not connect to Arduino Uno for objective stepper control!")

        self.VideoWriterThread = VideoWriterThread(2, 'video thread', self.c_p)
        self.VideoWriterThread.start()

        #self.DeepThread = DeepLearningAnalyserLDS(self.c_p, self.data_channels)
        #self.DeepThread.start()

        self.plot_windows = None

        # Set up camera window
        H = int(1080/4)
        W = int(1920/4)
        sleep(0.5)
        self.c_p['frame_size'] = int(self.c_p['camera_width']/2), int(self.c_p['camera_height']/2)
        self.camera_window_label = QLabel("Camera window")
        self.camera_window_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setCentralWidget(self.camera_window_label)
        self.camera_window_label.setMinimumSize(W,H)
        self.painter = QPainter(self.camera_window_label.pixmap())
        th = Worker(c_p=self.c_p, data=self.data_channels)
        th.changePixmap.connect(self.setImage)
        th.start()

        # Create toolbar
        create_camera_toolbar_external(self)
        self.addToolBarBreak() 
        self.create_mouse_toolbar()

        # Create menus and drop down options
        self.menu = self.menuBar()
        self.create_filemenu()
        self.drop_down_window_menu()
        self.action_menu()
        self.ObjectiveStepperControlltoolbar= ObjectiveStepperControllerToolbar(self.c_p,self.ArduinoUnoSerial,self)
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.ObjectiveStepperControlltoolbar)

        # self.MotorControllerToolbar = MotorControlWidget.MotorControllerToolbar(self.c_p)
        # self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.MotorControllerToolbar)
        
        # Creating UI elements to be used in the main window
        self.motorWidget = MotorControlWidget.MotorControllerWindow(self.c_p)
        self.MotorControlDock = QWidgetWindowDocker(self.motorWidget, "Motor controls")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.MotorControlDock)
        
        self.LaserPiezoWidget = LaserPiezoWidget(self.c_p, self.data_channels)
        self.laserPiezoDock = QWidgetWindowDocker(self.LaserPiezoWidget, "Wiggler controls")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.laserPiezoDock)

        self.laserControllerWidget = LaserController.LaserControllerWidget(self.c_p, self)
        self.LaserControllerDock = QWidgetWindowDocker(self.laserControllerWidget, "Laser controls")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.LaserControllerDock)

        self.pullingProtolWidget = PullingProtocolWidget(self.c_p, self.data_channels)
        self.pullingProtocolDock = QWidgetWindowDocker(self.pullingProtolWidget, "Pulling protocol")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.pullingProtocolDock)
        #self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.pullingProtocolDock)

        
        self.microfluidicsController = ElvesysMicrofluidicsController()
        self.microfluidicsController.connect(self.c_p['pump_adress'])

        self.MicrofludicsWidget = MicrofluidicsControllerWidget(self.c_p, self.microfluidicsController)
        self.MicrofludicsDock = QWidgetWindowDocker(self.MicrofludicsWidget, "Microfluidics controls")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.MicrofludicsDock)
        
        self.show()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.camera_window_label.setPixmap(QPixmap.fromImage(image))

    def start_threads(self):
        pass
    
    def create_mouse_toolbar(self):
        # Here is where all the tools in the mouse toolbar are added
        self.c_p['click_tools'].append(CameraClicks(self.c_p))
        self.c_p['click_tools'].append(MotorControlWidget.MinitweezersMouseMove(self.c_p, self.data_channels))
        # self.c_p['click_tools'].append(MotorControlWidget.MotorClickMove(self.c_p,)) # Thorlabs motors and controller
        self.c_p['click_tools'].append(MinitweezersLaserMove(self.c_p))
        #self.c_p['click_tools'].append(MouseAreaSelect(self.c_p))
        self.c_p['click_tools'].append(AutoController.SelectLaserPosition(self.c_p))
        self.c_p['click_tools'].append(CameraMeasurements(self.c_p))

        self.c_p['mouse_params'][5] = 0

        self.mouse_toolbar = QToolBar("Mouse tools")
        self.addToolBar(self.mouse_toolbar)
        self.mouse_actions = []
        number_keys = [Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5, 
               Qt.Key.Key_6, Qt.Key.Key_7, Qt.Key.Key_8, Qt.Key.Key_9, Qt.Key.Key_0]

        for idx, tool in enumerate(self.c_p['click_tools']):
            self.mouse_actions.append(QAction(tool.getToolName(), self))
            self.mouse_actions[-1].setToolTip(tool.getToolTip()+"\nShortcut: Ctrl+"+str(idx+1))
            command = partial(self.set_mouse_tool, idx)
            self.mouse_actions[-1].triggered.connect(command)
            self.mouse_actions[-1].setCheckable(True)
            if idx < 10:
                self.mouse_actions[-1].setShortcut(QKeySequence(Qt.Modifier.CTRL | number_keys[idx]))
            self.mouse_toolbar.addAction(self.mouse_actions[-1])
        self.mouse_actions[self.c_p['mouse_params'][5]].setChecked(True)
        
    def set_mouse_tool(self, tool_no=0):
        if tool_no > len(self.c_p['click_tools']):
            return
        self.c_p['mouse_params'][5] = tool_no
        for tool in self.mouse_actions:
            tool.setChecked(False)
        self.mouse_actions[tool_no].setChecked(True)
        print("Tool set to ", tool_no)

    def set_gain(self, gain):
        try:
            g = min(float(gain), 255)
            self.c_p['image_gain'] = g
            print(f"Gain is now {gain}")
        except ValueError:
            # Harmless, someone deleted all the numbers in the line-edit
            pass

    def create_filemenu(self):

        file_menu = self.menu.addMenu("File")
        file_menu.addSeparator()

        # Create submenu for setting recording(video) format
        format_submenu = file_menu.addMenu("Recording format")
        video_formats = ['avi','mp4','npy']

        for f in video_formats :

            format_command= partial(self.set_video_format, f)
            format_action = QAction(f, self)
            format_action.setStatusTip(f"Set recording format to {f}")
            format_action.triggered.connect(format_command)
            format_submenu.addAction(format_action)

        # Submenu for setting the image format
        image_format_submenu = file_menu.addMenu("Image format")
        image_formats = ['png','jpg','npy']
        for f in image_formats:

            format_command= partial(self.set_image_format, f)
            format_action = QAction(f, self)
            format_action.setStatusTip(f"Set image format to {f}")
            format_action.triggered.connect(format_command)
            image_format_submenu.addAction(format_action)

        # Add command to set the savepath of the experiments.
        set_save_action = QAction("Set save path", self)
        set_save_action.setStatusTip("Set save path")
        set_save_action.triggered.connect(self.set_save_path)
        file_menu.addAction(set_save_action)

        set_filename_action = QAction("Set filename", self)
        set_filename_action.setStatusTip("Set filename for saved, data, video and image files")
        set_filename_action.triggered.connect(self.set_default_filename)
        file_menu.addAction(set_filename_action)

        # Add command to save the data
        save_data_action = QAction("Save data", self)
        save_data_action.setStatusTip("Save data to an npy file")
        save_data_action.triggered.connect(self.save_data_to_dict) # Dump data before
        file_menu.addAction(save_data_action)

    def dump_data(self):
        text, ok = QInputDialog.getText(self, 'Filename dialog', 'Set name for data to be saved:')
        if not ok:
            print("No valid name entered")
            return
        path = self.c_p['recording_path'] + '/' + text
        print(f"Saving data to {path}")
        np.save(path,  self.data_channels, allow_pickle=True)
    
    def record_data(self):
        if (self.saving):
            self.stop_saving()
            self.toggle_data_record_action.setText("Start recording data")
            self.toggle_data_record_action.setChecked(False)
            #TODO have this detecting when it is called from another file so that it sets the button to be pushed.
        else:
            self.start_saving()
            self.toggle_data_record_action.setText("Stop recording data")
            self.toggle_data_record_action.setChecked(True)

    def start_saving(self):
        self.start_idx = self.data_channels['PSD_A_P_X'].index
        self.start_idx_motors = self.data_channels['Motor_x_pos'].index # Fewer data points for motors
        self.saving = True
        self.data_idx += 1
        print("Saving started")

    async def async_stop_save(self):
        pass
        await self.run_in_executor(None, self.stop_saving())

    def stop_saving(self):
        # TODO ensure that the there is no maximum limit on the filesize
        # - too many ifs and elses to make the code nice
        # - test this carefully and see if there is some better way to do this,
        #    i.e can we handle also the other channels such as computer time efficiently here?
        # - derived channels did not get saved correctly here, check if the fix works.
        #   Need to add parameter to the channel to indicate the sampling rate of it.
        # For instance by stopping, saving and making a new file. Not the most elegant solution but it should work.
        self.saving = False
        print("Saving stopped")
        self.stop_idx = self.data_channels['PSD_A_P_X'].index
        self.stop_idx_motors = self.data_channels['Motor_x_pos'].index
        sleep(0.1) # Waiting for all channels to reach this point
        data = {}
         # TODO make sure that it saves continously and not just once at the end...
        for channel in self.data_channels:
            if self.data_channels[channel].saving_toggled:
                if channel in self.c_p['multi_sample_channels'] or channel in self.c_p['derived_PSD_channels']:
                    if self.start_idx < self.stop_idx:
                        data[channel] = self.data_channels[channel].data[self.start_idx:self.stop_idx]
                    else:
                        data[channel] = np.concatenate([self.data_channels[channel].data[self.start_idx:],
                                                        self.data_channels[channel].data[:self.stop_idx]])
                else:
                    if self.start_idx_motors < self.stop_idx_motors:
                        data[channel] = self.data_channels[channel].data[self.start_idx_motors:self.stop_idx_motors]
                    else:
                        data[channel] = np.concatenate([self.data_channels[channel].data[self.start_idx_motors:],
                                                        self.data_channels[channel].data[:self.stop_idx_motors]])

        filename = self.c_p['recording_path'] + '/' + self.c_p['filename'] + str(self.data_idx)
        with open(filename, 'wb') as f:
                pickle.dump(data, f)

    def save_data_to_dict(self):

        text, ok = QInputDialog.getText(self, 'Filename dialog', 'Set name for data to be saved:')
        if not ok:
            print("No valid name entered")
            return
        filename = self.c_p['recording_path'] + '/' + text
        self.c_p['save_idx'] = self.data_channels['PSD_A_P_X'].index
        sleep(0.1) # Make sure all channels have reached this point
        data = {}
        for channel in self.data_channels:
            if self.data_channels[channel].saving_toggled:
                data[channel] = self.data_channels[channel].get_data_spaced(1e9)
        print(f"Saving data to {filename}")
        #np.save(path, data, allow_pickle=True)
        with open(filename, 'wb') as f:
                pickle.dump(data, f)

    def action_menu(self):
        action_menu = self.menu.addMenu("Actions")

        self.save_position_action = QAction("Save position", self)
        self.save_position_action.setStatusTip("Save current position")
        self.save_position_action.triggered.connect(self.save_position)
        action_menu.addAction(self.save_position_action)

        self.action_references = {}
        self.delete_action_references = {}
        self.saved_positions_submenu = action_menu.addMenu("Go to saved positions")

        self.zero_force_action = QAction("Zero force", self)
        self.zero_force_action.setStatusTip("Zero force for current value, resets it if it's already zeroed")
        self.zero_force_action.triggered.connect(self.zero_force_PSDs)
        action_menu.addAction(self.zero_force_action)

        self.reset_force_psds_action = QAction("Reset force PSDs", self)
        self.reset_force_psds_action.setStatusTip("Reset force PSDs their default values")
        self.reset_force_psds_action.triggered.connect(self.reset_force_PSDs)
        action_menu.addAction(self.reset_force_psds_action)

        self.zero_position_action = QAction("Zero position", self)
        self.zero_position_action.setStatusTip("Zero position for current value, resets it if it's already zeroed")
        self.zero_position_action.triggered.connect(self.zero_position_PSDs)
        action_menu.addAction(self.zero_position_action)

        self.reset_position_psds_action = QAction("Reset position PSDs", self)
        self.reset_position_psds_action.setStatusTip("Reset position PSDs their default values")
        self.reset_position_psds_action.triggered.connect(self.reset_position_PSDs)
        action_menu.addAction(self.reset_position_psds_action)

        # Added some space between the delete and move to submenues to avoid accidental clicks.
        self.delete_position_submenu = action_menu.addMenu("Delete saved positions")

        for idx in range(len(self.c_p['saved_positions'])):
            self.add_position(idx)
        for idx in range(len(self.c_p['saved_positions'])):
            self.delete_positon_action(idx)
        
        # Todo add option to remove positions, set shortcut, move position etc.
        # self.remove_positions_submenu = action_menu.addMenu("Remove saved positions")
        def toggle_central_circle():
            self.c_p['central_circle_on'] = not self.c_p['central_circle_on']

        self.central_circle_button = QAction("Toggle center circle", self)
        self.central_circle_button.setStatusTip("Turns on/off the central circle used for alignment")
        self.central_circle_button.setCheckable(True)
        self.central_circle_button.setChecked(self.c_p['central_circle_on'])
        self.central_circle_button.triggered.connect(toggle_central_circle)
        action_menu.addAction(self.central_circle_button)

    def zero_force_PSDs(self):
        # Very weird to first convert to uint and then add 32768
                
        self.c_p['PSD_force_means'][0] += np.mean(self.data_channels['PSD_A_F_X'].get_data_spaced(1000))
        self.c_p['PSD_force_means'][1] += np.mean(self.data_channels['PSD_A_F_Y'].get_data_spaced(1000))
        self.c_p['PSD_force_means'][2] += np.mean(self.data_channels['PSD_B_F_X'].get_data_spaced(1000))
        self.c_p['PSD_force_means'][3] += np.mean(self.data_channels['PSD_B_F_Y'].get_data_spaced(1000))

        self.c_p['PSD_means'][0] = 32768 + np.uint16(self.c_p['PSD_force_means'][0])
        self.c_p['PSD_means'][1] = 32768 + np.uint16(self.c_p['PSD_force_means'][1])
        self.c_p['PSD_means'][2] = 32768 + np.uint16(self.c_p['PSD_force_means'][2])
        self.c_p['PSD_means'][3] = 32768 + np.uint16(self.c_p['PSD_force_means'][3])
        print(self.c_p['PSD_means'][0], self.c_p['PSD_force_means'][0])
        
        self.c_p['portenta_command_1'] = 1

        # Zeroin the Z force by resetting the null offset
        self.c_p['Photodiode_sum_to_force'][2] -= np.mean(self.data_channels['F_total_Z'].get_data_spaced(1000))

    def reset_force_PSDs(self):
        self.c_p['PSD_force_means'] = [0, 0, 0, 0]
        self.c_p['portenta_command_1'] = 2

    def zero_position_PSDs(self):
        
        self.c_p['PSD_position_means'][0] += np.mean(self.data_channels['PSD_A_P_X'].get_data_spaced(1000))
        self.c_p['PSD_position_means'][1] += np.mean(self.data_channels['PSD_A_P_Y'].get_data_spaced(1000))
        self.c_p['PSD_position_means'][2] += np.mean(self.data_channels['PSD_B_P_X'].get_data_spaced(1000))
        self.c_p['PSD_position_means'][3] += np.mean(self.data_channels['PSD_B_P_Y'].get_data_spaced(1000))

        self.c_p['PSD_means'][0] = 32768 + np.uint16(self.c_p['PSD_position_means'][0])
        self.c_p['PSD_means'][1] = 32768 + np.uint16(self.c_p['PSD_position_means'][1])
        self.c_p['PSD_means'][2] = 32768 + np.uint16(self.c_p['PSD_position_means'][2])
        self.c_p['PSD_means'][3] = 32768 + np.uint16(self.c_p['PSD_position_means'][3])
        self.c_p['portenta_command_1'] = 4

    def reset_position_PSDs(self):
        self.c_p['PSD_position_means'] = [0, 0, 0, 0]
        self.c_p['portenta_command_1'] = 5

    def add_position(self, idx):
        # Adds position to submenu
        position_command = partial(self.goto_position, idx)
        position_action = QAction(self.c_p['saved_positions'][idx][0], self)
        position_action.setStatusTip(f"Move to saved position {self.c_p['saved_positions'][idx][0]}")
        position_action.setToolTip(f"Move to saved position at: {self.c_p['saved_positions'][idx]}")
        position_action.triggered.connect(position_command)
        self.action_references[idx] = position_action
        self.saved_positions_submenu.addAction(position_action) # Check how to remove this

        self.delete_positon_action(idx)

    def remove_position(self, idx):
        print("Removing position", idx)
        if idx in self.action_references:
            position_action = self.action_references[idx]
            self.saved_positions_submenu.removeAction(position_action)
            self.delete_position_submenu.removeAction(self.delete_action_references[idx])
            del self.action_references[idx]  # Remove the reference from the dictionary
            del self.delete_action_references[idx]
            print("Action removed")

    def delete_positon_action(self, idx):
        delete_command = partial(self.remove_position, idx)
        delete_action = QAction(self.c_p['saved_positions'][idx][0], self)
        delete_action.setStatusTip(f"Delete saved position {self.c_p['saved_positions'][idx][0]}")
        delete_action.triggered.connect(delete_command)
        self.delete_position_submenu.addAction(delete_action)
        self.delete_action_references[idx] = delete_action

    def delete_position(self, idx):

        self.c_p['saved_positions'].pop(idx)

    def set_default_filename(self):
        text, ok = QInputDialog.getText(self, 'Filename dialog', 'Enter name of your files:')
        if ok:
            self.video_idx = 0
            self.data_idx = 0
            self.c_p['image_idx'] = 0
            self.c_p['filename'] = text
            self.c_p['video_name'] = text + '_video_' + str(self.video_idx)
            print(f"Filename is now {text}")

    def save_position(self):
        if not self.c_p['minitweezers_connected']:
            print("Minitweezers not connected")
            x = self.c_p['stepper_current_position'][0]
            y = self.c_p['stepper_current_position'][1]
            z = 0
        else:
            x = self.data_channels['Motor_x_pos'].get_data(1)[0]
            y = self.data_channels['Motor_y_pos'].get_data(1)[0]
            z = self.data_channels['Motor_z_pos'].get_data(1)[0]

        text, ok = QInputDialog.getText(self, 'Save position dialog', 'Enter name of position:')
        if ok:
            self.c_p['saved_positions'].append([text, x, y, z])
            print(f"Saved position {x}, {y} ,{z} as position: {text}")
            self.add_position(len(self.c_p['saved_positions'])-1)
        else:
            print("No position saved")



    def goto_position(self,idx):
        print(f"Moving to position {self.c_p['saved_positions'][idx][0]}")
        # TODO replace this with the generic go2position from the autocontroller.
        if idx>len(self.c_p['saved_positions']):
            return
        if self.c_p['move_to_location']:
            self.c_p['move_to_location'] = False
            self.c_p['motor_x_target_speed'] = 0
            self.c_p['motor_y_target_speed'] = 0
            self.c_p['motor_z_target_speed'] = 0
            return
        if self.c_p['minitweezers_connected']:
            self.c_p['minitweezers_target_pos'][0] = int(self.c_p['saved_positions'][idx][1]) # Added +32768
            self.c_p['minitweezers_target_pos'][1] = int(self.c_p['saved_positions'][idx][2])
            self.c_p['minitweezers_target_pos'][2] = int(self.c_p['saved_positions'][idx][3])
            self.c_p['move_to_location'] = True
        else:
            self.c_p['stepper_target_position'][0:2] = self.c_p['saved_positions'][idx][1:3]

    def drop_down_window_menu(self):
        # Create windows drop down menu
        window_menu = self.menu.addMenu("Windows")
        window_menu.addSeparator()

        self.open_plot_window = QAction("Live plotter", self)
        self.open_plot_window.setToolTip("Open live plotting window.")
        self.open_plot_window.triggered.connect(self.show_new_window)
        self.open_plot_window.setCheckable(False)
        window_menu.addAction(self.open_plot_window)

        self.open_positions_window = QAction("Position PSDs", self)
        self.open_positions_window.setToolTip("Open window for position PSDs.\n This is a specially configured version of the live plotter")
        self.open_positions_window.triggered.connect(self.open_Position_PSD_window)
        self.open_positions_window.setCheckable(False)
        window_menu.addAction(self.open_positions_window)


        # TODO add a window for default stretching analysis(poistion vs force)
        self.open_force_window = QAction("Force PSDs", self)
        self.open_force_window.setToolTip("Open window for force PSDs.\n This is a specially configured version of the live plotter")
        self.open_force_window.triggered.connect(self.open_Force_PSD_window)
        self.open_force_window.setCheckable(False)
        window_menu.addAction(self.open_force_window)

        self.force_distance_window_action = QAction("Force distance curve", self)
        self.force_distance_window_action.setToolTip("Opens a plotting window with average laser position reading and y-force")
        self.force_distance_window_action.triggered.connect(self.open_Force_Distance_window)
        self.force_distance_window_action.setCheckable(False)
        window_menu.addAction(self.force_distance_window_action)
        
        self.open_motor_window = QAction("Minitweezers motor window", self)
        self.open_motor_window.setToolTip("Open window for manual motor control.")
        self.open_motor_window.triggered.connect(self.open_motor_control_window)
        self.open_motor_window.setCheckable(False)
        window_menu.addAction(self.open_motor_window)

        self.open_stepper_window = QAction("Objective motor window", self)
        self.open_stepper_window.setToolTip("Open window for manual motor control, objective stepper motor.")
        self.open_stepper_window.triggered.connect(self.open_stepper_objective)
        self.open_stepper_window.setCheckable(False)
        window_menu.addAction(self.open_stepper_window)

        self.open_deep_window = QAction("DL window", self)
        self.open_deep_window.setToolTip("Open window for deep learning control.")
        self.open_deep_window.triggered.connect(self.OpenDeepLearningWindow)
        self.open_deep_window.setCheckable(False)
        window_menu.addAction(self.open_deep_window)
        
        self.open_laser_piezo_window_action = QAction("Laser piezos window", self)
        self.open_laser_piezo_window_action.setToolTip("Open window controlling piezos of lasers.")
        self.open_laser_piezo_window_action.triggered.connect(self.OpenLaserPiezoWidget)
        self.open_laser_piezo_window_action.setCheckable(False)
        window_menu.addAction(self.open_laser_piezo_window_action)
        
        self.open_channel_viewer = QAction("Data channels", self)
        self.open_channel_viewer.setToolTip("Opens a separate window in which the current values \n of the data channels is displayed.")
        self.open_channel_viewer.triggered.connect(self.open_channels_winoow)
        self.open_channel_viewer.setCheckable(False)
        window_menu.addAction(self.open_channel_viewer)

        self.open_controlparameters_viewer_action = QAction("Control parameters", self)
        self.open_controlparameters_viewer_action.setToolTip("Opens a window for viewing.")
        self.open_controlparameters_viewer_action.triggered.connect(self.open_controlparameters_viewer)
        self.open_controlparameters_viewer_action.setCheckable(False)
        window_menu.addAction(self.open_controlparameters_viewer_action)

        self.auto_controller_action = QAction("Auto controller", self)
        self.auto_controller_action.setToolTip("Opens a window for interfacing the auto controller.")
        self.auto_controller_action.triggered.connect(self.openAutoControllerWindow)
        self.auto_controller_action.setCheckable(False)
        window_menu.addAction(self.auto_controller_action)

        self.open_laser_window_action = QAction("Laser controller", self)
        self.open_laser_window_action.setToolTip("Opens a window for interfacing the laser controller.")
        self.open_laser_window_action.triggered.connect(self.open_laser_window)
        self.open_laser_window_action.setCheckable(False)
        window_menu.addAction(self.open_laser_window_action)

        self.open_pulling_protocoL_window  = QAction("Pulling protocol", self)
        self.open_pulling_protocoL_window.setToolTip("Opens a window for the pulling protocol.")
        self.open_pulling_protocoL_window.triggered.connect(self.openPullingProtocolWindow)
        self.open_pulling_protocoL_window.setCheckable(False)
        window_menu.addAction(self.open_pulling_protocoL_window)

        self.open_microfluidics_window = QAction("Microfluidics controller", self)
        self.open_microfluidics_window.setToolTip("Opens a window for the microfluidics controller.")
        self.open_microfluidics_window.triggered.connect(self.open_microfluidics_window_func)
        self.open_microfluidics_window.setCheckable(False)
        window_menu.addAction(self.open_microfluidics_window)

        self.open_pump_config_window = QAction("Pump auto configuration", self)
        self.open_pump_config_window.setToolTip("Opens a window for configuring the pump for autonomous use, selecting which channel is connected to what type of particles.")
        self.open_pump_config_window.triggered.connect(self.openPumpConfigWidget)
        self.open_pump_config_window.setCheckable(False)
        window_menu.addAction(self.open_pump_config_window)

    def openPlanktonViwer(self):
        self.planktonView = PlanktonViewer(self.c_p)

    def open_channels_winoow(self):
        self.channelView = CurrentValueWindow(self.c_p, self.data_channels)
        self.channelView.show()

    def open_controlparameters_viewer(self):
        self.controlparametersView = ControlParametersViewer(self.c_p)
        self.controlparametersView.show()

    def set_video_format(self, video_format):
        self.c_p['video_format'] = video_format

    def open_motor_control_window(self):
        #TODO all these opening methods are really basically the same, factor them out into a single one
        self.MotorControlDock.show()
        if self.MotorControlDock.isFloating():
            self.MotorControlDock.setFloating(False)

    def open_laser_window(self):

        self.LaserControllerDock.show()
        if self.LaserControllerDock.isFloating():
            self.LaserControllerDock.setFloating(False)

    def open_stepper_objective(self):
        #self.obective_controller = ObjectiveStepperController(self.c_p, self.ArduinoUnoSerial)
        #self.obective_controller.show()
        pass

    def openPullingProtocolWindow(self):
        self.pullingProtocolDock.show()
        if self.pullingProtocolDock.isFloating():
            self.pullingProtocolDock.setFloating(False)

    def open_microfluidics_window_func(self):
        self.MicrofludicsDock.show()
        if self.MicrofludicsDock.isFloating():
            self.MicrofludicsDock.setFloating(False)

    def OpenLaserPiezoWidget(self):
        self.laserPiezoDock.show()
        if self.laserPiezoDock.isFloating():
            self.laserPiezoDock.setFloating(False)

    def open_thorlabs_motor_control_window(self):
        self.MCW_T = MotorControlWidget.ThorlabsMotorWindow(self.c_p)
        self.MCW_T.show()

    def set_image_format(self, image_format):
        self.c_p['image_format'] = image_format
        
    def set_video_name(self, string):
        self.c_p['video_name'] = string

    def set_exposure_time(self):
        # Updates the exposure time of the camera to what is inside the textbox
        self.c_p['exposure_time'] = float(self.exposure_time_LineEdit.text())
        self.c_p['new_settings_camera'] = [True, 'exposure_time']

    def set_frame_rate(self):
        # Updates the frame rate of the camera to what is inside the textbox
        self.c_p['target_frame_rate'] = float(self.frame_rate_LineEdit.text())
        self.c_p['new_settings_camera'] = [True, 'frame_rate']

    def set_save_path(self):
        fname = QFileDialog.getExistingDirectory(self, "Save path")
        if len(fname) > 3:
            # If len is less than 3 then the action was cancelled and we should not update
            # the path.
            self.c_p['recording_path'] = fname

    def ZoomOut(self):
        self.c_p['AOI'] = [0, self.c_p['camera_width'], 0,
                   self.c_p['camera_height']]
        self.c_p['new_settings_camera'] = [True, 'AOI']

    def ToggleRecording(self):
        # Turns on/off recording
        # Need to add somehting to indicate the number of frames left to save when recording.
        self.c_p['recording'] = not self.c_p['recording']
        if self.c_p['recording']:
            self.c_p['video_name'] = self.c_p['filename'] + '_video' + str(self.video_idx)
            self.video_idx += 1
            self.record_action.setToolTip("Turn OFF recording.\n can also be toggled with CTRL+R")
        else:
            self.record_action.setToolTip("Turn ON recording.\n can also be toggled with CTRL+R")

    def toggle_tracking_view(self):
        self.c_p['draw_particles'] = not self.c_p['draw_particles']

    def toggle_pipette_view(self):
        self.c_p['draw_pipette'] = not self.c_p['draw_pipette']
    
    def toggle_z_text(self):
        self.c_p['draw_z_text'] = not self.c_p['draw_z_text']
    
    def toggle_TnP_view(self):
        self.c_p['draw_TnP_particles'] = not self.c_p['draw_TnP_particles']

    def snapshot(self):
        # Captures a snapshot of what the camera is viewing and saves that
        # in the fileformat specified by the image_format parameter.
        idx = str(self.c_p['image_idx'])
        filename = self.c_p['recording_path'] + '/'+self.c_p['filename']+'image_' + idx +'.'+\
            self.c_p['image_format']
        if self.c_p['image_format'] == 'npy':
            np.save(filename[:-4], self.c_p['image'])
        else:
            cv2.imwrite(filename, cv2.cvtColor(self.c_p['image'],
                                           cv2.COLOR_RGB2BGR))

        self.c_p['image_idx'] += 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_image()

    def resize_image(self):
        # New code for it
        current_size = self.camera_window_label.size()
        width = current_size.width()
        height = current_size.height()
        self.c_p['frame_size'] = width, height

    def mouseMoveEvent(self, e):
        self.c_p['mouse_params'][3] = e.pos().x()-self.camera_window_label.pos().x()
        self.c_p['mouse_params'][4] = e.pos().y()-self.camera_window_label.pos().y()
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mouseMove()

    def mousePressEvent(self, e):
        
        self.c_p['mouse_params'][1] = e.pos().x()-self.camera_window_label.pos().x()
        self.c_p['mouse_params'][2] = e.pos().y()-self.camera_window_label.pos().y()

        if e.button() == Qt.MouseButton.LeftButton:
            self.c_p['mouse_params'][0] = 1
        if e.button() == Qt.MouseButton.RightButton:
            self.c_p['mouse_params'][0] = 2
        if e.button() == Qt.MouseButton.MiddleButton:
            self.c_p['mouse_params'][0] = 3
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mousePress()

    def mouseReleaseEvent(self, e):

        self.c_p['mouse_params'][3] = e.pos().x()-self.camera_window_label.pos().x()
        self.c_p['mouse_params'][4] = e.pos().y()-self.camera_window_label.pos().y()
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mouseRelease()
        self.c_p['mouse_params'][0] = 0


    def mouseDoubleClickEvent(self, e):
        # Double click to move center?
        x = e.pos().x()-self.camera_window_label.pos().x()
        y = e.pos().y()-self.camera_window_label.pos().y()
        print(x*self.c_p['image_scale'] ,y*self.c_p['image_scale'] )
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mouseDoubleClick()


    def show_new_window(self, checked):
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p, data=self.data_channels,
                                          x_keys=['T_time'], y_keys=['PSD_A_P_X']))

        self.plot_windows[-1].show()

    def open_Position_PSD_window(self):
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p, data=self.data_channels,
                                          x_keys=['PSD_A_P_X','PSD_B_P_X'], y_keys=['PSD_A_P_Y','PSD_B_P_Y'],
                                          aspect_locked=True, grid_on=True, title='Position PSDs'))
        self.plot_windows[-1].show()

    def open_Force_PSD_window(self):
        if self.plot_windows is None:
            self.plot_windows = []
        # TODO fix that A and B are mixed up here.
        self.plot_windows.append(PlotWindow(self.c_p, data=self.data_channels,
                                          x_keys=['PSD_A_F_X','PSD_B_F_X'], y_keys=['PSD_A_F_Y','PSD_B_F_Y'],
                                          aspect_locked=True, grid_on=True, title='Force PSDs'))
        self.plot_windows[-1].show()

    def open_Force_Distance_window(self):
        if self.plot_windows is None:
            self.plot_windows = []
        # TODO fix that A and B are mixed up here.
        self.plot_windows.append(PlotWindow(self.c_p, data=self.data_channels,
                                          x_keys=['Position_Y'], y_keys=['F_total_Y'],
                                          aspect_locked=False, grid_on=True, title='Force vs distance',
                                          default_plot_length=200_000))
        self.plot_windows[-1].show()

    def OpenTemperatureWindow(self):
        self.temp_control_window = TempereatureControllerWindow(self.c_p)
        self.temp_control_window.show()
        
    def OpenPIStage(self):
        self.PI_window = PIStageWidget(self.c_p)
        self.PI_window.show()

    def openPumpConfigWidget(self):
        self.pump_config_window = ConfigurePumpWidget(self.c_p)
        self.pump_config_window.show()

    def openAutoControllerWindow(self):
        self.auto_controller_window = AutoController.AutoControlWidget(self.c_p, self.data_channels)#, MainWindow=self)
        self.auto_controller_window.show()

    def DataWindow(self):
        self.data_window= SaveDataWindow(self.c_p, self.data_channels)
        self.data_window.show()

    def OpenDeepLearningWindow(self):
        self.dep_learning_window = DeepLearningControlWidget(self.c_p)
        self.dep_learning_window.show()



    def closeEvent(self, event):
        # TODO close also other widgets here
        if self.plot_windows is not None:
            for w in self.plot_windows:
                w.close()
        self.loop.close()
        self.__del__


    def __del__(self):
        self.c_p['program_running'] = False
        # TODO organize this better
        if self.CameraThread is not None:
            self.CameraThread.join()
        if self.TemperatureThread is not None:
            self.TemperatureThread.join()
        if self.PortentaReaderT is not None:
            self.PortentaReaderT.join()
        if self.PICReaderT is not None:
            self.PICReaderT.join()
        if self.PICWriterT is not None:
            self.PICWriterT.join()

        if self.ArduinoUnoSerial is not None and self.ArduinoUnoSerial.is_open:
            self.ArduinoUnoSerial.close()

        self.VideoWriterThread.join()

def create_camera_toolbar_external(main_window):
    # TODO do not have this as an external function, urk
    main_window.camera_toolbar = QToolBar("Camera tools")
    main_window.addToolBar(main_window.camera_toolbar)

    # main_window.add_camera_actions(main_window.camera_toolbar)
    main_window.zoom_action = QAction("Zoom out", main_window)
    main_window.zoom_action.setToolTip("Resets the field of view of the camera.\n CTRL+O")
    main_window.zoom_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_O))
    main_window.zoom_action.triggered.connect(main_window.ZoomOut)
    main_window.zoom_action.setCheckable(False)

    main_window.record_action = QAction("Record video", main_window)
    main_window.record_action.setToolTip("Turn ON recording.\n CTRL+R")
    main_window.record_action.setShortcut('Ctrl+R')
    main_window.record_action.triggered.connect(main_window.ToggleRecording)
    main_window.record_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_R))

    main_window.record_action.setCheckable(True)

    main_window.snapshot_action = QAction("Snapshot", main_window)
    main_window.snapshot_action.setToolTip("Take snapshot of camera view.\n CTRL+S")
    #main_window.snapshot_action.setShortcut('Shift+S')
    main_window.snapshot_action.triggered.connect(main_window.snapshot)
    main_window.snapshot_action.setCheckable(False)
    #self.button.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_D))
    # Create a shortcut and connect it to a custom method
    main_window.snapshot_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_S))

    main_window.set_exp_tim = QAction("Set exposure time", main_window)
    main_window.set_exp_tim.setToolTip("Sets exposure time to the value in the textboox")
    main_window.set_exp_tim.triggered.connect(main_window.set_exposure_time)

    main_window.camera_toolbar.addAction(main_window.zoom_action)
    main_window.camera_toolbar.addAction(main_window.record_action)
    main_window.camera_toolbar.addAction(main_window.snapshot_action)

    # Add a toggle buttons for drawing
    main_window.toggle_tracking_view_action = QAction("Toggle tracking view", main_window)
    main_window.toggle_tracking_view_action.setToolTip("Toggle whether the tracking should be drawn in real-time, on/off")
    main_window.toggle_tracking_view_action.setCheckable(True)
    main_window.toggle_tracking_view_action.triggered.connect(main_window.toggle_tracking_view)
    main_window.camera_toolbar.addAction(main_window.toggle_tracking_view_action)

    main_window.toggle_pipette_view_action = QAction("Toggle pipette view", main_window)
    main_window.toggle_pipette_view_action.setToolTip("Toggle if pipette should be drawn, can be on/off")
    main_window.toggle_pipette_view_action.setCheckable(True)
    main_window.toggle_pipette_view_action.triggered.connect(main_window.toggle_pipette_view)
    main_window.camera_toolbar.addAction(main_window.toggle_pipette_view_action)

    main_window.toggle_z_text_action = QAction("Toggle z text", main_window)
    main_window.toggle_z_text_action.setToolTip("Toggle if z text should be drawn, can be on/off")
    main_window.toggle_z_text_action.setCheckable(True)
    main_window.toggle_z_text_action.triggered.connect(main_window.toggle_z_text)
    main_window.camera_toolbar.addAction(main_window.toggle_z_text_action)
    
    main_window.exposure_time_LineEdit = QLineEdit()
    main_window.exposure_time_LineEdit.setValidator(QDoubleValidator(0.99,99.99,2))
    main_window.exposure_time_LineEdit.setText(str(main_window.c_p['exposure_time']))
    main_window.camera_toolbar.addWidget(main_window.exposure_time_LineEdit)
    main_window.camera_toolbar.addAction(main_window.set_exp_tim)


    main_window.set_frame_rate_action = QAction("Set target fps", main_window)
    main_window.set_frame_rate_action.setToolTip("Sets frame rate to the value in the textboox,\n is an upper bound on the actual frame rate.")
    main_window.set_frame_rate_action.triggered.connect(main_window.set_frame_rate)

    main_window.frame_rate_LineEdit = QLineEdit()
    main_window.frame_rate_LineEdit.setValidator(QDoubleValidator(0.1,99.99,2))
    main_window.frame_rate_LineEdit.setText(str(main_window.c_p['target_frame_rate']))
    main_window.camera_toolbar.addWidget(main_window.frame_rate_LineEdit)
    main_window.camera_toolbar.addAction(main_window.set_frame_rate_action)


    main_window.set_gain_action = QAction("Set gain", main_window)
    main_window.set_gain_action.setToolTip("Sets software gain to the value in the textboox")
    main_window.set_gain_action.triggered.connect(main_window.set_gain)

    # TODO add offset and label to this        
    main_window.gain_LineEdit = QLineEdit()
    main_window.gain_LineEdit.setToolTip("Set software gain on displayed image.")
    main_window.gain_LineEdit.setValidator(QDoubleValidator(0.1,3,3))
    main_window.gain_LineEdit.setText(str(main_window.c_p['image_gain']))
    #main_window.gain_LineEdit.textChanged.connect(main_window.set_gain)
    main_window.camera_toolbar.addWidget(main_window.gain_LineEdit)
    main_window.camera_toolbar.addAction(main_window.set_gain_action)

    main_window.toggle_data_record_action = QAction("Start saving data", main_window)
    main_window.toggle_data_record_action.setToolTip(f"Turn ON recodording of data.\nData will be saved to fileneame set in files windows.\n Can save a maximum of {main_window.data_channels['T_time'].max_len} data points before overwriting old ones.\n CTRL+D")
    main_window.toggle_data_record_action.setCheckable(True)
    main_window.toggle_data_record_action.triggered.connect(main_window.record_data)
    main_window.toggle_data_record_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_D))
    main_window.camera_toolbar.addAction(main_window.toggle_data_record_action)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()
    w.c_p['program_running'] = False
