# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:02:39 2022

@author: marti
"""
from CameraControlsNew import CameraInterface
import clr
import sys
# from System import Int32
import nicelib
import win32com.client as win32  # TODO check which of theese are necessary
sys.path.append('C:/Program Files/Thorlabs/Scientific Imaging/DCx Camera Support/Develop/Lib')
sys.path.append("C:/Program Files/Thorlabs/Scientific Imaging/DCx Camera Support/Develop/DotNet/signed")
from instrumental import instrument, list_instruments, drivers, u, Q_
# https://instrumental-lib.readthedocs.io/en/stable/uc480-cameras.html ->
# Link to website with the camera package


def number_to_millisecond(nbr):
    return str(nbr)+'ms'


class ThorlabsCamera(CameraInterface):

    def __init__(self):
        self.exposure_time = 100
        self.live_on = False
        self.cam = None

    def connect_camera(self):
        instr = list_instruments()
        if len(instr) < 1:
            return False
        self.cam = drivers.cameras.uc480.UC480_Camera()
        return True

    def disconnect_camera(self):
        self.cam.close()

    def set_exposure_time(self, exposure_time):
        self.cam.stop_live_video()
        self.live_on = False

        exposure_time = float(exposure_time+1e-4)
        e_t = number_to_millisecond(exposure_time)
        self.cam.set_defaults(exposure_time=e_t)

    def set_AOI(self, AOI):
        self.cam.stop_live_video()
        self.live_on = False
        left = AOI[0] - AOI[0] % 32
        right = AOI[1] - AOI[1] % 32
        top = AOI[2] - AOI[2] % 16
        bot = AOI[3] - AOI[3] % 16
        if right-left < 32 or bot - top < 16:
            return

        self.cam.set_defaults(left=left, right=right, top=top,
                              bot=bot)

    def get_fps(self):
        return float(str(self.cam.framerate)[0:9])

    def get_exposure_time(self):
        exp = self.cam._get_exposure()
        return float(str(exp)[0:7])

    def get_sensor_size(self):
        width = int(self.cam.max_width)
        height = int(self.cam.max_height)
        return width, height

    def capture_image(self):
        if not self.live_on:
            self.cam.start_live_video()
            self.live_on = True
        self.cam.wait_for_frame(timeout=None)
        return self.cam.latest_frame()[:, :]
