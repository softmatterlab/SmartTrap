# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:33:55 2022

@author: marti
"""
import numpy as np
from CameraControlsNew import CameraInterface
from pypylon import pylon  # For basler camera, maybe move that out of here
from time import sleep


class TimeoutException(Exception):
    print("Timeout of camera!")

class BaslerCamera(CameraInterface):

    def __init__(self):
        self.capturing = False
        self.img = pylon.PylonImage()
        self.is_grabbing = False

    def capture_image(self):
        if not self.is_grabbing:
            self.cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.is_grabbing = True
        try:
            with self.cam.RetrieveResult(3000) as result: # 3000
                self.img.AttachGrabResultBuffer(result)
                if result.GrabSucceeded():
                    # Consider if we need to put directly in c_p?
                    image = np.uint8(self.img.GetArray())
                    self.img.Release()  # This was commented out recently 
                    return image
        except TimeoutException as TE:
            print(f"Warning, camera timed out {TE}")

        except Exception as ex:
            """
            Here we are catching all other exceptions, this is not good practice but done to catch when the camera
            overheats.
            Reconnecting is not trivial but ususally works after a few tries. The best way to prevent it is to lower the
            framerate. This did not happen before improving the illumination and therby the framerate.
            """
            print(f"Warning, camera error!\n {ex}")
            print("Trying to reconnect camera\n Camera may be overheating! Consider lowering framerate!")
            self.disconnect_camera()
            sleep(0.5)
            self.connect_camera()

    def connect_camera(self):
        try:
            tlf = pylon.TlFactory.GetInstance()
            self.cam = pylon.InstantCamera(tlf.CreateFirstDevice())
            self.cam.Open()
            sleep(0.2)
            return True
        except Exception as ex:
            self.cam = None
            print(ex)
            return False
        
    def disconnect_camera(self):
        self.stop_grabbing()
        self.cam.Close()
        self.cam = None

    def stop_grabbing(self):
        try:
            self.cam.StopGrabbing()
        except Exception as ex:
            print(ex)
            pass
        self.is_grabbing = False

    def set_frame_rate(self, frame_rate):
        # TODO make it possible to disable frame rate
        print("setting framerate")
        self.cam.AcquisitionFrameRateEnable = True
        self.cam.AcquisitionFrameRateEnable.SetValue(True)
        
        try:
            #self.cam.AcquisitionFrameRate = frame_rate
            self.cam.AcquisitionFrameRate.SetValue(float(frame_rate))
        except Exception as ex:
            print(f"Frame rate not accepted by camera, {ex}")
        

    def set_AOI(self, AOI):
        '''
        Function for setting AOI of basler camera to c_p['AOI']
        '''
        self.stop_grabbing()
        try:
            '''
            The order in which you set the size and offset parameters matter.
            If you ever get the offset + width greater than max width the
            camera won't accept your valuse. Thereof the if-else-statements
            below. Conditions might need to be changed if the usecase of this
            funciton change.
            '''
            # TODO test with a real sample
            width = int(AOI[1] - AOI[0])
            offset_x = AOI[0]
            height = int(AOI[3] - AOI[2])
            offset_y = AOI[2]
            width -= width % 16
            height -= height % 16
            offset_x -= offset_x % 16
            offset_y -= offset_y % 16
            self.video_width = width
            self.video_height = height
            self.cam.OffsetX = 0
            self.cam.OffsetY = 0
            sleep(0.1)
            self.cam.Width = width
            self.cam.Height = height
            self.cam.OffsetX = offset_x
            self.cam.OffsetY = offset_y

        except Exception as ex:
            print(f"AOI not accepted, AOI: {AOI}, error {ex}")

    def set_exposure_time(self, exposure_time):
        self.stop_grabbing()
        try:
            self.cam.ExposureTime = exposure_time

        except Exception as ex:
            print(f"Exposure time not accepted by camera, {ex}")

    def get_exposure_time(self):

        return self.cam.ExposureTime()

    def get_fps(self):
        fps = round(float(self.cam.ResultingFrameRate.GetValue()), 1)
        return fps

    def get_sensor_size(self):
        width = int(self.cam.Width.GetMax())
        height = int(self.cam.Height.GetMax())
        return width, height
