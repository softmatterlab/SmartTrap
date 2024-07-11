# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:45:53 2022

@author: martin Selin
"""
import abc
import os
import cv2 # Certain versions of this won't work
import tkinter
from tkinter import simpledialog
from time import sleep, strftime, perf_counter
from threading import Thread
from copy import copy, deepcopy
# from queue import Queue

import skvideo.io
# run this command to install ffmpeg properly : "conda install ffmpeg -c mrinaljain17"


import numpy as np
from CustomMouseTools import MouseInterface
from PyQt6.QtGui import  QColor,QPen


class CameraClicks(MouseInterface):
    # TODO add measurements here
    def __init__(self, c_p):
        self.c_p = c_p
        self.x_0 = 0
        self.y_0 = 0
        self.red_pen = QPen()
        self.red_pen.setColor(QColor('red'))
        self.red_pen.setWidth(2)

    def draw(self, qp):
        if self.c_p['mouse_params'][0] == 1:
            # TODO use mouse params [0] to index different tools.
            # self.qp.setBrush(QColor(255, 255, 0, 20))#self.br)
            qp.setPen(self.red_pen)                
            x1,y1,x2,y2 = self.c_p['mouse_params'][1:5]
            qp.drawRect(x1,y1,x2-x1,y2-y1)
            return

    def mousePress(self):

        # left click
        if self.c_p['mouse_params'][0] == 1:
            pass
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            pass
        
    def mouseRelease(self):
        if self.c_p['mouse_params'][0] != 1:
            return
        x0, y0, x1, y1 = self.c_p['mouse_params'][1:5]
        dx = x1 - x0
        dy = y1 - y0
        if dx**2 < 100 or dy**2 < 100:
            print(dx,dy)
            return
        left = int(x0 * self.c_p['image_scale'])
        right = int(x1 *self.c_p['image_scale'])
        if right < left:
            tmp = right
            right = left
            left = tmp
        up = int(y0 * self.c_p['image_scale'])
        down = int(y1 * self.c_p['image_scale'])
        if up < down:
            tmp = up
            up = down
            down = tmp

        self.c_p['AOI'] = [self.c_p['AOI'][0] + left,self.c_p['AOI'][0] + right,
                           self.c_p['AOI'][2] + down,self.c_p['AOI'][2] + up]
        self.c_p['new_settings_camera'] = [True, 'AOI']
        
    def mouseDoubleClick(self):
        pass
    
    def mouseMove(self):
        if self.c_p['mouse_params'][0] == 2:
            pass
        
    def getToolName(self):
        return "Camera tool"

    def getToolTip(self):
        return "Use the mouse to zoom in on the screen."
        


class CameraInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'connect_camera') and
                callable(subclass.connect_camera) and
                hasattr(subclass, 'set_AOI') and
                callable(subclass.set_AOI) and
                hasattr(subclass, 'get_sensor_size') and
                callable(subclass.get_sensor_size) and
                hasattr(subclass, 'set_exposure_time') and
                callable(subclass.set_exposure_time) and
                hasattr(subclass, 'set_frame_rate') and
                callable(subclass.set_frame_rate) and
                hasattr(subclass, 'capture_image') and
                callable(subclass.capture_image) or
                NotImplemented)

    @abc.abstractmethod
    def connect_camera(self):
        """Connect to the camera"""
        raise NotImplementedError

    @abc.abstractmethod
    def capture_image(self):
        """Capture a single image"""
        raise NotImplementedError


class CameraThread(Thread):

    def __init__(self, c_p, camera):
        """
        Initiates a camera thread.

        Parameters
        ----------
        c_p : TYPE
            DESCRIPTION. Control parameters used to get commands from the GUI
            or from an automation procedure. Also transmits imformation the
            other direction.
        camera : TYPE
            DESCRIPTION. A camera object implementing the CameraInterface.

        Returns
        -------
        None.

        """
        Thread.__init__(self)
        self.camera = camera
        # TODO check that camera implements the correct interface
        self.camera.connect_camera()
        c_p['camera_width'], c_p['camera_height'] = camera.get_sensor_size()
        self.c_p = c_p

        # TODO remove temporary solution
        
        self.camera.cam.AcquisitionFrameRateEnable = False
        # self.camera.cam.AcquisitionFrameRate = 11 
        print("Framrate ", self.camera.cam.ResultingFrameRate())
        
        # Zoom out
        self.c_p['AOI'] = [0, self.c_p['camera_width'], 0,
                   self.c_p['camera_height']]
        self.c_p['new_settings_camera'] = [True, 'AOI']
        self.setDaemon(True)

    def update_camera_settings(self):
        if self.c_p['new_settings_camera'][1] == 'AOI':
            self.camera.set_AOI(self.c_p['AOI'])
        elif self.c_p['new_settings_camera'][1] == 'exposure_time':
            self.camera.set_exposure_time(self.c_p['exposure_time'])
        elif self.c_p['new_settings_camera'][1] == 'frame_rate':
            self.camera.set_frame_rate(self.c_p['target_frame_rate'])

        # Resetting the new_settings_camera parameter
        self.c_p['new_settings_camera'] = [False, None]

    def run(self):
        self.c_p['exposure_time'] = self.camera.get_exposure_time()
        count = 0
        while self.c_p['program_running']:
            if self.c_p['new_settings_camera'][0]:
                self.update_camera_settings()
            count += 1
            if count % 20 == 5:
                p_t = perf_counter()
            self.c_p['image'] = self.camera.capture_image()
            if self.c_p['image'] is None:
                print("None image error!!?!?")
            if self.c_p['recording']:
                img = copy(self.c_p['image'])
                name = copy(self.c_p['video_name'])
                # TODO add also the time for the frame captured as measured on the controller.
                self.c_p['frame_queue'].put([img, name,
                                             self.c_p['video_format']])
            if count % 20 == 15:
                self.c_p['fps'] = 11 / (perf_counter()-p_t)


class VideoFormatError(Exception):
    """
    Raised when a video format is not supported.
    """
    pass


def create_avi_video_writer(c_p, video_name, image_width, image_height):
    '''
    Funciton for creating a VideoWriter.
    Will also save the relevant parameters of the experiments.
    Returns
    -------
    video : VideoWriter
        A video writer for creating a video.
    experiment_info_name : String
        Name of experiment being run.
    exp_info_params : Dictionary
        Dictionary with controlparameters describing the experiment.
    '''
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    
    video_name = c_p['recording_path'] + '/' + video_name + '.avi'
    if len(np.shape(c_p['image'])) > 2 and np.shape(c_p['image'])[2] == 3:
        video = cv2.VideoWriter(video_name, fourcc, min(500, c_p['fps']),
                                (image_height, image_width), isColor=True)
    else:
        video = cv2.VideoWriter(video_name, fourcc, min(500, c_p['fps']),
                                (image_height, image_width), isColor=False)
    return video


def create_mp4_video_writer(c_p, video_name=None, image_width=None,
                            image_height=None):
    """
    Crates a high quality video writer for lossless recording.
    """

    if video_name is None:
        video_name = get_video_name(c_p=c_p)
    tmp = min(500, int(c_p['fps']))
    frame_rate = str(max(25, tmp))  # Can in principle reach 500fps
    if tmp < 25:
        print('Warning, skvideo cannot handle framerates below 25 fps so\
        reverting to 25.')
    # TODO figure out why this does not work on all computers and how to fix it.
    video_name = c_p['recording_path'] + '/' + video_name + '.mp4'
    # TODO fix so that exceptions in recording path can be handled
    video = skvideo.io.FFmpegWriter(video_name, outputdict={
                                     '-b': c_p['bitrate'],
                                     '-r': frame_rate,  # Does not like this
                                     # specifying codec and bitrate,
                                     # 'vcodec': 'libx264',
                                    })
    return video

def npy_generator(path):
    """
    Used to read all the images in a npy image folder one at a time. Takes the
    full path as input and outputs an image. Outputs None if there are no more
    images to read.
    """

    def sorting_key(file_name):
        """Key function to sort files based on the starting number."""
        start_num = file_name.split('-')[0]
        return int(start_num)

    directory = sorted(os.listdir(path), key=sorting_key)  # Sort the files using the custom key
    done = False
    num = '-1'  # First frame to load

    while not done:
        done = True
        for file in directory:
            print(file)
            idx = file.find('-')
            print(str(int(file[:idx])),str(int(num)+1))
            if file[:idx] == str(int(num)+1) and file[-4:] == '.npy':
                images = np.load(os.path.join(path, file))  # Use os.path.join() for better compatibility
                num = file[idx+1:-4]
                done = False
                for image in images:
                    yield image
    while True:
        yield None


def get_video_name(c_p, base_name=''):
    """
    Returns an auto-generated name of the video. The name has the time of
    creation in the title to be easy to locate.
    """
    import datetime
    now = datetime.now()
    # base_name +
    print(c_p['measurement_name'], base_name)
    name = 'video-' + c_p['measurement_name'] + '-' + str(now.hour)
    name += '-' + str(now.minute) + '-' + str(now.second)+'-fps-'
    name += str(c_p['fps'])
    return name


class VideoWriterThread(Thread):
    """
    A class which simply deques the latest frame and prints it to a video
    """

    def __init__(self, threadID, name, c_p):
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.c_p = c_p
        self.setDaemon(True)

        self.sleep_time = 0.1
        self.frame = None
        self.video_width = np.shape(self.c_p['image'])[0]
        self.video_height = np.shape(self.c_p['image'])[1]
        self.format = self.c_p['video_format']
        self.last_frame_format = self.format
        self.video_created = False
        # TODO update video name to avoid overwriting old videos/photos
        self.video_name = self.c_p['video_name']
        self.frame_buffer = []
        self.frame_buffer_size = 100
        self.frame_count = 0
        self.VideoWriter = None
        self.np_save_path = None

    def close_video(self):
        """
        Closes and the current video and deletes the python object.

        Returns
        -------
        None.

        """
        self.video_created = False
        try:
            if self.last_frame_format == 'mp4':
                if self.VideoWriter is not None:
                    self.VideoWriter.close()
                del self.VideoWriter
            elif self.last_frame_format == 'avi':
                if self.VideoWriter is not None:
                    self.VideoWriter.release()
                    print("Closed AVI writer")
                del self.VideoWriter
            else:
                # is npy, save what remains of buffer then clear it
                self.np_save_frames()
                self.frame_count = 0

        except Exception as err:
            # The program tries to close a
            print(f"No video to close {err}")
    # TODO add so that background is removed from videos and photos also when
    # saving! Would probably need a separate queue for the BGs to do this well.

    def np_save_frames(self):
        """
        Saves all the frames in the buffer to a .npy file.
        """
        nbr_frames = self.frame_count % self.frame_buffer_size
        if nbr_frames == 0 and self.frame_count != 0:
            nbr_frames = self.frame_buffer_size
        lower_lim = str(max(self.frame_count-nbr_frames, 0))
        upper_lim = str(self.frame_count-1)

        filename = lower_lim + '-' + upper_lim + '.npy'
        with open(self.np_save_path+filename, 'wb') as f:
            np.save(f, self.frame_buffer[:nbr_frames])
        self.frame_buffer *= 0

    def create_NP_writer(self, video_name):
        """
        Creates a folder and saves path to it for saving the numpy images
        """
        if video_name is None:
            video_name = get_video_name(c_p=self.c_p)
        self.np_save_path = self.c_p['recording_path'] + '/' + video_name + '/'
        try:
            os.mkdir(self.np_save_path)
        except Exception as ex:
            print(f"Directory already exist, {ex}")

    def write_to_NPY(self):
        nbr_frames = self.frame_count % self.frame_buffer_size
        if nbr_frames == 0 and self.frame_count > 0:
            # Save the frames into target folder and with suitable name
            self.np_save_frames()
        try:
            # TODO fix error here
            self.frame_buffer[nbr_frames, :, :] = deepcopy(self.frame)
            # Do we miss a frame here?
            self.frame_count += 1
        except Exception as ex:
            print(f"Trouble writing frame, {ex}")
            self.close_video()
            self.frame_count = 0

    def write_frame(self):
        """
        Writes a frame to the current video_writer.
        If the format is "npy" we instead put it in our np-array of images.
            If the np-array of images reaches a special threshold then it is
            automatically saved.
            # Reasonable threshold perhaps 100_000 frames?
        """
        # TODO add bg removal as parameter for each saved frame
        if self.format == 'mp4':
            self.VideoWriter.writeFrame(self.frame)
        elif self.format == 'avi':
            self.VideoWriter.write(self.frame)
        else:
            self.write_to_NPY()

        # Let the caller know that a frame was successfully added to the output
        return True

    def create_video_writer(self, video_name):
        """
        Creates a video writer for the current video.
        """
        # Adjust the video shape to match the images
        image_shape = np.shape(self.frame)
        self.video_width = int(image_shape[0])
        self.video_height = int(image_shape[1])
        if self.format == 'mp4':
            self.VideoWriter = create_mp4_video_writer(c_p=self.c_p,
                                                       video_name=video_name)

        elif self.format == 'avi':
            self.VideoWriter = create_avi_video_writer(self.c_p,
                                                       video_name,
                                                       self.video_width,
                                                       self.video_height)
            print("created avi writer")

        elif self.format == 'npy':
            # calculate an appropriate buffer size based on the size in memory
            # the frames take up.
            self.frame_buffer_size = int(501760000 /(self.video_width *self.video_height))
            if len(image_shape) < 3: # image_shape[2] == 1:
                # TODO uing uint8 may not be the best choice
                # Check if we can use higher bitrates too
                self.frame_buffer = np.uint8(np.zeros([self.frame_buffer_size,
                                                       self.video_width,
                                                       self.video_height,
                                                       ]))
            elif image_shape[2] == 3:
                # TODO check why  height and widht have changed order
                self.frame_buffer = np.uint8(np.zeros([self.frame_buffer_size,
                                                       self.video_width,
                                                       self.video_height, 3]))
            self.frame_count = 0

            self.create_NP_writer(video_name)
        else:
            raise VideoFormatError(f"Video format{self.c_p['video_format']}\
                                   not recognized!")

        self.video_created = True

    def write_video_info(self, filename):
        filepath = self.c_p['recording_path'] + '/' + filename+'_info.txt'
        with open(filepath, 'w') as f:
            f.write("Video framerate:"+str(self.c_p['fps'])+ "fps\n")
            f.write("Exposure time "+str(self.c_p['exposure_time']) +"microseconds (or milliseconds depending on model of camera)\n")
            f.write("Video format: "+str(self.c_p['video_format'])+"\n")
            f.write("Video resolution: "+str(self.video_width)+"x"+str(self.video_height) +"\n")

    def run(self):
        self.c_p['video_idx'] = 0
        while self.c_p['program_running']:
            sleep(self.sleep_time)
            self.c_p['saving_video'] = False

            while self.c_p['recording'] or not self.c_p['frame_queue'].empty()\
                    and self.c_p['program_running']:
                self.c_p['saving_video'] = True

                # Check empty twice since we don't want to wait longer than
                # necessary for the image to be printed to the videowriter

                if not self.c_p['frame_queue'].empty():

                    [self.frame, source_video, self.format] = self.c_p['frame_queue'].get()

                    # Check that name and size are correct, if not create a new
                    image_shape = np.shape(self.frame)
                    if image_shape[0] != self.video_width or\
                            image_shape[1] != self.video_height:
                        self.video_width = image_shape[0]
                        self.video_height = image_shape[1]
                        self.close_video()
                    # Check if name and format is ok
                    # TODO check how this handles leftover frames in buffer?
                    # Maybe change to comparing against source video?
                    if self.video_name != source_video:
                        self.close_video()
                        self.video_name = source_video
                    if not self.last_frame_format == self.format:
                        self.close_video()

                    if not self.video_created:
                        # TODO check naming convention, what happens when we
                        # change format
                        size = '_' + str(self.video_width) + 'x'
                        size += str(self.video_height)
                        #size += '_' + str(self.c_p['video_idx']) 
                        #self.c_p['video_idx'] += 1
                        self.write_video_info(self.video_name+size)
                        self.create_video_writer(self.video_name+size) # Changed so that there is only one index.
                    self.last_frame_format = self.format
                    self.write_frame()
                else:
                    # Queue empty
                    sleep(0.001)
            if self.video_created:
                self.close_video()


class CameraControlMenu():

    def __init__(self, root, c_p, font):
        # TODO add font parameter which sets scale of text
        self.c_p = c_p
        self.font = font
        self.window = tkinter.Toplevel(root)
        self.window.title("Camera window")
        self.window.columnconfigure(0, weight=3)
        self.window.columnconfigure(1, weight=3)
        self.window.columnconfigure(2, weight=3)
        self.camera_info = ""
        self.add_buttons()
        self.info_label = tkinter.Label(self.window,
                                        text=self.get_camera_info(),
                                        font=self.font)
        self.info_label.grid(row=5, column=1)
        self.update()

    def add_buttons(self):
        # TODO add so that info shows up when hovering above a button
        self.record_button = tkinter.Button(master=self.window,
                                            text="Start recording",
                                            command=self.toggle_recording,
                                            bg="green",
                                            font=self.font)

        self.snapshot_button = tkinter.Button(master=self.window,
                                              text="Snapshot",
                                              command=self.snapshot,
                                              font=self.font)

        self.zoom_out_button = tkinter.Button(master=self.window,
                                              text="zoom out",
                                              command=self.zoom_out,
                                              font=self.font)
        self.exposure_time_button = tkinter.Button(master=self.window,
                                                   text="Set exposure time",
                                                   command=self.set_exposure_time,
                                                   font=self.font)
        self.zoom_out_button.grid(row=0, column=0)
        self.record_button.grid(row=0, column=1)
        self.snapshot_button.grid(row=0, column=2)
        self.exposure_time_button.grid(row=0, column=3)

        self.video_format_selection()
        self.image_format_selection()

    def select_video_format(self):
        self.c_p['video_format'] = self.video_format.get()

    def zoom_out(self):
        self.c_p['AOI'] = [0, self.c_p['camera_width'], 0,
                           self.c_p['camera_height']]
        self.c_p['new_settings_camera'] = [True, 'AOI']

    def video_format_selection(self):
        self.video_frame = tkinter.Frame(self.window)
        self.video_format = tkinter.StringVar()
        self.video_format.set(self.c_p['video_format'])
        self.rb_avi = tkinter.Radiobutton(self.window, text="avi", variable=self.video_format,
            value="avi", command=self.select_video_format, font=self.font)
        self.rb_mp4 = tkinter.Radiobutton(self.window, text="mp4", variable=self.video_format,
            value="mp4", command=self.select_video_format, font=self.font)
        self.rb_npy = tkinter.Radiobutton(self.window, text="npy", variable=self.video_format,
            value="npy", command=self.select_video_format, font=self.font)

        self.recording_label = tkinter.Label(self.window,
            text="Recording format", font=self.font)

        self.recording_label.grid(row=1, column=1)
        self.rb_avi.grid(row=2, column=0)
        self.rb_mp4.grid(row=2, column=1)
        self.rb_npy.grid(row=2, column=2)

    def select_image_format(self):
        self.c_p['image_format'] = self.image_format_var.get()

    def set_exposure_time(self):
        """
        Uses a simpledialog to askt the user for a new exposuretime. Then
        transmits this to the camera.

        Returns
        -------
        None.

        """
        user_input = simpledialog.askstring(
            f"Current exposure time {self.c_p['exposure_time']} microseconds",
            " Set exposure time in microseconds: ")

        # Check that the input can be converted to int
        if user_input is None:
            return
        try:
            exposure_time = int(user_input)
        except ValueError:
            print('Cannot convert entry to integer')
            return
        self.c_p['exposure_time'] = exposure_time
        self.c_p['new_settings_camera'] = [True, 'exposure_time']

    def image_format_selection(self, row=3):
        self.image_format_var = tkinter.StringVar()
        self.image_format_var.set("png")
        self.rb_png = tkinter.Radiobutton(self.window, text="png",
                                          variable=self.image_format_var,
                                          value="png",
                                          command=self.select_image_format,
                                          font=self.font)
        self.rb_jpg = tkinter.Radiobutton(self.window, text="jpg",
                                          variable=self.image_format_var,
                                          value="jpg",
                                          command=self.select_image_format,
                                          font=self.font)

        self.image_label = tkinter.Label(self.window, text="Image format",
                                         font=self.font)
        self.image_label.grid(row=row, column=1)
        self.rb_png.grid(row=row+1, column=0)
        self.rb_jpg.grid(row=row+1, column=1)

    def toggle_recording(self):

        if self.c_p['recording'] is False:
            self.c_p['video_name'] = "video"
            self.c_p['video_name'] += str(strftime("%d-%m-%Y-%H-%M-%S"))
            sleep(0.1)
            self.c_p['recording'] = True
        else:
            self.c_p['recording'] = False

    def config_recording_button(self):
        if self.c_p['recording']:
            self.record_button.config(text="Stop recording", bg="red")
        else:
            self.record_button.config(text="Start recording", bg="green")

    def get_camera_info(self):
        w = self.c_p['camera_width']
        h = self.c_p['camera_height']
        self.camera_info = f"Sensor size {w}x{h}\n"
        self.camera_info += f"Exposure time {self.c_p['exposure_time']} \n"
        fps = np.round(self.c_p['fps'], 3)
        self.camera_info += f"Frame rate: {fps} fps\n"
        self.camera_info += f"Image format: {self.c_p['image_format']} \n"
        self.camera_info += f"Frames left to save: {self.c_p['frame_queue'].qsize()}"
        return self.camera_info

    def get_default_filename(self):
        filename = str(strftime("%d-%m-%Y-%H-%M-%S")) + "."
        return filename

    def snapshot(self):
        filename = self.c_p['recording_path'] + self.get_default_filename() +\
            self.c_p['image_format']
        cv2.imwrite(filename, cv2.cvtColor(self.c_p['image'],
                                           cv2.COLOR_RGB2BGR))
        np.save(filename[:-4], self.c_p['image'])

    def update(self):
        self.info_label.config(text=self.get_camera_info())
        self.config_recording_button()
        self.window.after(500, self.update)
