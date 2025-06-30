import torch
import torch.nn as nn

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from threading import Thread
from time import sleep
from PIL import Image
from PyQt6.QtGui import  QColor,QPen
from PyQt6.QtWidgets import (
    QMainWindow, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QToolBar,
    QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

sys.path.append('C:/Users/Martin/OneDrive/PhD/AutOT/') 
import find_particle_threshold as fpt
from unet_model import UNet, UNetZ_split
from CustomMouseTools import MouseInterface

def load_yolov5_model(model_path):
    model = torch.hub.load('.', 'custom', path=model_path, source='local') 
    return model

class ParticleCNN(nn.Module):
    def __init__(self):
        super(ParticleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Assuming grayscale images
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Size now 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Size now 32x32
        )
        # Calculate the size after convolutions and pooling
        # For 128x128 input, after two pooling layers, the size is 32x32
        # And if you have 64 output channels from the last conv layer, then:
        self.size_after_convs = 64 * 32 * 32
        self.fc_layers = nn.Sequential(
            nn.Linear(self.size_after_convs, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Predicting a single value
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # Flatten the output for the fully connected layer
        x = self.fc_layers(x)
        return x


def torch_unet_prediction(model, image, device, fac=1.4, threshold=260):
    """
    Makes a particle tracking prediction using a U-NET. Is not used anymore but
    can be used for more accurate real-time tracking of particles than what the 
    YOLO achieves.
    """
    
    new_size = [int(np.shape(image)[1]/fac),int(np.shape(image)[0]/fac)]
    rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    s = np.shape(rescaled_image)
    rescaled_image = rescaled_image[:s[0]-s[0]%32, :s[1]-s[1]%32]

    if np.shape(rescaled_image)[0] < 100 or np.shape(rescaled_image)[1] <100:
        return np.array([])
    rescaled_image = np.float32(np.reshape(rescaled_image,[1,1,np.shape(rescaled_image)[0],np.shape(rescaled_image)[1]]))
    rescaled_image /= np.std(rescaled_image)
    torch.cuda.empty_cache()
    with torch.no_grad():
        predicted_image = model(torch.tensor(rescaled_image).to(device))
        resulting_image = predicted_image.detach().cpu().numpy()
    
    x,y,_ = fpt.find_particle_centers_fast(np.array(resulting_image[0,0,:,:]), threshold)
    ret = []
    for x_,y_ in zip(x,y):
        ret.append([x_*fac, y_*fac])
    return np.array(ret)

def find_pipette_in_prediction(prediction, z_prediction, threshold=240, size_threshold=10_000):
    """
    Alternative prediction for the pipette which does not rely on YOLO.
    """
    thresholded_image = np.zeros_like(prediction, dtype=np.uint8)
    thresholded_image[prediction > threshold] = 255
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image.astype(np.uint8), connectivity=8)
    for i in range(1, num_labels):  # Skipping the first label as it's the background
        area = stats[i, cv2.CC_STAT_AREA]
        y_bottom = np.max(np.where(labels == i)[0]) # Used to check that the pipette is coming from the bottom.

        if area > size_threshold and y_bottom>np.shape(prediction)[0]-10:
            # Find min of y-axis to identify the top
            y_max = np.min(np.where(labels == i)[0])
            z = np.median(z_prediction[labels == i])
            if y_max+10 > np.shape(labels)[0]:
                return centroids[i][0], y_max, z
            # Find x as the center of the pipette top
            x_mean = np.mean(np.where(labels[y_max:y_max+10,:] == i)[1])
            return x_mean, y_max, z

    return None, None, None

def extract_particles_3dpos(prediction, threshold=250):
    
    tmp = prediction[2,:,:]-np.min(prediction[2,:,:])
    tmp = tmp/np.max(tmp)*255
    tmp = np.uint8(tmp)
    tmp[prediction[0,:,:]<threshold] = 0
    
    circles = cv2.HoughCircles(tmp, cv2.HOUGH_GRADIENT, dp=1, minDist=14, # np.array(prediction[0,:,:],dtype=np.uint8)
                            param1=60, param2=14, minRadius=6, maxRadius=0)

    x,y,z = [],[],[]

    if circles is None:
        return np.array(x),np.array(y),np.array(z)

    for circle in circles[0]:
        r = int(circle[2])
        particle_intensity = np.sum(prediction[0,:,:][int(circle[1])-r:int(circle[1])+r,int(circle[0])-r:int(circle[0])+r])
        if particle_intensity < r**2*threshold*4:
            print("Particle intensity too low")
            continue
        x.append(circle[0])
        y.append(circle[1])
        # Extract z value as the average in the circle
        try:
            z.append(np.median(prediction[2,:,:][int(circle[1])-r:int(circle[1])+r,int(circle[0])-r:int(circle[0])+r]))
        except:
            # If we did not get the whole circle then take the center as the z value
            z.append(prediction[2,:,:][int(circle[1]),int(circle[0])])
    return np.array(x),np.array(y),np.array(z)

def torch_unet_z_prediction(model, image, device, fac=2, num_channels_to_modify=2, sigmoid=True):
    new_size = [int(np.shape(image)[1]/fac),int(np.shape(image)[0]/fac)]
    rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    s = np.shape(rescaled_image)
    rescaled_image = rescaled_image[:s[0]-s[0]%32,:s[1]-s[1]%32]
    rescaled_image = np.float32(np.reshape(rescaled_image,[1,1,np.shape(rescaled_image)[0],np.shape(rescaled_image)[1]]))
    with torch.no_grad():
        predicted_image = model(torch.tensor(rescaled_image).to(device))
        sigmoid_channels = predicted_image[:, :num_channels_to_modify, :, :]  # Select the first N channels
        non_sigmoid_channels = predicted_image[:, num_channels_to_modify:, :, :]  # Select the remaining channels
        if sigmoid:
            # Apply sigmoid to the selected channels
            sigmoid_applied = torch.sigmoid(sigmoid_channels)
            sigmoid_applied *= 255
            # Concatenate the modified and unmodified channels back together
            predicted_image = torch.cat([sigmoid_applied, non_sigmoid_channels], dim=1)
        resulting_image = predicted_image.detach().cpu().numpy()
    
    return np.array(resulting_image[0,:,:,:])

def load_torch_unet(model_path, nbr_of_output_channels=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU {torch.cuda.is_available()}\nDevice name: {torch.cuda.get_device_name(0)}")

    try:
        model = UNetZ_split(
            input_shape=(1, 1, 256, 256),
            number_of_output_channels_detect=2,
            number_of_output_channels_z=2,
            conv_layer_dimensions=(64, 128, 256, 512, 1024),
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model, device
    except Exception as e:
        print(e)
        print("Could not load z model")
    try:
        model = UNet(
            input_shape=(1, 1, 256, 256),
            number_of_output_channels=nbr_of_output_channels,  # 2 for binary segmentation and 3 for multiclass segmentation
            conv_layer_dimensions=(8, 16, 32, 64, 128, 256),  # smaller UNet (faster execution)
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model, device
    except Exception as e:
        print(e)
        print("Could not load small model")
    try:
        model = UNet(
            input_shape=(1, 1, 256, 256),
            number_of_output_channels=nbr_of_output_channels,  # 2 for binary segmentation and 3 for multiclass segmentation
            conv_layer_dimensions=(64, 128, 256, 512, 1024),  # standard UNet
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model, device    
    except Exception as e:
        print(e)
        print("Could not load big model")


    return None, None

class DeepLearningAnalyserLDS(Thread):
    """
    Thread which analyses the real-time image for detecting particles.
    """
    def __init__(self, c_p, data_channels, particle_type=0, model=None):
        """
        

        Parameters
        ----------
        c_p : TYPE
            DESCRIPTION.
        model : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        Thread.__init__(self)
        self.c_p = c_p
        self.data_channels = data_channels
        self.c_p['model'] = model
        self.training_target_size = (64, 64)
        self.confidence_threshold_particle = 0.55 # Deteciton limit for the YOLO model
        self.confidence_threshold_pipette = 0.5
        self.particle_type = particle_type # Type of particle to be tracked/analyzed
        self.particle_size_limit = 1.5/self.c_p['microns_per_pix'] # Smaller than 1.5 microns ish is too small to be relevant

        # Load the default networks for tracking in xy and z.
        if self.c_p['yolo_path'] is not None:
            self.c_p['model'] = load_yolov5_model(self.c_p['yolo_path'])
            self.c_p['network'] = "YOLOv5"
            self.c_p['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Loaded default network from {self.c_p['yolo_path']}")
        if self.c_p['default_z_model_path'] is not None:
            self.c_p['z-model'] = torch.load(self.c_p['default_z_model_path'])
            print(f"Loaded default z-model from {self.c_p['default_z_model_path']}")
        self.setDaemon(True)

    def yolo_prediction(self):
        """
        Runs YOLO model inference on the input image to detect particles and pipette, 
        and updates internal state with the detection results.
        The function processes the model's predictions, filters them based on confidence 
        thresholds and size limits, and distinguishes between particles and pipette detections. 
        Detected particle positions and radii are stored, and the pipette location is updated 
        if found.
        Updates the following keys in self.c_p:
            - 'pipette_located': Boolean indicating if the pipette was detected.
            - 'predicted_particle_positions': Numpy array of detected particle positions.
            - 'predicted_particle_radii': Numpy array of detected particle radii.
            - 'pipette_location': List containing pipette bounding box information if detected.
        Returns:
            None
        """

        results = self.c_p['model'](self.c_p['image']).xywh[0].cpu().numpy()
        particle_positions = []
        radii = []
        pipette_located = False
        for prediction in results:
            # Check if particle
            if prediction [-1] == 0 and prediction[4] > self.confidence_threshold_particle:
                if prediction[2] > self.particle_size_limit and prediction[3] > self.particle_size_limit:
                    particle_positions.append(prediction[:2])
                    radii.append((prediction[2]+prediction[3])/4)

            # Check if pipette
            elif prediction[-1] == 1 and prediction[4] > self.confidence_threshold_pipette:
                pipette_located = True
                self.c_p['pipette_location'][0] = prediction[0]
                self.c_p['pipette_location'][1] = prediction[1] - prediction[3]/2
                self.c_p['pipette_location'][2] = prediction[2]
                self.c_p['pipette_location'][3] = prediction[3]

        self.c_p['pipette_located'] = pipette_located
        self.c_p['predicted_particle_positions'] = np.array(particle_positions)
        self.c_p['predicted_particle_radii'] = np.array(radii)

    def find_closest_particle(self, reference_position, return_idx):
        """
        Checks for the particle closest to the optical trap
        """
        try:
            LX = reference_position[0]
            LY = reference_position[1]
            min_x = 1000
            min_dist = 2e10
            min_y = 1000
            idx = 0
            min_idx = 0
            for x,y in self.c_p['predicted_particle_positions']:

                if (x-LX)**2+(y-LY)**2<min_dist:
                    min_dist = (x-LX)**2+(y-LY)**2
                    min_x = x
                    min_y = y
                    min_idx = idx
                idx += 1
            if return_idx:
                return [min_x, min_y], min_idx
            return [min_x, min_y]
        except Exception as e:
            return None

    def check_in_pipette(self, threshold=2_000, offset=np.array([0, 50])): # threhsold distance used to be 20_000
        if not self.c_p['pipette_located'] or not self.c_p['tracking_on'] or self.c_p['pipette_location'] is None:
            return
        try:
            potential_pos, idx = self.find_closest_particle(np.array(self.c_p['pipette_location'][0:2]) - offset, True) # Take into account that particle is above the pipette
        except TypeError as te:
            print("Error in finding closest particle to pipette", te)
            return
        if potential_pos is None:
            return
        
        # Use particle redii to determine position, check that it has not been updated erroneously.
        
        radii = 1.5/self.c_p['microns_per_pix'] # radii estimate
        if (self.c_p['pipette_location'][0] - potential_pos[0])**2/3 + ((self.c_p['pipette_location'][1]-radii) - potential_pos[1])**2 < threshold:
            self.c_p['particle_in_pipette'] = True
            self.c_p['pipette_particle_location'][0:2] = potential_pos
            self.c_p['pipette_particle_location'][3] = self.c_p['predicted_particle_radii'][idx]
            try:
                if self.c_p['z-tracking']:
                    self.c_p['pipette_particle_location'][2] = self.c_p['z-predictions'][idx]
                else:
                    self.c_p['pipette_particle_location'][2] = None
            except IndexError as ie:
                self.c_p['pipette_particle_location'][2] = None

        else:
            self.c_p['particle_in_pipette'] = False

    def check_trapped(self, threshold=10_000):
        """
        Checks if a particle is trapped within a specified threshold distance from the laser position.
        This method updates the internal state to reflect whether any predicted particle is within the
        threshold distance from the laser position (considered "trapped"). It also updates the trapped
        particle's position (x, y, and optionally z) and radius in the internal data structure.
        Parameters:
            threshold (float, optional): The maximum squared distance from the laser position for a particle
                to be considered trapped. Defaults to 10,000.
        Returns:
            bool: True if a particle is in view and within the threshold distance (trapped), False otherwise.
        Notes:
            - If no predicted particle positions are available, the function returns False and sets
              `particles_in_view` to False.
            - The function attempts to update the z-position of the trapped particle if z-tracking is enabled.
              If the z-prediction is unavailable, the z-position is set to None.
        """
        
        if len(self.c_p['predicted_particle_positions']) < 1:
            self.particles_in_view = False
            return False
        self.particles_in_view = True

        LX = self.c_p['laser_position'][0] - self.c_p['AOI'][0]
        LY = self.c_p['laser_position'][1] - self.c_p['AOI'][2]
        distances = [(x-LX)**2+(y-LY)**2 for x,y in self.c_p['predicted_particle_positions']]
        self.c_p['Trapped_particle_position'][0:2], idx = self.find_closest_particle([LX, LY],True)
        self.c_p['Trapped_particle_position'][3] = self.c_p['predicted_particle_radii'][idx]
        # Check if we can get also the z-position, different units tough. Set to None if no z-position found.
        if self.c_p['z-tracking']:
            try:
                self.c_p['Trapped_particle_position'][2] = self.c_p['z-predictions'][idx]
            except IndexError as ie:
                self.c_p['Trapped_particle_position'][2] = None
        else:
            self.c_p['Trapped_particle_position'][2] = None
        self.c_p['particle_trapped'] = min(distances) < threshold

    def predict_z_positions(self):
        """
        Function which makes a prediction of the z-positions of the particles located with
        the deep learning model. The z-positions are then stored in the control parameters.
        """

        if not self.c_p['tracking_on'] or self.c_p['z-model'] is None:
            return

        # Pre-compute constants
        image_shape = np.shape(self.c_p['image'])
        try:
            image_width, image_height = image_shape[1], image_shape[0]
        except:
            self.c_p['z-predictions'] = []
            return 
        device = self.c_p['device']
        width = self.c_p['crop_width']

        # List to collect crops
        crops = []

        # Loop through predicted positions to collect crops
        for pos in self.c_p['predicted_particle_positions']:
            x, y = int(pos[0]), int(pos[1])

            # Check if the crop is within the image
            if 0 <= x - width < x + width <= image_width and 0 <= y - width < y + width <= image_height:
                try:
                    crop = self.c_p['image'][y - width:y + width, x - width:x + width].astype(np.float32)
                    crop /= 2 * np.std(crop)
                    crop -= np.mean(crop)
                    # crop /= 20 # Changed here
                    crop = np.reshape(crop, (128, 128, 1))
                    crops.append(crop)
                except ValueError as e:
                    print(e) # Most likely the shape of the image changed during prediction, no worries.
                    pass

        # Convert list of crops to a tensor and prepare for the model
        if crops:  # Check if there are any crops to process
            crops_tensor = torch.tensor(crops, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():  # No gradients needed for inference
                predictions = self.c_p['z-model'](crops_tensor)
            z_vals = predictions.squeeze().tolist()  # Convert predictions to a list
        else:
            z_vals = []
    
        if isinstance(z_vals, float):
            z_vals = np.array([z_vals])
        self.c_p['z-predictions'] = z_vals

    def run(self):
        
        while self.c_p['program_running']:
            # By default check a central square of the frame. Maybe even have a ROI for this thread
            if self.c_p['model'] is not None and self.c_p['tracking_on']:
                # Changed here to just use YOLO since it is the best.
                self.yolo_prediction()
                if self.c_p['z-tracking']:
                    self.predict_z_positions()
                self.check_trapped()
                self.check_in_pipette()
            else:
                sleep(0.1)
            if self.c_p['train_new_model']:
                print("training new model")
                self.train_new_model(self.c_p['training_image'])
                self.c_p['train_new_model'] = False
            
class PlotParticleProfileWidget(QMainWindow):
    """
    Helps plot the two particle profiles in real time to compare the z-positions.
    Will need to update this to make sure that it works properly.
    
    """

    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p   # Control parameters
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.setWindowTitle('Particle profiles')

        self.image_idx = 0
        self.filename = "TrappedAndPipetteimage_"

        trappped_center = (200,204) # Placeholder
        pipette_center = (204,323)

        self.centers = [trappped_center, pipette_center]
        self.particle_width = 60

        self.x = list(range(100))  # 100 time points
        self.y = [np.random.normal() for _ in range(100)]  # 100 data points
        self.y2 = [np.random.normal() for _ in range(100)]

        self.graphWidget.setBackground('w')
        self.pen1 = pg.mkPen(color=(255, 0, 0))
        self.data_line1 = self.graphWidget.plot(self.x, self.y, pen=self.pen1)

        self.pen2 = pg.mkPen(color=(0, 255, 0))
        self.data_line2 = self.graphWidget.plot(self.x, self.y2, pen=self.pen2)
        
        self.timer = QTimer()
        self.timer.setInterval(200)  # Update interval in milliseconds
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

    def update_plot_data(self):
        """
        Updates the prediction data used for plotting.
        """
        if len(self.c_p['predicted_particle_positions']) < 2:
            return

        self.x = np.linspace(-self.particle_width, self.particle_width, 2*self.particle_width)
        
        for idx, pos in enumerate(self.c_p['predicted_particle_positions']):

            if idx>=2:
                break
            if idx==0:
                center_1 = [int(pos[0]), int(pos[1])]
            else:
                center_2 = [int(pos[0]), int(pos[1])]

        self.y = self.c_p['image'][center_1[1],
                            center_1[0]-self.particle_width:
                            center_1[0]+self.particle_width]
    
        self.data_line1.setData(self.x, self.y)  # Update the data.
        self.y2 = self.c_p['image'][center_2[1],
                            center_2[0]-self.particle_width:
                            center_2[0]+self.particle_width]
        self.data_line2.setData(self.x, self.y2)

class DeepLearningControlWidget(QWidget):
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        layout = QVBoxLayout()
        
        self.setWindowTitle("Deep learning controller")

        self.toggle_tracking_button = QPushButton('Tracking on')
        self.toggle_tracking_button.pressed.connect(self.toggle_tracking)
        self.toggle_tracking_button.setCheckable(True)
        self.toggle_tracking_button.setChecked(self.c_p['tracking_on'])
        layout.addWidget(self.toggle_tracking_button)

        self.locate_pipette_button = QPushButton('Locate pipette')
        self.locate_pipette_button.pressed.connect(self.locate_pipette)
        self.locate_pipette_button.setCheckable(True)
        self.locate_pipette_button.setChecked(self.c_p['locate_pipette'])
        self.locate_pipette_button.setToolTip("Locate the pipette tip in the image")
        layout.addWidget(self.locate_pipette_button)

        self.toggle_z_tracking_button = QPushButton('Z-tracking on')
        self.toggle_z_tracking_button.pressed.connect(self.toggle_z_tracking)
        self.toggle_z_tracking_button.setCheckable(True)
        self.toggle_z_tracking_button.setChecked(self.c_p['z-tracking'])
        layout.addWidget(self.toggle_z_tracking_button)

        self.set_Z_zero_button = QPushButton('Set Z zero')
        self.set_Z_zero_button.pressed.connect(self.set_Z_zero)
        self.set_Z_zero_button.setCheckable(False)
        layout.addWidget(self.set_Z_zero_button)

        self.load_z_model_button = QPushButton('Load z-model')
        self.load_z_model_button.pressed.connect(self.load_z_model)
        self.load_z_model_button.setCheckable(False)
        layout.addWidget(self.load_z_model_button)

        self.load_YOLO_model_button = QPushButton('Load YOLO model')
        self.load_YOLO_model_button.pressed.connect(self.load_YOLO_model)
        self.load_YOLO_model_button.setCheckable(False)
        layout.addWidget(self.load_YOLO_model_button)
                
        self.setLayout(layout)

    def set_threshold(self, threshold):
        self.c_p['cutoff'] = threshold

    def save_network(self):
        # Not finished
        filename = QFileDialog.getSaveFileName(self, 'Save network',
            self.c_p['recording_path'],"Network (*.h5)")
        print(f"Filename for saving {filename} .")
    
    def load_network(self):
        filename = QFileDialog.get(self, 'Load network', self.c_p['recording_path'])
        print(f"You want to open network {filename}")
        backend = tf.keras.models.load_model(filename) 
        self.c_p['model'] = dt.models.LodeSTAR(backend.model) 
        self.c_p['prescale_factor'] = 0.106667

    def load_deeptrack_unet(self):
        network_name = QFileDialog.getExistingDirectory(self, 'Load network', self.c_p['recording_path'])
        custom_objects = {"unet_crossentropy": dt.losses.weighted_crossentropy((10, 1))}
        with tf.keras.utils.custom_object_scope(custom_objects):
            #try:
            self.c_p['model'] = tf.keras.models.load_model(network_name)
            self.c_p['network'] = "DeepTrack Unet"
            
    def load_pytorch_unet(self):

        network_name = QFileDialog.getOpenFileName(self, 'Load network')
        print(f"Opening network {network_name[0]}")
        self.c_p['model'], self.c_p['device']  = load_torch_unet(network_name[0])
        self.c_p['network'] = "Pytorch Unet"

    def load_z_model(self):
        network_name = QFileDialog.getOpenFileName(self, 'Load network')
        print(f"Opening network {network_name[0]}")
        try:
            self.c_p['z-model'] = torch.load(network_name[0])
        except Exception as e:
            print(e)
            print("Could not load model")

    def locate_pipette(self):
        self.c_p['locate_pipette'] = not self.c_p['locate_pipette']

    def toggle_tracking(self):
        self.c_p['tracking_on'] = not self.c_p['tracking_on']

    def toggle_z_tracking(self):
        self.c_p['z-tracking'] = not self.c_p['z-tracking']

    def set_tracking_prescale_factor(self, scale):
        self.c_p['prescale_factor'] = scale

    def openPlotWindow(self):
        self.plotWindow = PlotParticleProfileWidget(self.c_p)
        self.plotWindow.show()
 
    def train_network(self):
        self.c_p['train_new_model'] = True

    def set_Z_zero(self):
        if self.c_p['particle_trapped']:
            self.c_p['z-offset'] += self.c_p['Trapped_particle_position'][2]

    def load_YOLO_model(self):
        
        network_name = QFileDialog.getOpenFileName(self, 'Load network')
        print(f"Opening network {network_name[0]}")
        try:
            model = load_yolov5_model(network_name[0])
            self.c_p['model'] = model
        except Exception as e:
            print(e)
            print("Could not load model")

    def show_training_image(self):
        plt.imshow(self.c_p['training_image'])
        plt.show()

class MouseAreaSelect(MouseInterface):
    
    def __init__(self, c_p):
        self.c_p = c_p
        self.x_0 = 0
        self.y_0 = 0
        self.x_0_motor = 0
        self.y_0_motor = 0
        self.image = np.zeros([64,64,3])
        
        self.blue_pen = QPen()
        self.blue_pen.setColor(QColor('blue'))
        self.blue_pen.setWidth(2)


    def draw(self, qp):
        if self.c_p['mouse_params'][0] == 1:
            qp.setPen(self.blue_pen)                
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
        im = self.c_p['image']
        if len(np.shape(im)) > 2:
            image = im[down:up,left:right,:]
        else:
            image = im[down:up, left:right]
        plt.imshow(image)
        plt.show()

        if up-down > right-left:
            width = right-left
        else:
            width = up-down
        crop = im[down:down+width, left:left+width]
        self.c_p['prescale_factor'] = 64 / width
        print(self.c_p['prescale_factor'])
        res = cv2.resize(crop, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
        plt.imshow(res)
        plt.show()
        self.c_p['training_image'] = np.reshape(res,[32,32,1])

    def mouseDoubleClick(self):
        pass
    
    def mouseMove(self):
        if self.c_p['mouse_params'][0] == 2:
            pass
    
    def getToolName(self):
        return "Area select tool"

    def getToolTip(self):
        return "Use the mouse to select an area to train network on by dragging."
