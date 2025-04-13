import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
import scipy
import math
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import os

import find_particle_threshold as fpt
from unet_model import UNet
from CameraControlsNew import npy_generator
from time import time
import AnalysisHelperFunctions as AHF



def find_particle_centers_fast(image, threshold=120, particle_size_threshold=200,
                               particle_upper_size_threshold=5000, bright_particle=True,
                               fill_holes=False, check_circular=True, circularity_threshold=(0.9, 1.0)):
    """
    Function which locates particle centers using OpenCV's connectedComponentsWithStats.
    
    Parameters:
        image: Image with the particles.
        threshold: Threshold value of the particle.
        particle_size_threshold: Minimum area of particle in image measured in pixels.
        particle_upper_size_threshold: Maximum area of particle in image measured in pixels.
        bright_particle: If the particle is brighter than the background or not.
        fill_holes: If true, fills holes in binary objects.
        check_circular: If true, checks if the object is circular.
        circularity_threshold: Tuple of min and max circularity values for valid particles.
        
    Returns:
        x, y: Arrays with the x and y coordinates of the particle in the image in pixels.
              Returns empty arrays if no particle was found.
        thresholded_image: The binary image after thresholding.
    """
    import cv2
    import numpy as np

    # Blur and threshold the image
    blurred_image = cv2.blur(image, (8, 8))
    ret, thresholded_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY if bright_particle else cv2.THRESH_BINARY_INV)
    
    if fill_holes:
        # Morphological operation to fill holes
        contour, hier = cv2.findContours(thresholded_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(thresholded_image, [cnt], 0, 255, -1)
    
    # Use connectedComponentsWithStats to find particles
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image.astype(np.uint8), connectivity=8)

    x = []
    y = []

    for i in range(1, num_labels):  # Skipping the first label as it's the background
        area = stats[i, cv2.CC_STAT_AREA]
        if particle_size_threshold < area < particle_upper_size_threshold:
            # Extract contour for the current particle
            mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Calculate perimeter and circularity
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:  # Avoid division by zero
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    if not check_circular or (circularity_threshold[0] <= circularity <= circularity_threshold[1]):
                        # Valid particle, get its center
                        cx, cy = centroids[i]
                        x.append(cx)
                        y.append(cy)

    return x, y, thresholded_image

def find_jump(trajectory):
    smoothed = rolling_mean(trajectory,10)
    diff = smoothed[100:] - smoothed[:-100]
    idx = np.argmax(np.abs(diff))
    return idx

def find_closest_x(x, y, target_y):
    """Finds the x-value closest to the target_y crossing point."""
    idx = np.argmin(np.abs(y - target_y))
    return x[idx]

def align_signals(x1, y1, x2, y2, target_y, target_x):
    # Find the x positions where y1 and y2 are closest to target_y
    x1_cross = find_closest_x(x1, y1, target_y)
    x2_cross = find_closest_x(x2, y2, target_y)

    # Calculate the shifts needed to align each signal at target_x
    shift_x1 = target_x - x1_cross
    shift_x2 = target_x - x2_cross

    # Apply shifts to x1 and x2
    x1_aligned = x1 + shift_x1
    x2_aligned = x2 + shift_x2

    return x1_aligned, x2_aligned

def avi_frame_generator(video_path):
    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture object has been successfully initialized
    if not cap.isOpened():
        print("Error: Could not open video file.")
        yield None
        return

    # Loop through the video frames
    while True:
        ret, frame = cap.read()

        if not ret:
            # No more frames to read
            break

        yield frame[:,:,0]

    # Release the video capture object
    cap.release()

    # Indicate end of video stream
    yield None

def find_particle_radii(image,threshold=120, particle_size_threshold=200,
                        particle_upper_size_threshold=5000,
                        bright_particle=True, fill_holes=False, check_circular=False):
    """
    Function which locates particle centers using thresholding.
    Parameters :
        image - Image with the particles
        threshold - Threshold value of the particle
        particle_size_threshold - minimum area of particle in image measured in pixels
        bright_particle - If the particle is brighter than the background or not
    Returns :
        x,y - arrays with the x and y coordinates of the particle in the image in pixels.
            Returns empty arrays if no particle was found
    """

    # Do thresholding of the image
    thresholded_image = cv2.blur(image, (8, 8)) > threshold
    if fill_holes:
        # Fill holes in the image before labeling
        # Something wrong with fill_holes when using dark particle
        thresholded_image = ndi.morphology.binary_fill_holes(thresholded_image)

    # Separate the thresholded image into different sections
    separate_particles_image = measure.label(thresholded_image)
    # Count the number of pixels in each section
    counts = np.bincount(np.reshape(separate_particles_image,(np.shape(separate_particles_image)[0]*np.shape(separate_particles_image)[1])))

    x = []
    y = []
    r = []

    for group, pixel_count in enumerate(counts): # First will be background
        if particle_upper_size_threshold>pixel_count>particle_size_threshold:
            # Particle found, locate center of mass of the particle
            cy, cx = ndi.center_of_mass(separate_particles_image==group) # This is slow
            if check_circular:
                M = measure.moments_central(separate_particles_image==group, order=2)
                if 0.7 < (M[0, 2] / M[2, 0]) < 1.3:
                    x.append(cx)
                    y.append(cy)
                else:
                    print('Noncircular object!', (M[0, 2] / M[2, 0]))
            else:
                x.append(cx)
                y.append(cy)
            r.append(np.sqrt(pixel_count/np.pi))

    return x, y, r, thresholded_image


def get_torch_prediction_array(model, images, device, fac=4, threshold=150):

    # Resize the first image to determine the new size
    image = images[0]
    new_size = [int(np.shape(image)[1] / fac), int(np.shape(image)[0] / fac)]
    rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)

    # Calculate padding to make dimensions a multiple of 32
    height, width = rescaled_image.shape[:2]
    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32

    # Determine padded dimensions
    padded_height = height + pad_h
    padded_width = width + pad_w

    # Create an array to hold all padded images
    rescaled_images = np.zeros([len(images), 1, padded_height, padded_width], dtype=np.float32)

    # Process each image in the array
    for i, image in enumerate(images):
        rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
        padded_image = np.pad(
            rescaled_image,
            ((0, pad_h), (0, pad_w)),
            mode="reflect"
        )
        rescaled_images[i, 0, :, :] = padded_image

    # Perform prediction with the model
    with torch.no_grad():
        predicted_images = model(torch.tensor(rescaled_images).to(device))

    resulting_images = predicted_images.detach().cpu().numpy()

    x = []
    y = []

    # Analyze the predictions
    for i in range(len(images)):
        x_, y_, _ = find_particle_centers_fast( #fpt.find_particle_centers_fast was changed from to get only circular predictions
            np.array(resulting_images[i, 0, :, :]), threshold, particle_size_threshold=600
        )
        if len(x_) > 0:
            x.append(x_)
            y.append(y_)
        else:
            # Retry individual prediction if no particles were found
            x_, y_, _ = get_torch_prediction(model, images[i], device, fac=fac, threshold=threshold)
            if len(x_) > 0:
                x.append(x_)
                y.append(y_)
            else:
                x.append([0])
                y.append([0])

    return x, y


def get_torch_prediction(model, image, device, fac=4, threshold=150):


    new_size = [int(np.shape(image)[1] / fac), int(np.shape(image)[0] / fac)]
    rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)

    # Calculate padding to make dimensions a multiple of 32
    height, width = rescaled_image.shape[:2]
    pad_h = (32 - height % 32) % 32
    pad_w = (32 - width % 32) % 32

    # Pad with flipped content from the opposite sides
    padded_image = np.pad(
        rescaled_image,
        ((0, pad_h), (0, pad_w)),
        mode="reflect"
    )

    # Reshape to match input dimensions expected by the model
    padded_image = np.float32(padded_image).reshape(1, 1, padded_image.shape[0], padded_image.shape[1])

    with torch.no_grad():
        predicted_image = model(torch.tensor(padded_image).to(device))

    resulting_image = predicted_image.detach().cpu().numpy()
    x, y, _ = find_particle_centers_fast(
        np.array(resulting_image[0, 0, :, :]), threshold, particle_size_threshold=600
    )

    return x, y, resulting_image[0, 0, :, :]
"""

def get_torch_prediction_array(model, images,device, fac=4, threshold=150):
    image = images[0]
    new_size = [int(np.shape(image)[1]/fac),int(np.shape(image)[0]/fac)]
    rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)

    s = np.shape(rescaled_image)
    rescaled_image = rescaled_image[:s[0]-s[0]%32, :s[1]-s[1]%32]
    rescaled_images = np.zeros([len(images),1,np.shape(rescaled_image)[0],np.shape(rescaled_image)[1]])
    for i,image in enumerate(images):
        rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
        rescaled_image = rescaled_image[:s[0]-s[0]%32, :s[1]-s[1]%32]
        rescaled_images[i,0,:,:] = rescaled_image
    rescaled_images = np.float32(rescaled_images)
    with torch.no_grad():
        predicted_images = model(torch.tensor(rescaled_images).to(device))
    resulting_images = predicted_images.detach().cpu().numpy()
    x = []
    y = []
    for i in range(len(images)):
        x_,y_,_ = fpt.find_particle_centers_fast(np.array(resulting_images[i,0,:,:]), threshold, particle_size_threshold=600)
        if len (x_)>0:
            x.append(x_)
            y.append(y_)
        else:
            
            x_,y_,_ = get_torch_prediction(model,images[i],device,fac=fac,threshold=threshold)
            if len(x_)>0:
                x.append(x_)
                y.append(y_)
            else:
                x.append([0])
                y.append([0])
    return x, y


def get_torch_prediction(model, image,device, fac=4, threshold=150):

    new_size = [int(np.shape(image)[1]/fac),int(np.shape(image)[0]/fac)]
    rescaled_image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    s = np.shape(rescaled_image)
    rescaled_image = rescaled_image[:s[0]-s[0]%32, :s[1]-s[1]%32]
    # TODO do more of this in pytorch which is faster since it works on GPU
    rescaled_image = np.float32(np.reshape(rescaled_image,[1,1,np.shape(rescaled_image)[0],np.shape(rescaled_image)[1]]))
    with torch.no_grad():
        predicted_image = model(torch.tensor(rescaled_image).to(device))
    resulting_image = predicted_image.detach().cpu().numpy()
    x,y,_ = fpt.find_particle_centers_fast(np.array(resulting_image[0,0,:,:]), threshold, particle_size_threshold=600)
    
    return x, y, resulting_image[0,0,:,:]
"""

def split_image_np(img_np, num_images):
    h, w, _ = img_np.shape
    num_images_side = int(math.sqrt(num_images))  # number of images on one side of the grid

    size = min(h // num_images_side, w // num_images_side)  # size of each sub-image
    return [img_np[i:i+size, j:j+size, :] for i in range(0, h, size) for j in range(0, w, size)]


def rolling_mean(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_prediction(image,model,device, fac=1,threshold=200):
    start = time()
    x,y,_  = get_torch_prediction(model,image,device,fac=fac,threshold=threshold)
    stop = time()
    print(f"Prediction time was {stop-start} s")

    # Display normal image
    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(image, cmap='gray')
    plt.plot(fac*np.array(x),fac*np.array(y),'*r')
    # Display segmentation
    plt.subplot(1,3,2)
    plt.imshow(_)#resulting_image[0,0,:,:]>150)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(_>threshold)#resulting_image[0,0,:,:]>150)
    plt.plot(x,y,'*r')
    plt.title(f"Segmentation no ")
    plt.show()
"""
def track_video(video_path, model,fac,threshold=150,format='npy',frame_limit=-1):
    x_pos = []
    y_pos = []
    r_meas = []
    if format == 'npy':
        generator = npy_generator(video_path)
    elif format =="avi":
        generator = avi_frame_generator(video_path)
    for idx,image in enumerate(generator):
        if image is None:
            break
        if frame_limit>0 and idx>frame_limit:
            break
        if idx%100==0:
            print(idx)
        x,y,ret  = get_torch_prediction(model,image,fac=fac,threshold=threshold)
        xt,yt,r,_ = find_particle_radii(ret,threshold=threshold)

        r_meas.append(fac*np.array(r))
        x_pos.append(fac*np.array(x))
        y_pos.append(fac*np.array(y))
    y_microns = np.transpose(np.array(y_pos))/18.28
    x_microns = np.transpose(np.array(x_pos))/18.28
    r_microns = np.transpose(np.array(r_meas))/18.28
    return x_microns, y_microns, r_microns
"""

def synchronize_data(data_path, dy,window_size=100):
    data = np.load(data_path, allow_pickle=True)
    #dy = y_microns[0,:]-y_microns[1,:]
    cam_start, cam_stop = AHF.get_cam_move_lims(dy)
    cam_start += 5
    cam_stop += -2
    psd_start, psd_stop = AHF.get_limits_PSD(data)
    # TODO add the tilt to the analysis

    X_data, Y_data, X_data_A, Y_data_A, X_data_B, Y_data_B = AHF.prepare_plot_data(data,Window_size=window_size,start=psd_start,stop=psd_stop, shorten=True)
    print(cam_start,cam_stop, len(dy))
    cam_data_pos = -AHF.resample_signal(dy[cam_start:cam_stop],len(X_data))
    return cam_data_pos, Y_data

def check_tracking(path,fac,threshold,model,device,checked_indices=[0,1,2]):
    if path[-3:] == 'avi':
        generator = avi_frame_generator(path)
    else:
        generator = npy_frame_generator(path)
    for idx, image in enumerate(generator):
        if idx in checked_indices:
            print(np.shape(image))
            plot_prediction(image,model,device,fac=fac,threshold=threshold)
        if image is None:
            break
    

def convert_to_absolute_position(data, offset_Y):
    PSD_TO_POS =  [14.252,14.252]#[14.252,12.62]
    AY = data['PSD_A_P_Y']/data['PSD_A_P_sum']*PSD_TO_POS[0]
    BY = data['PSD_B_P_Y']/data['PSD_B_P_sum']*PSD_TO_POS[1]

    AX = data['PSD_A_P_X']/data['PSD_A_P_sum']*PSD_TO_POS[0]
    BX = data['PSD_B_P_X']/data['PSD_B_P_sum']*PSD_TO_POS[1]

    AY -= np.mean(AY[:2000]) - offset_Y
    BY -= np.mean(BY[:2000]) - offset_Y

    return AY,BY, (AY+BY)/2

def calc_Y_force(data):
    PSD_to_force = [0.02505,0.02565,0.02755,0.0287]
    AY = data['PSD_A_F_Y']*PSD_to_force[1]
    BY = data['PSD_B_F_Y']*PSD_to_force[3]
    return -(AY+BY)


def get_laser_pos_force(data_path,video_path,fac=1.4, threshold=30):
    x_cam, y_cam, r = track_video(video_path, model,fac,threshold,format='npy',frame_limit=12)
    data = np.load(data_path, allow_pickle=True)
    offset_Y = np.mean(y_cam[1,:10])-np.mean(y_cam[0,:10])
    AY,BY,mean_Y = convert_to_absolute_position(data, offset_Y)
    force_y = calc_Y_force(data)
    return mean_Y, force_y,x_cam, y_cam,data


def extract_frames(cap, frame_indices):
    extracted_frames = []

    for i in frame_indices:
        # Set the current position of the video to the specified frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        # Check if the frame was successfully read
        if ret:
            extracted_frames.append((i, frame))
        else:
            print(f"Frame {i} could not be read")
    return extracted_frames

def npy_frame_generator(path):
    """
    Used to read all the images in a npy image folder one at a time. Takes the
    full path as input and outputs an image. Outputs None if there are no more
    images to read.
    """

    def sorting_key(file_name):
        """Key function to sort files based on the starting number."""
        start_num = file_name.split('-')[0]
        return int(start_num)
    files = os.listdir(path)
    if "frame_time.npy" in files:
        files.remove("frame_time.npy")
    directory = sorted(files, key=sorting_key)  # Sort the files using the custom key
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

def convert_2_video(npy_path, output_path, frame_indices=None,fps=5):
    """
    Used to convert a npy image folder to a video.
    """
    import skvideo.io
    generator = npy_frame_generator(npy_path)
    for idx,image in enumerate(generator):
        if idx == 0:
            plt.imshow(image)
            width, height = np.shape(image)
            break

    video_writer = skvideo.io.FFmpegWriter(output_path, outputdict={
                                        '-b': "30000000",
                                        '-r': str(fps),
                                        })
    
    for idx,image in enumerate(generator):
        if frame_indices is None:
            video_writer.writeFrame(np.reshape(image, [width,height,1]))
        elif idx in frame_indices:
            video_writer.writeFrame(np.reshape(image, [width,height,1]))
        if image is None:
            break
    video_writer.close()


def track_video(video_path,model,device,format="npy",indices=None, frames_per_prediction=1000,fac=1,threshold=60, max_no=1e8):
    
    if format == 'npy':
        generator = npy_frame_generator(video_path)
    elif format =="avi":
        generator = avi_frame_generator(video_path)
    images = []
    x = []
    y = []
    nbr_frames_predicted = 0
    for i,image in enumerate(generator):
        
        if indices is not None and i not in indices and image is not None:
            continue
        if image is not None:
            images.append(image)
        # A bit of an ugly solution to out-of memory problem when tracking large movies.
        if image is not None and len(np.shape(image))>1:
            if np.shape(image)[0]>1000 or np.shape(image)[1]>1000:
                frames_per_prediction = 3
        nbr_frames_predicted +=1
        if i > max_no:
            break
        if image is None or nbr_frames_predicted%frames_per_prediction == 0:
            if len(images) == 0:
                break
            print(f"Predicting up to frame {nbr_frames_predicted}")

            x_,y_ = get_torch_prediction_array(model, images,device, fac=fac, threshold=threshold)
            images = []
            x.extend(x_)
            y.extend(y_)
            if image is None:
                break
    return x,y

def track_video_smart(video_path, model, device, format="npy", indices=None, frames_per_prediction=100, fac=1, threshold=60, max_length=100000):
    if format == 'npy':
        generator = npy_frame_generator(video_path)
    elif format == "avi":
        generator = avi_frame_generator(video_path)

    images = []
    x = np.empty(max_length, dtype=np.float32)
    y = np.empty(max_length, dtype=np.float32)
    current_length = 0
    nbr_frames_predicted = 0

    for i, image in enumerate(generator):
        if indices is not None and i not in indices and image is not None:
            continue
        if image is not None:
            images.append(image)
        nbr_frames_predicted += 1
        if np.shape(image)[0]>1000 or np.shape(image)[1]>1000:
            frames_per_prediction = 3
        
        if image is None or nbr_frames_predicted % frames_per_prediction == 0:
            if len(images) == 0:
                break
            print(f"Predicting up to frame {nbr_frames_predicted}")

            x_, y_ = get_torch_prediction_array(model, images, device, fac=fac, threshold=threshold)
            images = []
            
            # Check if pre-allocated space is sufficient; if not, extend it
            if current_length + len(x_) > max_length:
                max_length = current_length + len(x_)  # New required length
                x = np.resize(x, max_length)
                y = np.resize(y, max_length)
            
            x[current_length:current_length + len(x_)] = x_
            y[current_length:current_length + len(y_)] = y_
            current_length += len(x_)

            if image is None:
                break
    
    # Trim x and y to the actual data size before returning
    return x[:current_length], y[:current_length]

def get_model_and_device(model_path="NeuralNetworks/TorchBigmodelJune_1"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU {torch.cuda.is_available()}/nDevice name: {torch.cuda.get_device_name(0)}")

    model = UNet(
        input_shape=(1, 1, 256, 256),
        number_of_output_channels=1,  # 2 for binary segmentation and 3 for multiclass segmentation
        #conv_layer_dimensions=(8, 16, 32, 64, 128, 256),  # smaller UNet (faster training)
        conv_layer_dimensions=(64, 128, 256, 512, 1024),  # standard UNet
    )   
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model, device


ticks_per_micron = 6.24 
# Functions for calculating the stokes radius and doing a stokes test


def find_movement_areas(data, axis, threshold = 200, offset=10, positive=True,margin=100):
    """
    Extracts areas with movement speed greater than a certain threshold
    """
    key = "Motor_" + axis + "_pos"
    y = data[key]/ticks_per_micron
    motor_time = (data['Motor time']-data['Motor time'][0])/1e6
    speed = (y[offset:]-y[:-offset])/(motor_time[:-offset]-motor_time[offset:]) # Speed in microns/s
    
    #return speed
    _, move_indices = extract_high_rising_indices(speed, positive, threshold=threshold, margin=margin)
    
    # Extract the mean forces in positive and negative direction
    mean_force_A_x = []
    mean_force_B_x = []

    mean_force_A_y = []
    mean_force_B_y = []
    mean_speed = []
    
    for index_pair in move_indices:
        mean_speed.append(np.mean(speed[int(index_pair[0]):int(index_pair[1])]))

        start = int(index_pair[0]*14) + 8000
        stop = int(index_pair[1]*14) - 8000
        if start > stop:
            start = int(index_pair[0]*14)
            stop = int(index_pair[1]*14)
        
        mean_force_A_x.append(np.mean(data['PSD_A_F_X'][start:stop]))
        mean_force_B_x.append(np.mean(data['PSD_B_F_X'][start:stop]))

        mean_force_A_y.append(np.mean(data['PSD_A_F_Y'][start:stop]))
        mean_force_B_y.append(np.mean(data['PSD_B_F_Y'][start:stop]))
    ret = np.mean(mean_force_A_x)
    return mean_speed, mean_force_A_x, mean_force_B_x, mean_force_A_y, mean_force_B_y

def find_movement_areas_force(data, axis, threshold = 200, offset=10, positive=True,margin=100):
    """
    Extracts areas with movement speed greater than a certain threshold.
    Used to extract the force directly
    """
    key = "Motor_" + axis + "_pos"
    y = data[key]/ticks_per_micron
    motor_time = (data['Motor time']-data['Motor time'][0])/1e6
    speed = (y[offset:]-y[:-offset])/(motor_time[:-offset]-motor_time[offset:]) # Speed in microns/s
    
    #return speed
    _, move_indices = extract_high_rising_indices(speed, positive, threshold=threshold, margin=margin)
    
    # Extract the mean forces in positive and negative direction
    mean_force_x = []
    mean_force_y = []
    mean_speed = []
    
    for index_pair in move_indices:
        mean_speed.append(np.mean(speed[int(index_pair[0]):int(index_pair[1])]))

        start = int(index_pair[0]*14) + 8000
        stop = int(index_pair[1]*14) - 8000
        if start > stop:
            start = int(index_pair[0]*14)
            stop = int(index_pair[1]*14)
        
        mean_force_x.append(np.mean(data['F_total_X'][start:stop]))
        mean_force_y.append(np.mean(data['F_total_Y'][start:stop]))
    return mean_speed, mean_force_x, mean_force_y

def find_movement_areas_Z(data, threshold = 200, offset=10, positive=True,margin=100):
    """
    Extracts areas with movement speed greater than a certain threshold
    """
    axis='z'
    key = "Motor_" + axis + "_pos"
    y = data[key]/ticks_per_micron
    motor_time = (data['Motor time']-data['Motor time'][0])/1e6
    speed = (y[offset:]-y[:-offset])/(motor_time[:-offset]-motor_time[offset:]) # Speed in microns/s
    _, move_indices = extract_high_rising_indices(speed, positive, threshold=threshold, margin=margin)
    
    # Extract the mean forces in positive and negative direction
    Photodiode_A = []
    Photodiode_B = []

    PSD_A_F_sum = []
    PSD_B_F_sum = []
    mean_speed = []
    
    for index_pair in move_indices:
        mean_speed.append(np.mean(speed[int(index_pair[0]):int(index_pair[1])]))

        start = int(index_pair[0]*14)
        stop = int(index_pair[1]*14)
        
        Photodiode_A.append(np.mean(data['Photodiode_A'][start:stop]))
        Photodiode_B.append(np.mean(data['Photodiode_B'][start:stop]))

        PSD_A_F_sum.append(np.mean(data['PSD_A_F_sum'][start:stop]))
        PSD_B_F_sum.append(np.mean(data['PSD_B_F_sum'][start:stop]))
    
    return mean_speed, Photodiode_A, Photodiode_B, PSD_A_F_sum, PSD_B_F_sum


    
def extract_high_rising_indices(array, positive=True, threshold=200, margin=0):
    """
    
    """
    rising_indices = []
    falling_indices = []
    total_indices = []
    if not positive:
        array = -array
    high = False
    
    for idx, val in enumerate(array):
        if val>threshold and high == False:
            total_indices.append(idx+margin) 
            high = True
        if val<threshold and high == True:
            if idx - margin > total_indices[-1]:
                total_indices.append(idx-margin) 
            else:
                total_indices.pop()
            high = False
    total_indices = np.array(total_indices)
    indices_2d = np.zeros([int(len(total_indices)/2),2])
    indices_2d[:,0] = total_indices[0::2] # Rising
    indices_2d[:,1] = total_indices[1::2] # falling
    return total_indices, np.array(indices_2d)

def analyze_file(filepath, axis='y', radii=2.12e-6,eta=0.9321e-3,corrected=False,d=100e-6):
    """
    Analyzes the file and calculates the sensitivity of the force sensor.
    The sensitivity is calculated using the stokes equation and the drag force.
    Can use the corrected version of the drag force which accounts for movement near walls,
    in that case the distances to the walls are needed, assumed to be the same equal to d.
    Corrected i used to toggle on/off the wall correction.
    
    """

    data = np.load(filepath, allow_pickle=True)
    analysis = find_movement_areas(data,axis,offset=10, positive=True)
    start_length = 1000 # used to be 5000, was giving problems with some fast measurements
    if axis == 'y':
        data_axis = "Y"
        id1 = 3
        id2 = 4
    elif axis == 'x':
        data_axis = "X"
        id1 = 1
        id2 = 2
    start_mean_A = np.mean(data['PSD_A_F_'+data_axis][0:start_length])
    start_mean_B = np.mean(data['PSD_B_F_'+data_axis][0:start_length])
    
    df_A = analysis[id1]-start_mean_A
    df_B = analysis[id2]-start_mean_B
    # mean_speed = np.mean(analysis[0])
    # speed_ms = mean_speed*1e-6
    # print(f"Speed is {mean_speed} microns per second")
     #23 C 0.9795e-3 # ca 1 mm*^2/s, viscocity at 21 C. lower if higher temp
    a = radii
    if corrected:
        drag = 6 * np.pi*eta*a*(1+18*a/(16*d))
    else:
        drag = 6 * np.pi*eta*a
    f = np.array(analysis[0])*1e-6*drag
    positive_A = np.mean(f*1e12/df_A)/2
    positive_B = np.mean(f*1e12/df_B)/2
    print(f"Sensitivity pos A{axis} = {positive_A} pN/bit \nSensitivity pos B{axis} = {positive_B} pN/bit")

    # DO negative
    analysis = find_movement_areas(data,axis,offset=10, positive=False)
    
    df_A = analysis[id1] - start_mean_A
    df_B = analysis[id2] - start_mean_B

    f = np.array(analysis[0])*1e-6*drag
    negative_A = np.mean(f*1e12/df_A)/2
    negative_B = np.mean(f*1e12/df_B)/2
    print(f"Sensitivity neg A{axis} = {negative_A} pN/bit \nSensitivity neg B{axis} = {negative_B} pN/bit")
    
    print(f"Sensitivity A{axis} = {round(np.mean([positive_A,negative_A]),5)} pN/bit \nSensitivity B{axis} = {round(np.mean([positive_B,negative_B]),5)} pN/bit")
    return np.mean([positive_A,negative_A]), np.mean([positive_B,negative_B])

def analyze_file_hydrodynamic_radius(filepath,eta=0.9321e-3):
    data = np.load(filepath, allow_pickle=True)
    def calc_radii(data,axis,positive):
        analysis = find_movement_areas_force(data,axis,offset=10, positive=positive)
        start_length = 5000
        if axis == 'y':
            data_axis = "Y"
            force = np.mean(analysis[2])-np.mean(data['F_total_X'][0:start_length]) # Are x and y mixed up?
        elif axis == 'x':
            data_axis = "X"
            force = np.mean(analysis[1])-np.mean(data['F_total_Y'][0:start_length])      
        drag = 6 * np.pi*eta
        radius = force/(6*np.pi*eta*np.mean(analysis[0])*1e6)
        return radius
    r1 = calc_radii(data,'y',True)
    r2 = calc_radii(data,'y',False)
    r3 = calc_radii(data,'x',True)
    r4 = calc_radii(data,'x',False)
    return np.mean([r1,r2,r3,r4]),[r1,r2,r3,r4]
    
def analyze_file_hydrodynamic_radius_wall_correction(filepath,eta=0.9321e-3,d=100e-6,force_correction=1):
    """
    Calculates the hydrodynamic radius of the particle using the stokes equation with correction
    for moving close to two walls each a distance d from the bead.
    
    """
    data = np.load(filepath, allow_pickle=True)
    def calc_radii(data,axis,positive):
        analysis = find_movement_areas_force(data,axis,offset=10, positive=positive)
        start_length = 5000
        if axis == 'y':
            data_axis = "Y"
            force = np.mean(analysis[2])-np.mean(data['F_total_X'][0:start_length]) # Are x and y mixed up?
        elif axis == 'x':
            data_axis = "X"
            force = np.mean(analysis[1])-np.mean(data['F_total_Y'][0:start_length])      
        force *= force_correction
        drag = 6 * np.pi*eta*np.mean(analysis[0])
        wall_effect = 2*(9/(16*d*1e6)) # From corrected equation for drag force
        radius = -1 / (2*wall_effect) + (force/(drag*wall_effect) + (1/(4*wall_effect**2))  )**0.5
        return radius
    r1 = calc_radii(data,'y',True)
    r2 = calc_radii(data,'y',False)
    r3 = calc_radii(data,'x',True)
    r4 = calc_radii(data,'x',False)
    return np.mean([r1,r2,r3,r4]),[r1,r2,r3,r4]


def coarse_spline_with_lsq(x, y, n_segments=20, k=3):
    """
    Fit a coarse spline to data (x, y) by specifying a small number of segments
    (n_segments). This uses LSQUnivariateSpline, which allows you to set the
    'knots' (i.e. breakpoints) explicitly.

    Parameters
    ----------
    x : array_like
        x-data (may contain duplicates, may be unsorted).
    y : array_like
        y-data corresponding to x.
    n_segments : int, optional
        Approximate number of spline segments you want. 
        The number of interior knots will be n_segments - 1.
    k : int, optional
        Degree of the spline. Defaults to 3 (cubic).

    Returns
    -------
    spline_func : LSQUnivariateSpline
        A fitted spline function that can be called as spline_func(x_new).
    (x_unique, y_agg) : (np.ndarray, np.ndarray)
        The deduplicated, sorted x-values and their aggregated y-values 
        actually used for fitting.
    """
    from scipy.interpolate import LSQUnivariateSpline
    # Convert x,y to arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # --- 1) Sort and remove duplicate x-values by averaging their y-values ---
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    x_unique = []
    y_agg = []

    current_x = x_sorted[0]
    group_ys = [y_sorted[0]]

    for i in range(1, len(x_sorted)):
        if x_sorted[i] == current_x:
            group_ys.append(y_sorted[i])
        else:
            x_unique.append(current_x)
            y_agg.append(np.mean(group_ys))
            current_x = x_sorted[i]
            group_ys = [y_sorted[i]]

    # Add the last group
    x_unique.append(current_x)
    y_agg.append(np.mean(group_ys))

    x_unique = np.array(x_unique)
    y_agg = np.array(y_agg)

    # Ensure x_unique is strictly ascending
    # (should already be from sorting + merging duplicates)
    # but just to be safe:
    if np.any(np.diff(x_unique) <= 0):
        raise ValueError("x-values must be strictly increasing after deduplication.")

    # --- 2) Choose interior knots ---
    # For n_segments segments, we want n_segments-1 interior knots.
    # LSQUnivariateSpline excludes the boundary points automatically.
    x_min, x_max = x_unique[0], x_unique[-1]
    
    # Create n_segments+1 equally spaced points from x_min to x_max,
    # then skip the first and last so those are "interior" knots.
    # This yields (n_segments - 1) interior knots.
    all_candidates = np.linspace(x_min, x_max, n_segments + 1)
    t_knots = all_candidates[1:-1]  # interior knots only
    
    # If you want fewer or more knots, adjust n_segments or your selection strategy.

    # --- 3) Fit the spline using the chosen knots (LSQUnivariateSpline) ---
    # This does a least-squares fit subject to those fixed knots.
    spline_func = LSQUnivariateSpline(x_unique, y_agg, t_knots, k=k)

    return spline_func, (x_unique, y_agg)

def subtract_background(x1, y1, x2, y2, num_splines=20):
    f,_ = coarse_spline_with_lsq(x2,y2,num_splines)
    return x1, y1 - f(x1)