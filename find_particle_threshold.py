import cv2
import scipy
import numpy as np
import scipy.ndimage as ndi
from skimage import measure

#from numba import njit, jit


#@njit(parallel=True)
def parallel_center_of_masses(data):
    # Without "parallel=True" in the jit-decorator
    # the prange statement is equivalent to range
    ret_x = []
    ret_y = []
    for d in data:
        #ret.append(center_of_mass(d))
        pos = np.nonzero(d)
        tot = len(d[0])#np.sum(d)
        #tot = np.shape(pos)[1]
        x = np.sum(pos[0][:]) # check if these two calculations can be exchanged for one
        y = np.sum(pos[1][:])
        ret_x.append(x/tot)
        ret_y.append(y/tot)
    return ret_x, ret_y

def find_single_particle_center(img,threshold=127):
    """
    Locate the center of a single particle in an image.
    Obsolete now, find_particle_centers is surperior in every way
    """
    img_temp = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img_temp,threshold,255,cv2.THRESH_BINARY)
    cy, cx = ndi.center_of_mass(th1)
    # if np.isnan(cx) return inf?
    return cx,cy,th1

def threshold_image(image,threshold=120, bright_particle=True):
        img_temp = cv2.medianBlur(image,5)
        if bright_particle:
            ret,thresholded_image = cv2.threshold(img_temp,threshold,255,cv2.THRESH_BINARY)
            return thresholded_image
        else:
            ret,thresholded_image = cv2.threshold(img_temp,threshold,255,cv2.THRESH_BINARY_INV)
            return thresholded_image

def find_groups_of_interest(counts, particle_upper_size_threshold,
                            particle_size_threshold, separate_particles_image):
    '''
    Exctract the particles into separate images to be center_of_massed in parallel
    '''
    particle_images = []
    for group, pixel_count in enumerate(counts): # First will be background
        if particle_upper_size_threshold>pixel_count>particle_size_threshold:
            #target_groups.append(group)
            particle_images.append(separate_particles_image==group)
    return particle_images


def get_x_y(counts, particle_upper_size_threshold, particle_size_threshold,
            separate_particles_image):
    x = []
    y = []
    for group, pixel_count in enumerate(counts): # First will be background
        if particle_upper_size_threshold>pixel_count>particle_size_threshold:
            # TODO: Parallelize this thing
            # Particle found, locate center of mass of the particle
            cy, cx = ndi.center_of_mass(separate_particles_image==group)

            x.append(cx)
            y.append(cy)
    return x, y

def find_particle_centers_fast(image, threshold=120, particle_size_threshold=200,
                          particle_upper_size_threshold=5000, bright_particle=True,
                          fill_holes=False, check_circular=False):
    """
    Function which locates particle centers using OpenCV's connectedComponentsWithStats.
    
    Parameters:
        image: Image with the particles.
        threshold: Threshold value of the particle.
        particle_size_threshold: Minimum area of particle in image measured in pixels.
        particle_upper_size_threshold: Maximum area of particle in image measured in pixels.
        bright_particle: If the particle is brighter than the background or not.
        fill_holes: If true, fills holes in binary objects.
        check_circular: If true, checks if the object is circular (not implemented in this version).
        
    Returns:
        x, y: Arrays with the x and y coordinates of the particle in the image in pixels.
              Returns empty arrays if no particle was found.
        thresholded_image: The binary image after thresholding.
    """
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
            # Valid particle, get its center
            cx, cy = centroids[i]
            # Circular check would go here, but it requires a custom implementation
            x.append(cx)
            y.append(cy)

    return np.array(x), np.array(y), thresholded_image

#@jit
def find_particle_centers(image,threshold=120, particle_size_threshold=200,
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
    #cv2.medianBlur(image, 5) > threshold # Added thresholding here

    # Separate the thresholded image into different sections
    separate_particles_image = measure.label(thresholded_image)
    # use cv2.findContours instead?
    # Count the number of pixels in each section
    counts = np.bincount(np.reshape(separate_particles_image,(np.shape(separate_particles_image)[0]*np.shape(separate_particles_image)[1])))

    x = []
    y = []

    for group, pixel_count in enumerate(counts): # First will be background
        if particle_upper_size_threshold>pixel_count>particle_size_threshold:
            # Particle found, locate center of mass of the particle
            # TODO here we should really use the brighness weighted center of mass, not just the center of mass
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

    return x, y, thresholded_image


def find_pipette_top(image,threshold=120, particle_size_threshold=10_000,
                        particle_upper_size_threshold=1000_000, fill_holes=False, ratio=2):
    """
    Function which locates pipette top  using thresholding.
    Parameters :
        image - Image with the pipette
        threshold - Threshold value of the pipette
        particle_size_threshold - minimum area of particle in image measured in pixels
        bright_particle - If the pipette is brighter than the background or not
    Returns :
        x,y - arrays with the x and y coordinates of the pipette in the image in pixels.
            Returns empty arrays if no pipette was found
    """
    # TODO add a downsample here
    gf = scipy.ndimage.gaussian_filter(image,sigma=20) # Add sigma as a parameter
    res = scipy.ndimage.gaussian_filter(image-gf,3)
    # Do thresholding of the image
    thresholded_image = cv2.blur(res, (8, 8)) > threshold
    if fill_holes:
        # Fill holes in the image before labeling
        thresholded_image = ndi.morphology.binary_fill_holes(thresholded_image)

    # Separate the thresholded image into different sections
    separate_particles_image = measure.label(thresholded_image)
    # Count the number of pixels in each section
    counts = np.bincount(np.reshape(separate_particles_image,(np.shape(separate_particles_image)[0]*np.shape(separate_particles_image)[1])))

    for group, pixel_count in enumerate(counts): # First will be background
        if particle_upper_size_threshold>pixel_count>particle_size_threshold:
            targ = separate_particles_image==group
            contours = [np.argwhere(targ).astype(np.int32)]
            x, y, w, h = cv2.boundingRect(contours[0])

            # Checking aspect ratio to make highten the likelihood that we are in the vicinity of the pippette
            if w / h > ratio:
                y_t = np.argmax(targ[x,:]) # Want top pixel.
                return x, y_t, targ
    return None, None, None


def find_pipette_top_GPU(image, threshold=120, particle_size_threshold=10_000,
                     particle_upper_size_threshold=1000_000, fill_holes=False, ratio=2,
                      subtract_particles=False,positions=None,radii=50):
    """
    Function which locates pipette top using thresholding. Also uses GPU and specifically cupy.
    Significantly faster than the CPU version if there is a GPU available.
    Parameters :
        image - Image with the pipette
        threshold - Threshold value of the pipette
        particle_size_threshold - minimum area of pipette in image measured in pixels
        bright_particle - If the pipette is brighter than the background or not
    Returns :
        x,y - arrays with the x and y coordinates of the pipette in the image in pixels.
            Returns empty arrays if no pipette was found
    """
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndi
    from skimage.measure import label
    image_gpu = cp.array(image)

    if subtract_particles:
        fac = 1
        blur = cp_ndi.gaussian_filter(image_gpu,300)
        for pos in positions:
            #image_gpu[pos[0]-radii:pos[0]+radii,pos[1]-radii*fac:pos[1]+radii*fac] = blur[pos[0]-radii:pos[0]+radii,pos[1]-radii*fac:pos[1]+radii*fac]
            image_gpu[pos[1]-radii:pos[1]+radii,pos[0]-radii*fac:pos[0]+radii*fac] = blur[pos[1]-radii:pos[1]+radii,pos[0]-radii*fac:pos[0]+radii*fac]

    gf = cp_ndi.gaussian_filter(image_gpu, sigma=20)
    res = cp_ndi.gaussian_filter(image_gpu - gf, 3)

    # Do thresholding of the image
    thresholded_image = cp.asarray(cv2.blur(cp.asnumpy(res), (8, 8))) > threshold
    if fill_holes:
        # Fill holes in the image before labeling
        thresholded_image = cp_ndi.morphology.binary_fill_holes(thresholded_image)


    separate_particles_image = measure.label(cp.asnumpy(thresholded_image))
    # Count the number of pixels in each section
    # TODO do this step with CV2 instead to make it faster.
    counts = np.bincount(np.reshape(separate_particles_image,(np.shape(separate_particles_image)[0]*np.shape(separate_particles_image)[1])))

    for group, pixel_count in enumerate(counts): # First will be background
        if particle_upper_size_threshold>pixel_count>particle_size_threshold:
            targ = separate_particles_image==group
            contours = [np.argwhere(targ).astype(np.int32)]
            x, y, w, h = cv2.boundingRect(contours[0])
            # Checking ratio to make highten the likelihood that we are in the vicinity of the pippette
            if w / h > ratio:
                y_t = np.argmax(targ[x,:]) # Want top pixel.
                return x, y_t, cp.asnumpy(thresholded_image)#targ
    return None, None, None
