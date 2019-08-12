"""
Huanxiang Su
CS410: Intro to Computer Vision
Project: SIFT Implementation
03/17/2019
"""

import sys
import math
import cv2 as cv
import numpy as np

########################3############ Constant #########################################
USE_DEFAULT_OCTAVE = 0
DEFAULT_OCTAVES = 4   # default number of scale space
S = 3    # intervals, s + 3 images in the stack of blurred img for each octave
K = 2 ** (1 / S)
#INITIAL_SIGMA = K / 2
INITIAL_SIGMA = 1.6
PRE_SIGMA = 0.5   # pre smooth
NORMALIZE = 1     # normalize the input image to range [0..1]
CONTRAST_THRESHOLD = 0.04   # for low-contrast
R = 10    # for curvature test
CURVATURE_THRESHOLD = (R + 1)**2 / R    # for edge response 
ILLUMINATION_THRESHOLD = 0.2   # for descriptor intensity
MAX_KPT_INTER = 5      # max time to perform keypoint location interpolation
MAX_ORIENT_PASS = 2    # max num of different orientations for each keypoint

if NORMALIZE == 1:
    PRE_CONTRAST_THRESHOLD = 0.5 * CONTRAST_THRESHOLD / S   # for low-contrast extrema
else:
    PRE_CONTRAST_THRESHOLD = 0.5 * CONTRAST_THRESHOLD / S * 255.0   # for low-contrast extrema

#################################### Constant ##########################################


class mysift:
    def __init__(self, my_img):
        """
        SIFT Initialization
        """
        self.img = np.copy(my_img)
        if len(self.img.shape) >= 3:
            self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY).astype(np.float32)

        if NORMALIZE == 1:
            self.img = self.img / 255.0

        if USE_DEFAULT_OCTAVE == 0:
            self.octaves = int(np.log(min(self.img.shape[0], self.img.shape[1])) / np.log(2) - 2)
        else:
            self.octaves = DEFAULT_OCTAVES   # number of octave

        self.intervals = S           # number of intervals
        self.pyramid_list = []       # 2d octaves pyramid list
        self.dog_list = []           # 2d difference of gaussian list
        self.extrema_list = []       # 2d array to store extrema list
        self.magnitude_list = []     # 2d array to store magnitude information
        self.orientation_list = []   # 2d array to store orientation information
        self.keypoints = []          # keypoints list
        self.descriptors = []        # descriptors list


    def my_detect_and_compute(self):
        if self.keypoints != []:
            if self.descriptors == []:
                self.compute_descriptor()
                self.free_list()
                return self.keypoints, self.descriptors
            else:
                return self.keypoints, self.descriptors
        self.build_scale_space()
        self.detect_extrema()
        self.local_orientation_assign()
        if self.keypoints != []:
            self.keypoints = sorted(self.keypoints, key=sort_key)
        self.compute_descriptor()
        self.free_list()
        return self.keypoints, self.descriptors


    def my_detect_keypoints(self):
        if self.keypoints != []:
            return self.keypoints
        self.build_scale_space()
        self.detect_extrema()
        self.local_orientation_assign()
        if self.keypoints != []:
            self.keypoints = sorted(self.keypoints, key=sort_key)
        return self.keypoints


    def my_compute_descriptors(self):
        if self.keypoints == []:
            return []
        self.compute_descriptor()
        self.free_list()
        return self.descriptors

    

    def build_scale_space(self):
        '''
        build gaussian pyramid and difference of guassian (DoG) pyramid
        '''
        # Pre-smooth doubled size image
        double_img = img_resize(self.img, 2)   # resize using bilinear interpolation
        ini = np.sqrt(INITIAL_SIGMA ** 2 - PRE_SIGMA ** 2 * 4)
        double_img = cv.GaussianBlur(double_img, (0,0), ini)
        
        my_sig_list = np.zeros([S + 3], dtype=np.float32)
        my_sig_list[0] = INITIAL_SIGMA
        my_sig_list[1] = INITIAL_SIGMA * np.sqrt(K ** 2 - 1)
        for i in range(2, self.intervals + 3):
            my_sig_list[i] = my_sig_list[i-1] * K
        
        #print ("my_sig: ", my_sig_list)
        
        for i in range(0, self.octaves):
            a_list = []
            for j in range(0, self.intervals + 3):
                if i == 0 and j == 0:  # pre smoothed doubled image
                    my_img = double_img
                    a_list.append(my_img)
                elif j == 0:  # downsample, use image from previous octave that is twice of the first sigma
                    my_img = self.pyramid_list[i-1][self.intervals]
                    my_img = img_resize(my_img, 0.5)  # halved
                    a_list.append(my_img)
                else:
                    my_img = cv.GaussianBlur(a_list[-1], (0,0), my_sig_list[j])
                    a_list.append(my_img)
                #save_image('pyramid/img'+str(i)+'-'+str(j)+'.jpg', my_img)
            self.pyramid_list.append(a_list)


        # do difference of Gaussian for each octaves
        for i in range(0, self.octaves):
            a_dog_list = []
            for j in range(0, self.intervals + 2):
                a_dog = self.pyramid_list[i][j+1] - self.pyramid_list[i][j]
                a_dog_list.append(a_dog)
                #save_image('dog/img'+str(i)+'-'+str(j)+'.jpg', a_dog)
            self.dog_list.append(a_dog_list)
    # end def



    def detect_extrema(self):
        real_num = 0
        extrema_num = 0

        # initialize extrema list
        for i in range(0, self.octaves):
            height, width = self.pyramid_list[i][0].shape
            a_list = []
            for j in range(0, self.intervals):
                aa = np.zeros([height, width, 6], dtype=np.float32)
                a_list.append(aa)
            self.extrema_list.append(a_list)
        
        # find extrema
        for i in range(0, self.octaves):
            a_dog_list = self.dog_list[i]
            height, width = a_dog_list[0].shape

            for j in range(1, self.intervals + 1):  # j for each dog between
                top = a_dog_list[j-1]  # above the dog
                middle = a_dog_list[j]  # a dog image to compare
                bottom = a_dog_list[j+1]  # below the dog

                for h in range(1, height - 1):
                    for w in range(1, width - 1):
                        #if np.absolute(middle[h][w]) <= PRE_CONTRAST_THRESHOLD:  # filter out low-contrast points first to avoid any unnecessary computations.
                         #   continue

                        is_extrema = False
                        my_point = middle[h][w]
                        # compare all 26 neighbors
                        if my_point < 0:  # check if it is minima
                            is_minima = True
                            is_minima = (is_minima and my_point < middle[h-1][w-1] and my_point < middle[h-1][w] and my_point < middle[h-1][w+1]
                                and my_point < middle[h][w-1] and my_point < middle[h][w+1]
                                and my_point < middle[h+1][w-1] and my_point < middle[h+1][w] and my_point < middle[h+1][w+1] 
                                and my_point < top[h-1][w-1] and my_point < top[h-1][w] and my_point < top[h-1][w+1] 
                                and my_point < top[h][w-1] and my_point < top[h][w] and my_point < top[h][w+1] 
                                and my_point < top[h+1][w-1] and my_point < top[h+1][w] and my_point < top[h+1][w+1] 
                                and my_point < bottom[h-1][w-1] and my_point < bottom[h-1][w] and my_point < bottom[h-1][w+1] 
                                and my_point < bottom[h][w-1] and my_point < bottom[h][w] and my_point < bottom[h][w+1] 
                                and my_point < bottom[h+1][w-1] and my_point < bottom[h+1][w] and my_point < bottom[h+1][w+1])
                            if is_minima:
                                is_extrema = True
                        
                        elif my_point > 0:  # else check if it is maxima
                            is_maxima = True
                            is_maxima = (is_maxima and my_point > middle[h-1][w-1] and my_point > middle[h-1][w] and my_point > middle[h-1][w+1] 
                                and my_point > middle[h][w-1] and my_point > middle[h][w+1] 
                                and my_point > middle[h+1][w-1] and my_point > middle[h+1][w] and my_point > middle[h+1][w+1] 
                                and my_point > top[h-1][w-1] and my_point > top[h-1][w] and my_point > top[h-1][w+1] 
                                and my_point > top[h][w-1] and my_point > top[h][w] and my_point > top[h][w+1] 
                                and my_point > top[h+1][w-1] and my_point > top[h+1][w] and my_point > top[h+1][w+1] 
                                and my_point > bottom[h-1][w-1] and my_point > bottom[h-1][w] and my_point > bottom[h-1][w+1] 
                                and my_point > bottom[h][w-1] and my_point > bottom[h][w] and my_point > bottom[h][w+1] 
                                and my_point > bottom[h+1][w-1] and my_point > bottom[h+1][w] and my_point > bottom[h+1][w+1])
                            if is_maxima:
                                is_extrema = True
                        
                        if is_extrema == True:  # if the point is extrema over all 26 neighbors
                            # generate subpixels, filter low contrast, check edge response
                            extrema_num += 1
                            if self.extrema_interpolation(i, j, a_dog_list, h, w) == True:
                                real_num += 1
                    # end for w
                # end for h
            # end for j
        # end for i
        #print ('total extrema = ', extrema_num)
        #print ('total real extrema = ', real_num)
    # end def


    def extrema_interpolation(self, oc, interval, a_dog_list, current_h, current_w):
        i = 0
        h = current_h
        w = current_w
        s = interval
        height, width = a_dog_list[0].shape
        d1 = 0.5
        d2 = 1.0
        d3 = 0.25

        while i < MAX_KPT_INTER:   # Max time to do keypoint interpolation
            top = a_dog_list[s - 1]  # above the dog
            middle = a_dog_list[s]  # a dog image to compare
            bottom = a_dog_list[s + 1]  # below the dog
            my_point = middle[h][w]
            if NORMALIZE == 0:
                Dx = (middle[h][w+1] - middle[h][w-1]) * d1 / 255.0
                Dy = (middle[h+1][w] - middle[h-1][w]) * d1 / 255.0
                Ds = (bottom[h][w] - top[h][w]) * d1 / 255.0
                Dxx = (middle[h][w-1] + middle[h][w+1] - 2*my_point) * d2 / 255.0
                Dyy = (middle[h-1][w] + middle[h+1][w] - 2*my_point) * d2 / 255.0
                Dss = (bottom[h][w] + top[h][w] - 2*my_point) * d2 / 255.0
                Dxy = (middle[h+1][w+1] - middle[h+1][w-1] - middle[h-1][w+1] + middle[h-1][w-1]) * d3 / 255.0
                Dxs = (bottom[h][w+1] - bottom[h][w-1] - top[h][w+1] + top[h][w-1])  * d3 / 255.0
                Dys = (bottom[h+1][w] - bottom[h-1][w] - top[h+1][w] + top[h-1][w])  * d3 / 255.0
            else:  # already normalized in range [0, 1]
                Dx = (middle[h][w+1] - middle[h][w-1]) * d1
                Dy = (middle[h+1][w] - middle[h-1][w]) * d1
                Ds = (bottom[h][w] - top[h][w]) * d1
                Dxx = (middle[h][w-1] + middle[h][w+1] - 2*my_point)
                Dyy = (middle[h-1][w] + middle[h+1][w] - 2*my_point)
                Dss = (bottom[h][w] + top[h][w] - 2*my_point)
                Dxy = (middle[h+1][w+1] - middle[h+1][w-1] - middle[h-1][w+1] + middle[h-1][w-1]) * d3
                Dxs = (bottom[h][w+1] - bottom[h][w-1] - top[h][w+1] + top[h][w-1]) * d3
                Dys = (bottom[h+1][w] - bottom[h-1][w] - top[h+1][w] + top[h-1][w]) * d3
            
            my_D = np.matrix([[Dx], [Dy], [Ds]], dtype=np.float32)
            my_H = np.matrix([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]], dtype=np.float32)
            xx = np.linalg.lstsq(my_H, my_D, rcond=None)[0]   # solve x for Ax = b
            offset_s = -float(xx[2])
            offset_y = -float(xx[1])
            offset_x = -float(xx[0])

            if abs(offset_s) < 0.5 and abs(offset_x) < 0.5 and abs(offset_y) < 0.5:  # success
                break

            h += int(np.round(offset_y))
            w += int(np.round(offset_x))
            s += int(np.round(offset_s))

            if h < 3 or w < 3 or h >= height - 3 or w >= width - 3 or s < 1 or s > self.intervals:  # discard border
                return False
            i += 1
        # end while

        if i == MAX_KPT_INTER:  # discard this extrema
            return False

        # update latest location
        top = a_dog_list[s - 1]  # above the dog
        middle = a_dog_list[s]  # a dog image to compare
        bottom = a_dog_list[s + 1]  # below the dog
        my_point = middle[h][w]

        # check low-contrast
        if NORMALIZE == 0:
            Dx = (middle[h][w+1] - middle[h][w-1]) * d1 / 255.0
            Dy = (middle[h+1][w] - middle[h-1][w]) * d1 / 255.0
            Ds = (bottom[h][w] - top[h][w]) * d1 / 255.0
            a_point = my_point / 255.0
        else:
            Dx = (middle[h][w+1] - middle[h][w-1]) * d1
            Dy = (middle[h+1][w] - middle[h-1][w]) * d1
            Ds = (bottom[h][w] - top[h][w]) * d1
            a_point = my_point
        my_D = [Dx, Dy, Ds]
        my_xx = [offset_x, offset_y, offset_s]
        contrast = a_point + 0.5 * np.dot(my_D, my_xx)
        if np.abs(contrast) < (CONTRAST_THRESHOLD / self.intervals):  # low-contrast
            return False


        # check edge by principal curvatures
        if NORMALIZE == 0:
            Dxx = (middle[h][w-1] + middle[h][w+1] - 2*my_point) * d2 / 255.0
            Dyy = (middle[h-1][w] + middle[h+1][w] - 2*my_point) * d2 / 255.0
            Dxy = (middle[h+1][w+1] - middle[h+1][w-1] - middle[h-1][w+1] + middle[h-1][w-1]) * d3 / 255.0
        else:
            Dxx = (middle[h][w-1] + middle[h][w+1] - 2*my_point)
            Dyy = (middle[h-1][w] + middle[h+1][w] - 2*my_point)
            Dxy = (middle[h+1][w+1] - middle[h+1][w-1] - middle[h-1][w+1] + middle[h-1][w-1]) * d3

        TrH = Dxx + Dyy
        DetH = (Dxx*Dyy) - (Dxy ** 2)

        cur_ratio = (TrH ** 2) / DetH
        if cur_ratio < CURVATURE_THRESHOLD and DetH > 0:
            self.extrema_list[oc][s - 1][h][w][0] = 1   # to indicate this is a keypoint
            self.extrema_list[oc][s - 1][h][w][1] = offset_x   # store offset, to be used later
            self.extrema_list[oc][s - 1][h][w][2] = offset_y
            self.extrema_list[oc][s - 1][h][w][3] = offset_s
            size = (INITIAL_SIGMA * (2 ** ((s + offset_s) / S + oc)))
            #scale = size / (2 ** oc)
            self.extrema_list[oc][s - 1][h][w][4] = np.abs(contrast)   # keypoint response
            self.extrema_list[oc][s - 1][h][w][5] = size   # keypoint size
            return True



    def local_orientation_assign(self):
        # initialize magnitude and orientation list
        for i in range(0, self.octaves):
            m_list = []
            o_list = []
            height, width = self.dog_list[i][0].shape
            for j in range(0, self.intervals):
                mm = np.zeros([height, width], dtype=np.float32)
                oo = np.zeros([height, width], dtype=np.float32)
                m_list.append(mm)
                o_list.append(oo)
            self.magnitude_list.append(m_list)
            self.orientation_list.append(o_list)


        # pre-compute the gradient magnitude and orientation for each image sample for efficiency
        for i in range(0, self.octaves):    
            a_dog_list = self.dog_list[i]
            #a_pry_list = self.pyramid_list[i]
            height, width = a_dog_list[0].shape

            for j in range(0, self.intervals):
                a_dog = a_dog_list[j+1]
                #a_pry = a_pry_list[j+1]
                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        xx = a_dog[y][x+1] - a_dog[y][x-1]
                        yy = a_dog[y+1][x] - a_dog[y-1][x]
                        #xx = a_pry[y][x+1] - a_pry[y][x-1]
                        #yy = a_pry[y-1][x] - a_pry[y+1][x]
                        magnitude = np.sqrt((xx * xx) + (yy * yy))
                        orientation = math.atan2(yy, xx)
                        self.magnitude_list[i][j][y][x] = magnitude
                        self.orientation_list[i][j][y][x] = orientation


        # assign orientation to each keypoint
        for i in range(0, self.octaves):
            height = self.extrema_list[i][0].shape[0]
            width = self.extrema_list[i][0].shape[1]

            for j in range(1, self.intervals + 1):

                for y in range(1, height - 1):

                    for x in range(1, width - 1):
                        if self.extrema_list[i][j-1][y][x][0] == 1:
                            orientation_histogram = np.zeros([36], dtype=np.float32)
                            
                            kernel_size = int(round((self.extrema_list[i][j-1][y][x][5]) / (2 ** i) * 4.5 * 2 + 1))
                            if kernel_size % 2 == 0:
                                kernel_size -= 1
                            sig = (self.extrema_list[i][j-1][y][x][5]) / (2 ** i) * 1.5   # 1.5*scale
                            circular_guassian_window = guassian_function(kernel_size, sig)  #  Gaussian-weighted circular window
                            kernel_size = int(np.floor(kernel_size / 2))  # convert to radius

                            for yy in range(-1 * kernel_size, kernel_size + 1):

                                for xx in range(-1 * kernel_size, kernel_size + 1):
                                    if y + yy < 0 or x + xx < 0 or y + yy > height - 1 or x + xx > width - 1:
                                        continue   # skip out of bound
                                    # Each sample added to the histogram is weighted by its gradient magnitude
                                    #w = np.exp(-(yy*yy + xx*xx) / (2*sig*sig))
                                    #weight = self.magnitude_list[i][j-1][y+yy][x+xx] * w
                                    weight = self.magnitude_list[i][j-1][y+yy][x+xx] * circular_guassian_window[yy+kernel_size][xx+kernel_size]
                                    orient = self.orientation_list[i][j-1][y+yy][x+xx]
                                    #print (orient)
                                    orientation = int((orient + np.pi) * (180 / np.pi))   # convert to degree
                                    bin_index = np.clip(int(orientation / 10), 0, 35)   # 36 bins for 360 degrees, 360/36 = 10
                                    orientation_histogram[bin_index] += weight
                                # end xx
                            # end yy

                            # get calculated subpixel location, offset solved in detect extrema()
                            x_coordinate = (x + self.extrema_list[i][j-1][y][x][1]) * (2 ** (i-1))  # subpixel x coordinate
                            y_coordinate = (y + self.extrema_list[i][j-1][y][x][2]) * (2 ** (i-1))  # subpixel y coordinate   
                            response = self.extrema_list[i][j-1][y][x][4]
                            size = (self.extrema_list[i][j-1][y][x][5])

                            # smooth orientation histogram
                            copy_hist = np.copy(orientation_histogram)
                            prev = 35
                            for ii in range(0, 36):
                                nex = ii + 1
                                if nex == 36:
                                    nex = 0
                                #orientation_histogram[ii] = (copy_hist[prev] + copy_hist[ii] + copy_hist[nex]) / 3.0   # box blur
                                orientation_histogram[ii] = 0.25 * copy_hist[prev] + 0.5 * copy_hist[ii] + 0.25 * copy_hist[nex]   # gaussian blur
                                prev = ii

                            # find peak orientation and above 80% of peak
                            peak = np.max(orientation_histogram)
                            accept_value = peak * 0.8   # 80% of peak value
                            num_pass = 0
                            copy_hist = np.copy(orientation_histogram)
                            for iii in range(0, 36):
                                peak = np.max(copy_hist)
                                ii = np.argmax(copy_hist)
                                if peak >= accept_value:   # a good peak
                                    # fit a parabola to the 3 histogram values closest to each peak to interpolate the peak position for better accuracy
                                    if ii == 0:
                                        y1 = orientation_histogram[35]
                                        y3 = orientation_histogram[1]
                                    elif ii == 35:
                                        y1 = orientation_histogram[34]
                                        y3 = orientation_histogram[0]
                                    else:
                                        y1 = orientation_histogram[ii - 1]
                                        y3 = orientation_histogram[ii + 1]
                                    y2 = orientation_histogram[ii]
                                    if y2 <= y1 or y2 <= y3:   # discard peak that is close to other peaks
                                        copy_hist[ii] = 0
                                        continue

                                    ori_bin = ii + 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3)
                                    while ori_bin < 0.0:
                                        ori_bin += 36.0
                                    while ori_bin >= 36.0:
                                        ori_bin -= 36.0
                                    ori = ori_bin * 10.0
                                    #ori = 360.0 - ori
                                    #keypoint = [inter_x, inter_y, x, y, angle, octave, interval, response, size]
                                    a_keypoint = [x_coordinate, y_coordinate, x, y, ori, i, j, response, size]
                                    self.keypoints.append(a_keypoint)
                                    num_pass += 1
                                else:
                                    break
                                copy_hist[ii] = 0
                                if num_pass == MAX_ORIENT_PASS:  # Max is 2 ori
                                    break
                        # end if extrema == 1
                    # end x
                # end y for a sample image
            # end j for each interval
        # end i for each octave
        return



    def compute_descriptor(self):
        self.descriptors = np.zeros([len(self.keypoints), 128], dtype=np.float32)
        #my_guassian = guassian_function(16, 8)
        pp = 2 * np.pi / 36   # helper var
        bin_degree = 8.0 / (2 * np.pi)   # 8 bins for each 45 degrees, 0-1.27 => bin 0

        for i in range(0, len(self.keypoints)):
            xx = self.keypoints[i][2]  # extract discrete coordinate
            yy = self.keypoints[i][3]
            orientation = pp * (self.keypoints[i][4] / 10.0) - np.pi  # convert to [-3.14, 3.14]
            cos_t = np.cos(orientation)
            sin_t = np.sin(orientation)
            oc_idx = self.keypoints[i][5]   # octave index
            interval_idx = (self.keypoints[i][6]) - 1   # interval index
            scale = (self.keypoints[i][-1]) / (2 ** oc_idx)
            height, width = self.magnitude_list[oc_idx][interval_idx].shape
            my_hist = np.zeros([4,4,8], dtype=np.float32)  # each 4x4 window has 8 bins
            my_des = np.zeros([128], dtype=np.float32)

            histw = 3.0 * scale
            radius = int(histw * np.sqrt(2) * 2.5 + 0.5)
            for y in range(-radius, radius + 1):
                for x in range(-radius, radius + 1):
                    x_rot = (x * cos_t - y * sin_t) / histw
                    y_rot = (x * sin_t + y * cos_t) / histw
                    my_hist_x = x_rot + 1.5
                    my_hist_y = y_rot + 1.5
                    if my_hist_x > -1.0 and my_hist_x < 4 and my_hist_y > 1.0 and my_hist_y < 4:
                        xxx = xx + x
                        yyy = yy + y
                        if xxx > 0 and xxx < width - 1 and yyy > 0 and yyy < height - 1:
                            my_mag = self.magnitude_list[oc_idx][interval_idx][yyy][xxx]
                            my_ori = self.orientation_list[oc_idx][interval_idx][yyy][xxx]
                            my_ori -= orientation
                            while my_ori < 0.0:  # within [0, 2pi]
                                my_ori += (2 * np.pi)
                            while my_ori >= (2 * np.pi):
                                my_ori -= (2 * np.pi)
                            my_hist_bin = my_ori * bin_degree
                            w = np.exp( -(x_rot * x_rot + y_rot * y_rot) / 8.0 )  # circular Gaussian blur
                            weight = w * my_mag
                            floor_y = np.floor(my_hist_y)
                            floor_x = np.floor(my_hist_x)
                            floor_bin = np.floor(my_hist_bin)
                            d_y = my_hist_y - floor_y
                            d_x = my_hist_x - floor_x
                            d_bin = my_hist_bin - floor_bin

                            # perform trilinear interpolation
                            # weight * (1 - d) each dimension
                            for aa in range(0, 2):
                                yb = floor_y + aa
                                if yb >= 0 and yb < 4:  # if within 4x4
                                    if aa == 0:
                                        value_y = weight * (1 - d_y)
                                    else:
                                        value_y = weight * d_y
                                    y_idx = int(yb)
                                    for bb in range(0, 2):
                                        xb = floor_x + bb
                                        if xb >= 0 and xb < 4: # if within 4x4
                                            if bb == 0:
                                                value_x = value_y * (1 - d_x)
                                            else:
                                                value_x = value_y * d_x
                                            x_idx = int(xb)
                                            for cc in range(0, 2):
                                                b_idx = int((floor_bin + cc) % 8)  # bin index
                                                if cc == 0:
                                                    my_value = value_x * (1 - d_bin)
                                                else:
                                                    my_value = value_x * d_bin
                                                my_hist[y_idx][x_idx][b_idx] += my_value
                            # end interpolation
                        # end if xxx
                    # end if
                # end for x
            # end for y (hist finished)
            des_idx = 0
            for y in range(0, 4):
                for x in range(0, 4):
                    for k in range(0, 8):
                        my_des[des_idx + k] = my_hist[y][x][k]
                    des_idx += 8
            
            # check illumination
            norm = np.linalg.norm(my_des)
            my_des = my_des / norm   # normalize vector
            my_des = np.clip(my_des, 0, ILLUMINATION_THRESHOLD)   # threshold = 0.2
            norm = np.linalg.norm(my_des)
            my_des = my_des / norm   # normalize again after illumination check

            # convert to integer value
            for j in range(128):
                my_des[j] = int(my_des[j] * 512.0)
            my_des = np.clip(my_des, 0, 255)

            self.descriptors[i] = my_des  # store to descriptors list

        # end each keypoint
        return self.descriptors



    def draw_keypoints(self, my_img, my_color=(200,200,50)):
        img = np.copy(my_img)
        if len(self.keypoints) == 0:
            return

        for kp in self.keypoints:
            kpx = kp[0]
            kpy = kp[1]
            angle = 360.0 - kp[4]
            radius = int(round(kp[-1] / 2))   # size
            y = np.sin(np.radians(angle))
            x = np.cos(np.radians(angle))
            y = radius * y
            x = radius * x
            x = kpx + x
            y = kpy - y
            cv.circle(img, (int(kpx), int(kpy)), radius, my_color, 1, lineType=cv.LINE_8)
            cv.line(img, (int(kpx), int(kpy)), (int(x), int(y)), (0,0,255), 1)
        return img
    

    def free_list(self):
        self.pyramid_list = []
        self.dog_list = []
        self.extrema_list = []
        self.magnitude_list = []
        self.orientation_list = []
    


# image resizing using bilinear interpolation
def img_resize(img, r):
    if r == 1:  # original image
        return img

    size_y = int(img.shape[0] * r)
    size_x = int(img.shape[1] * r)
    resize_img = np.zeros([size_y, size_x], dtype=np.float32)

    for y in range(0, size_y):
        for x in range(0, size_x):
            yy = y / r
            xx = x / r
            i = int(np.floor(xx))
            j = int(np.floor(yy))
            a = xx - i
            b = yy - j
            if i >= img.shape[1]-1 or j >= img.shape[0]-1:
                result = img[j, i]
            else:
                bi1 = ((1 - a) * (1 - b)) * (img[j, i])
                bi2 = (a * (1 - b)) * (img[j, i + 1])
                bi3 = (a * b) * (img[j + 1, i + 1])
                bi4 = ((1 - a) * b) * (img[j + 1, i])
                result = bi1 + bi2 + bi3 + bi4
            resize_img[y][x] = result
    return resize_img


# generate a circular guassian kernel with given size and sigma
def guassian_function(w_size, sigma):
    my_guassian = np.zeros([w_size, w_size], dtype=np.float32)
    kernel_size = w_size / 2 - 0.5

    for i in range(0, w_size):
        for j in range(0, w_size):
            x = i - kernel_size
            y = j - kernel_size
            a = (1.0 / (2*np.pi*sigma*sigma)) * (np.exp(-(x**2 + y**2) / (2*sigma*sigma)))
            my_guassian[i][j] = a

    total = np.sum(my_guassian)
    my_guassian = my_guassian / total
    return my_guassian



def save_image(filename, img):
    cv.imwrite(filename, img)


def show_image(window_name, img):
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(window_name, img)
    cv.waitKey(0)

def sort_key(a):
    return a[0]


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            result_filename = sys.argv[2]
        else:
            result_filename = "result.png"
    else:
        print ("Error! No input file!")
        exit(0)

    img = cv.imread(filename)
    
    my_sift = mysift(img)
    kp, des = my_sift.my_detect_and_compute()
    #kp = my_sift.my_detect_keypoints()
    #des = my_sift.compute_descriptor()

    """
    for i in kp:
        print (i)
    print ('keypoint len = ', len(kp))
    print ('descriptor len = ', len(des))
    """
    
    color = (212, 68, 170)
    my_img = my_sift.draw_keypoints(img, my_color=color)
    #show_image('my_img', my_img)
    save_image(result_filename, my_img)
    

if __name__ == '__main__':
  main()
