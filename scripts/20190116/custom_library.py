import math
import numpy as np
import cv2

### COMMON COLOR ###
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

def get_x_sign(x):
    if x == 0:
        return 0
    return (x / abs(x))

def lightness_sobel(img, thresh=50, maxval=255):
    '''Return the mask of the image after thresholding lightness channel
    Parameter
    ---------
    img : numpy.ndarray
        The image in BGR format
    thresh : int
        The 'thresh' parameter used for cv2.threshold() method
    maxval : int
        The 'maxval' parameter used for cv2.threshold() method
    '''
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]

    # Sobel both x and y directions of lightness channel
    sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
    abs_sobel = np.abs(sobel)  # absolute all negative gradient values

    l_ret, l_thresh = cv2.threshold(abs_sobel, 50, 255, cv2.THRESH_BINARY)
    return l_thresh


def binary_HSV(img, minThreshold=(0, 0, 180), maxThreshold=(179, 30, 255)):
    '''Return the mask of lane
    Parameter
    ---------
    img : numpy.ndarray
        The image in BGR format
    minThreshold : tuple
        Parameter for 'lowerb' in cv2.inRange() method (HSV format)
    maxThreshold : tuple
        Parameter for 'upperb' in cv2.inRange() method (HSV format)
    '''
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, minThreshold, maxThreshold)
    return mask


def shadow_HSV(img):
    minShadowTh = (30, 43, 36)
    maxShadowTh = (120, 81, 171)

    minLaneInShadow = (90, 43, 97)
    maxLaneInShadow = (120, 80, 171)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shadowMaskTh = cv2.inRange(imgHSV, minShadowTh, maxShadowTh)
    shadow = cv2.bitwise_and(img, img, mask=shadowMaskTh)
    shadowHSV = cv2.cvtColor(shadow, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(shadowHSV, minLaneInShadow, maxLaneInShadow)
    return mask


def get_center_line(leftx, rightx, ys):
    '''Return a line (x = ay + b)
    Parameter
    ---------
    leftx : list
        An array of x coordinate positions correspond to ys of the left line
    rightx : list
        An array of x coordinate positions correspond to ys of the right line
    ys : list
    '''
    xs = np.mean(np.vstack((leftx, rightx)), axis=0)
    line = np.polyfit(ys, xs, 1)
    return line, xs, ys


def radian_to_degree(x):
    return (x/math.pi) * 180


class Bird_View:
    def __init__(self, ratio=0.3, v_crop_ratio=0.55):
        self.ratio = ratio
        self.v_crop_ratio = v_crop_ratio

    def get_bird_view(self, img, dsize=(480, 320)):
        height, width = img.shape[:2]
        SKYLINE = int(height*self.v_crop_ratio)

        roi = img.copy()
        cv2.rectangle(roi, (0, 0), (width, SKYLINE), 0, -1)

        dst_width, dst_height = dsize

        src_pts = np.float32([
            [0, SKYLINE],
            [width, SKYLINE],
            [0, height],
            [width, height]
        ])
        dst_pts = np.float32([
            [0, 0],
            [dst_width, 0],
            [dst_width*self.ratio, dst_height],
            [dst_width*(1-self.ratio), dst_height]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        dst = cv2.warpPerspective(roi, M, dsize)

        return dst

    def inv_bird_view(self, img, dsize=(320, 240)):
        height, width = img.shape[:2]

        dst_width, dst_height = dsize

        SKYLINE = int(dst_height*self.v_crop_ratio)

        src_pts = np.float32([[0, 0], [width, 0], [
                             width*self.ratio, height], [width*(1-self.ratio), height]])
        dst_pts = np.float32([[0, SKYLINE], [dst_width, SKYLINE], [
                             0, dst_height], [dst_width, dst_height]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        dst = cv2.warpPerspective(img, M, dsize)

        return dst

    def draw_lines(self, src_img, plot_dsize, leftx_pts, rightx_pts, y_pts, color=(255, 0, 0)):

        color_image = np.zeros((plot_dsize[0], plot_dsize[1], 3))

        left = np.array(
            [np.flipud(np.transpose(np.vstack([leftx_pts, y_pts])))])
        right = np.array([np.transpose(np.vstack([rightx_pts, y_pts]))])
        points = np.hstack((left, right))

        cv2.fillPoly(color_image, np.int_(points), color)
        inv = self.inv_bird_view(color_image)
        retval = cv2.addWeighted(src_img.astype(
            np.uint8), 1, inv.astype(np.uint8), 1, 1)

        return retval


class Sliding_Windows:
    def __init__(self):
        self.left_a, self.left_b = [], []
        self.right_a, self.right_b = [], []

    def get_hist(self, img):
        hist = np.sum(img, axis=0)
        return hist

    def sliding_window(self, img, nwindows=16, margin=20, minpix=1, draw_windows=True, left_color=(0, 0, 255), right_color=(0, 255, 0), thickness=1):
        left_fit_ = np.empty(2)
        right_fit_ = np.empty(2)
        out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        s_hist = self.get_hist(img)
        # find peaks of left and right halves
        midpoint = int(s_hist.shape[0]/2)
        leftx_base = np.argmax(s_hist[:midpoint])
        rightx_base = np.argmax(s_hist[midpoint:]) + midpoint

        # set the height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # step through the windows one by one
        for window in range(nwindows):
            # identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # draw the windows on the visualization image
            if draw_windows == True:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), left_color, thickness)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), right_color, thickness)

            # identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # concatenate the arrays of indices ???
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) == 0 or len(lefty) == 0:
            left_fit = np.array([0, 0])
        else:
            left_fit = np.polyfit(lefty, leftx, 1)

        if len(rightx) == 0 or len(righty) == 0:
            right_fit = np.array([0, 0])
        else:
            right_fit = np.polyfit(righty, rightx, 1)

        self.left_a.append(left_fit[0])
        self.left_b.append(left_fit[1])

        self.right_a.append(right_fit[0])
        self.right_b.append(right_fit[1])

        left_fit_[0] = np.mean(self.left_a[-10:])  # ???
        left_fit_[1] = np.mean(self.left_b[-10:])  # ???

        right_fit_[0] = np.mean(self.right_a[-10:])
        right_fit_[1] = np.mean(self.right_b[-10:])

        # generate x and y values for plotting (x = ay + b => y = (x - b) / a)
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

        left_fitx = left_fit_[0]*ploty + left_fit_[1]
        right_fitx = right_fit_[0] * ploty + right_fit_[1]

        return out_img.astype(np.uint8), (left_fitx, right_fitx), (left_fit_, right_fit_), ploty
