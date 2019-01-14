#!/usr/bin/env python
from __future__ import print_function
import sys
import time

import cv2
import numpy as np
import math

import rospy
import roslib

from sensor_msgs.msg import CompressedImage
import std_msgs
from std_msgs.msg import Float32


class Lane_Detect:

    def __init__(self):
        self.ST = True  # software testing
        self.CODE = 'Team1' if self.ST else 'Team614'
        self.NAME = 'Stardust'

        self.counter = 0  # counter for reducing the number of processed image
        # x = ay + b
        self.left_a, self.left_b = [], []
        self.right_a, self.right_b = [], []
        self.angle_fix = 0
        self.wait = 0

        # self.subscriber = rospy.Subscriber('/unity_image/compressed', CompressedImage, self.callback, queue_size=1)
        # self.steer_angle = rospy.Publisher('steerAngle', Float32, queue_size=1)
        # self.speed_pub = rospy.Publisher('speed', Float32, queue_size=1)

        self.subscriber = rospy.Subscriber("/{}_image/compressed".format(self.CODE), CompressedImage, self.callback, queue_size=1)
        self.steer_angle = rospy.Publisher('{}_steerAngle'.format(self.CODE), Float32, queue_size=1)
        self.speed_pub = rospy.Publisher('{}_speed'.format(self.CODE), Float32, queue_size=1)

    def get_hist(self, img):
        hist = np.sum(img, axis=0)
        return hist

    def get_bird_view(self, img, shrink_ratio, dsize=(480, 320)):
        height, width = img.shape[:2]
        SKYLINE = int(height*0.55)

        roi = img.copy()
        cv2.rectangle(roi, (0, 0), (width, SKYLINE), 0, -1)

        dst_width, dst_height = dsize

        src_pts = np.float32([[0, SKYLINE], [width, SKYLINE], [0, height], [width, height]])
        dst_pts = np.float32([[0, 0], [dst_width, 0], [dst_width*shrink_ratio, dst_height], [dst_width*(1-shrink_ratio), dst_height]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        dst = cv2.warpPerspective(roi, M, dsize)

        return dst

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        SKYLINE = int(height * 0.45)
        vertices = np.array(
            [[(width/2, SKYLINE), (width/2, 0), (width, 0), (width, SKYLINE)]], np.int32)
        mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def sign_detect(self, img):
        img = self.region_of_interest(img)
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(img_HSV, (100, 100, 0), (110, 255, 255))
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        hull = []
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        for i in range(len(contours)):
            color_contours = (0, 255, 0)
            color = (255, 0, 0)
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            cv2.drawContours(drawing, hull, i, color, 1, 8)
            if cv2.contourArea(contours[i]) > 70:
                x,y,w,h = cv2.boundingRect(contours[i])
                if w*h > 500 and np.abs(w-h) <= 10:
                    cv2.rectangle(drawing, (x,y),(x+w,y+h),(0, 0, 255),2)
                    crop = thresh[y:y+h, x:x+w]
                    self.cal_sign(crop)
        if self.angle_fix != 0:
            if self.wait > 0:
                self.wait = self.wait - 0.1
            elif self.angle_fix > 0:
                self.angle_fix = self.angle_fix - 0.75
                if (self.angle_fix < 0): self.angle_fix = 0
            else:
                self.angle_fix = self.angle_fix + 0.75
                if (self.angle_fix > 0): self.angle_fix = 0
        # cv2.imshow('drawing',drawing)
    
    # v = 35, w = 4.5, a_f = 36, delta = +-0.6
    # v = 40, w = 2.7, a_f = 28, delta = +-0.5
    # v = 50, w = 1.5, a_f = 28, delta = +-0.75
    def cal_sign(self, crop_img):
        x,y = crop_img.shape
        left = crop_img[int(0):y, 0:x/2]
        right = crop_img[int(0):y, x/2+1:x]
        count_left = cv2.countNonZero(left)
        count_right = cv2.countNonZero(right)
        balance = count_left-count_right
        print ('balance:', balance)
        if self.wait <= 0 and int(self.angle_fix) == 0:
            if balance > 20:
                self.angle_fix = 28
            else:
                self.angle_fix = -28
            self.wait = 2.7


    def inv_bird_view(self, img, stretch_ratio, dsize=(320, 240)):
        height, width = img.shape[:2]

        dst_width, dst_height = dsize

        SKYLINE = int(dst_height*0.55)

        src_pts = np.float32([[0, 0], [width, 0], [
                             width*stretch_ratio, height], [width*(1-stretch_ratio), height]])
        dst_pts = np.float32([[0, SKYLINE], [dst_width, SKYLINE], [
                             0, dst_height], [dst_width, dst_height]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        dst = cv2.warpPerspective(img, M, dsize)

        return dst

    def sliding_window(self, img, nwindows=16, margin=20, minpix=1, draw_windows=True, left_color=(0, 0, 255), right_color=(0, 255, 0), thickness=1):
        # global left_a, left_b, left_c, right_a, right_b, right_c
        left_fit_ = np.empty(2)
        right_fit_ = np.empty(2)
        # I haven't understood this line of code
        out_img = np.dstack((img, img, img))*255

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

        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

    def draw_lines(self, src_img, plot_dsize, leftx_pts, rightx_pts, ploty, color=(255, 0, 0)):
        stretch_ratio = 0.3

        color_image = np.zeros((plot_dsize[0], plot_dsize[1], 3))

        left = np.array(
            [np.flipud(np.transpose(np.vstack([leftx_pts, ploty])))])
        right = np.array([np.transpose(np.vstack([rightx_pts, ploty]))])
        points = np.hstack((left, right))

        cv2.fillPoly(color_image, np.int_(points), color)
        inv = self.inv_bird_view(color_image, stretch_ratio)
        retval = cv2.addWeighted(src_img.astype(
            np.uint8), 1, inv.astype(np.uint8), 1, 1)

        return retval

    def get_desired_line(self, leftx, rightx, ys):
        xs = np.mean(np.vstack((leftx, rightx)), axis=0)
        line = np.polyfit(ys, xs, 1)
        return line, xs, ys


    def radian_to_degree(self,x):
        return (x/math.pi) * 180

    def binary_HSV(self, img):
        minThreshold = (0, 0, 180)
        maxThreshold = (179, 30, 255)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        out = cv2.inRange(hsv_img, minThreshold, maxThreshold)
        return out

    def shadow_HSV(self, img):
        minShadowTh = (30, 43, 36)
        maxShadowTh = (120, 81, 171)

        minLaneInShadow = (90, 43, 97)
        maxLaneInShadow = (120, 80, 171)

        imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        shadowMaskTh = cv2.inRange(imgHSV, minShadowTh, maxShadowTh)
        shadow = cv2.bitwise_and(img, img, mask=shadowMaskTh)
        shadowHSV = cv2.cvtColor(shadow, cv2.COLOR_RGB2HSV)
        out = cv2.inRange(shadowHSV, minLaneInShadow, maxLaneInShadow)
        return out

    def calc_shadow(self, img, sobel_kernel=5, thresh=(40, 255), canny_thresh=(50,150)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
        mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
        retval, threshold = cv2.threshold(scaled_sobel, thresh[0], thresh[1], cv2.THRESH_BINARY)

        gausImage = cv2.GaussianBlur(gray, (sobel_kernel, sobel_kernel), 0)
        # Run the canny edge detection
        cannyImage = cv2.Canny(gausImage, canny_thresh[0], canny_thresh[1])
        out = np.bitwise_and(threshold, cannyImage)
        return out


    def pipeline(self,img):

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        # Sobel both x and y directions of lightness channel
        sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
        abs_sobel = np.abs(sobel)  # absolute all negative gradient values

        l_ret, l_thresh = cv2.threshold(abs_sobel, 50, 255, cv2.THRESH_BINARY)

        m1= self.binary_HSV(img)
        m2 = self.shadow_HSV(img)
        m3 = cv2.bitwise_or(m1.astype(np.uint8),m2.astype(np.uint8))
        m4 = cv2.bitwise_or(l_thresh.astype(np.uint8),m3.astype(np.uint8))
        m5 = self.calc_shadow(img)
        m6 = cv2.bitwise_and(m4, m5)

        ratio = 0.3  # shrink the bottom by 30%
        bird_view = self.get_bird_view(m6, ratio)

        s_window, (left_fitx, right_fitx), (left_fit_,right_fit_), ploty = self.sliding_window(bird_view)

        d_line, d_xs, d_ys = self.get_desired_line(left_fitx, right_fitx, ploty)

        lines_plot_img = np.zeros_like(s_window, dtype=np.uint8)
        cv2.line(lines_plot_img, (240, 0), (240, 320), color=(0, 0, 255), thickness=1)
        cv2.line(lines_plot_img, (int(d_xs[0]), 0), (int(d_xs[len(d_xs)-1]), 320), color=(0, 255, 0), thickness=1)

        draw_lane = self.draw_lines(img, bird_view.shape[:2], left_fitx, right_fitx, ploty)

        cv2.imshow('frame', img)
        cv2.imshow('sliding windows', s_window)
        cv2.imshow('lines', lines_plot_img)
        cv2.imshow('draw_lane', draw_lane)
        self.sign_detect(img)
        top_delta = 240 - d_xs[0]
        bot_delta = 240 - d_xs[len(d_xs) - 1]
        # steer_delta = - (top_delta - bot_delta) / 4
        roi_idx = int(320*2/3)
        delta = 240 - d_xs[roi_idx] - 10

        # steer_delta = math.atan(delta/(320-roi_idx))

        steer_delta = -delta / 4
        if self.wait <= 0 and self.angle_fix != 0: 
            steer_delta = self.angle_fix
        print('steer_delta:', steer_delta)
        print('angle_fix:', self.angle_fix)
        self.steer_angle.publish(steer_delta)
        self.speed_pub.publish(40)

        cv2.waitKey(2)

    def callback(self, ros_data):

        self.counter += 1
        if self.counter % 2 == 0:  # reduce half of images for processing
            self.counter = 0
            return

        # decode image
        np_arr = np.fromstring(ros_data.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0

        self.pipeline(img)

def shutdown_hook():
    print('Shutting down ROS lane detection module')

def main(args):
    lane_detect = Lane_Detect()
    rospy.init_node('team_614_lane_detect', anonymous=True)
    rospy.on_shutdown(shutdown_hook)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down ROS lane detection module')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
