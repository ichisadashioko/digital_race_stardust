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

from custom_library import *
from traffic_sign import Traffic_Sign

class Lane_Detect:

    def __init__(self):
        self.ST = False  # software testing
        # self.CODE = 'Team1' if self.ST else 'Team614'
        self.CODE = 'Team1'
        self.NAME = 'Stardust'

        self.counter = 0  # counter for reducing the number of processed image
        self.fixed_angle = 40
        self.fixed_speed = 40

        self.swsObject = Sliding_Windows()
        self.bvObject = Bird_View()
        self.tf_detector = Traffic_Sign()
        self.turn_start = 0
        self.turn_duration = 4 # 4 seconds

        self.sign_list = []

        self.subscriber = rospy.Subscriber("/{}_image/compressed".format(self.CODE), CompressedImage, self.callback, queue_size=1)
        self.steer_angle = rospy.Publisher('/{}_steerAngle'.format(self.CODE), Float32, queue_size=1)
        self.speed_pub = rospy.Publisher('/{}_speed'.format(self.CODE), Float32, queue_size=1)

    def pipeline(self,img):
        
        cv2.imshow('frame', img)

        l_thresh = lightness_sobel(img)

        ratio = 0.3  # shrink the bottom by 30%
        bird_view = self.bvObject.get_bird_view(l_thresh).astype(np.uint8)

        s_window, (left_fitx, right_fitx), (left_fit_,right_fit_), ploty = self.swsObject.sliding_window(bird_view)
        cv2.imshow('sliding windows', s_window)

        center_line, center_xs, center_ys = get_center_line(left_fitx, right_fitx, ploty)

        
        left_a, left_b = left_fit_
        right_a, right_b = right_fit_

        ## CONTROL CAR WITH TRAFFIC SIGN
        sign = self.tf_detector.get_traffic_sign(img)[0]
        
        if sign != 0:
            self.sign_list.append(sign)

        cur_sign = np.sum(self.sign_list) # current sign
        
        if (left_a == 0 and left_b == 0) or (right_a == 0 and right_b == 0):
            if abs(cur_sign) != 0:
                if int(self.turn_start) == 0:# and cur_sign != 0:
                    self.turn_start = time.time()
            elif cur_sign == 0: ## CONTROL CAR WITH CURVED LANE
                print("Curved lane")
                pass

        if self.turn_start != 0:
            # print("TURNING")
            self.steer_angle.publish(self.fixed_angle * get_x_sign(cur_sign))
            self.speed_pub.publish(self.fixed_speed)

            delta_timer = time.time() - self.turn_start
            print('delta_timer',delta_timer)
            if delta_timer > self.turn_duration:
                self.turn_start = 0
                self.sign_list = []
            return

        ## CONTROL CAR AT SINGLE LANE
        if self.ST:
            lines_plot_img = np.zeros_like(s_window, dtype=np.uint8)
            cv2.line(lines_plot_img, (240, 0), (240, 320), color=RED, thickness=1) # car line
            cv2.line(lines_plot_img, (int(center_xs[0]), 0), (int(center_xs[len(center_xs)-1]), 320), color=GREEN, thickness=1) # center line
            cv2.line(lines_plot_img, (int(left_fitx[0]), 0), (int(left_fitx[len(left_fitx)-1]), 320), color=BLUE, thickness=1)
            cv2.line(lines_plot_img, (int(right_fitx[0]), 0), (int(right_fitx[len(right_fitx)-1]), 320), color=BLUE, thickness=1)
            cv2.imshow('lines', lines_plot_img)

        if self.ST:
            draw_lane = self.bvObject.draw_lines(img, bird_view.shape[:2], left_fitx, right_fitx, ploty)
            cv2.imshow('draw_lane', draw_lane)

        roi_idx = int(320*2/3)
        delta = 240 - center_xs[roi_idx]


        steer_delta = -delta / 4
        self.steer_angle.publish(steer_delta)
        self.speed_pub.publish(self.fixed_speed)
        cv2.waitKey(1)

    def callback(self, ros_data):
        
        # if self.ST:
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