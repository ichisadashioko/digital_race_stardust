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

        self.counter = 0 

        self.fixed_angle = 18 # angle for turning at traffic sign
        self.fixed_speed = 45 # speed for normal running speed
        self.fixed_turn_speed = 40 # speed for running at intersect
        self.cur_speed = self.fixed_speed # speed to be published at non-intersect road

        self.swsObject = Sliding_Windows()
        self.bvObject = Bird_View()
        self.tf_detector = Traffic_Sign()

        self.turn_start = 0 # hold the time when the car starts turning at intersect
        self.turn_duration = 1.6  # 4 seconds # how long the car will turn
        self.wait_start = 0 # hold the time when recognize traffic sign
        self.wait_duration = 0.9  # how long the car will wait before turning

        self.sign_list = []
        self.loc_fix = []
        self.pos_fix = 125

        self.subscriber = rospy.Subscriber(
            "/{}_image/compressed".format(self.CODE), CompressedImage, self.callback, queue_size=1)
        self.steer_angle = rospy.Publisher(
            '/{}_steerAngle'.format(self.CODE), Float32, queue_size=1)
        self.speed_pub = rospy.Publisher(
            '/{}_speed'.format(self.CODE), Float32, queue_size=1)

    def pipeline(self, img):

        cv2.imshow('frame', img)

        l_thresh = lightness_sobel(img)

        ratio = 0.3  # shrink the bottom by 30%
        bird_view = self.bvObject.get_bird_view(l_thresh).astype(np.uint8)

        s_window, (left_fitx, right_fitx), (left_fit_,
                                            right_fit_), ploty = self.swsObject.sliding_window(bird_view)
        cv2.imshow('sliding windows', s_window)

        center_line, center_xs, center_ys = get_center_line(
            left_fitx, right_fitx, ploty)

        left_a, left_b = left_fit_
        right_a, right_b = right_fit_

        # --------------------------------------------
        roi_idx = int(320*2/3)
        center_x = center_xs[roi_idx]
        # left_x = np.round(roi_idx*left_a + left_b)
        left_x = left_fitx[roi_idx]
        # right_x = np.round(roi_idx*right_a + right_b)
        right_x = right_fitx[roi_idx]
        locx = -left_x + center_x
        locy = right_x - center_x
        if abs(locx + locy - 2*self.pos_fix) <= 10:
            if len(self.loc_fix) < 30:
                self.loc_fix.append(locx)
                self.loc_fix.append(locy)
            if len(self.loc_fix) >= 30:
                self.loc_fix.pop(0)
                self.loc_fix.pop(0)
        print(self.loc_fix)
        self.pos_fix = np.average(self.loc_fix)
        # --------------------------------------------
        # CONTROL CAR WITH TRAFFIC SIGN
        sign = self.tf_detector.get_traffic_sign(img)

        if sign != 0:
            self.sign_list.append(sign)

        cur_sign = np.sum(self.sign_list)  # current sign

        if cur_sign != 0: # There is a traffic sign here!
            if self.wait_start == 0:
                self.wait_start = time.time()
                self.cur_speed = self.fixed_turn_speed # reduce the car speed
                print("Waiting")

            elif (time.time() - self.wait_start) > self.wait_duration and self.turn_start == 0:
                # Waiting is done. Start turning
                self.turn_start = time.time() # 
                print("Turning")

        if self.turn_start != 0:
            a = self.fixed_angle * get_x_sign(cur_sign)
            print(a)
            self.steer_angle.publish(a)
            self.speed_pub.publish(self.fixed_turn_speed)

            delta_timer = time.time() - self.turn_start
            if delta_timer > self.turn_duration:
                # Turning is done
                # RESET VARIABLES
                self.turn_start = 0
                self.sign_list = []
                self.cur_speed = self.fixed_speed
                self.wait_start = 0
            return

        roi_idx = int(320*2/3)
        delta = 240 - center_xs[roi_idx]
        delta = 240 - left_x - self.pos_fix
        print(self.pos_fix)
        steer_delta = -delta / 3.7
        self.steer_angle.publish(steer_delta)
        self.speed_pub.publish(self.cur_speed)

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
