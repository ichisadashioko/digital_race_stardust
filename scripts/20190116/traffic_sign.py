import cv2
import numpy as np


class Traffic_Sign:
    def __init__(self):
        # label of traffic sign detected
        self.sign = 0
        self.labels = ['right', 'left']
        self.min_area = 400

        # range of traffic_sign background
        self.lowerBound = np.array([98, 109, 20])
        self.upperBound = np.array([112, 255, 255])

        self.area_left = 0
        self.area_right = 0
        self.area_left_before = 0
        self.area_right_before = 0
    
    # find contours of traffic sign
    def find_contour(self, image):
        kernelOpen = np.ones((5, 5))
        kernelClose = np.ones((20, 20))

        # convert BGR to HSV
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # create the Mask
        mask = cv2.inRange(imgHSV, self.lowerBound, self.upperBound)

        # morphology to remove noise
        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        maskFinal = maskClose
        la, conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return conts

    def get_traffic_sign(self, image):
        image = image.copy()
        # the upper right conner of image
        height, width = image.shape[:2]
        SKYLINE = int(height * 0.45)
        roi = image[0:SKYLINE, 0:width]
        # roi = self.region_of_interest(image)

        # contours of the roi
        conts = self.find_contour(roi)

        if len(conts) == 0:  # no_sign
            # self.area of the rois contain two upper half of traffic sign
            self.area_left = 0
            self.area_right = 0
            self.area_left_before = 0
            self.area_right_before = 0
            self.sign = 0

        for i in range(len(conts)):
            M = cv2.moments(conts[i])

            # (cX,xY) is a center point of bounding box
            cX = int(M['m10']/M['m00'])
            cY = int(M['m01']/M['m00'])

            x, y, w, h = cv2.boundingRect(conts[i])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # (cX,cY) is a center point of 1/2 bounding box.
            if (cY - y) % 2 == 0:
                cY = int(y + 0.5 * (cY - y))
            else:
                cY = int(y + 0.5*(cY - 1 - y))

            # remove small independent objects
            if (cv2.contourArea(conts[i]) > self.min_area):
                img1 = image[y:cY, x:cX]
                conts1 = self.find_contour(img1)

                # the upper left conner of Rectangle
                img2 = image[y:cY, cX:x+w]
                conts2 = self.find_contour(img2)
                if len(conts1) != 0 and len(conts2) != 0:
                    # self.area each contour of each part
                    self.area_left = self.area_left + \
                        cv2.contourArea(conts1[i])
                    self.area_right = self.area_right + \
                        cv2.contourArea(conts2[i])

                    if self.area_left > self.area_right and self.area_left > self.area_left_before:  # right
                        self.sign = 1
                        self.area_left_before = self.area_left
                    elif self.area_left < self.area_right and self.area_right > self.area_right_before:  # left
                        self.sign = -1
                        self.area_right_before = self.area_right
                        
        return self.sign#, image