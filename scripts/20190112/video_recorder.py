import os
import sys
import time
import math
import datetime
from datetime import datetime

import numpy as np
import cv2


class Video_Writer:
    def __init__(self, name=None, dimension=(320, 240)):
        self.recorder = self.get_video_writer(name, dimension)

    def get_timestamp(self):
        now = datetime.now()
        timestamp = '{}{}{}_{}{}{}'.format(str(now.year).zfill(4), str(now.month).zfill(2), str(
            now.day).zfill(2), str(now.hour).zfill(2), str(now.minute).zfill(2), str(now.second).zfill(2))
        return timestamp

    def get_video_writer(self, name=None, dimension=(320, 240)):
        usr = os.path.expanduser('~')  # return /home/<username>
        path = os.path.join(usr, 'Videos')  # return /home/<username>/Videos

        file_name = ""

        if name is None:
            timestamp = self.get_timestamp()  # format "YYYYMMDD_HHMMSS"
            # format "YYYYMMDD_HHMMSS.avi"
            file_name = os.path.join(path, timestamp + '.avi')
        else:
            file_name = os.path.join(path,  name + '.avi')

        fps = 40
        frame_size = dimension
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        video_writer = cv2.VideoWriter(file_name, fourcc, fps, frame_size)
        return video_writer

    def write(self, img):
        self.recorder.write(img)

    def release(self):
        self.recorder.release()
