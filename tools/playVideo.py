#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, os
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from scipy import io
import numpy as np
import imutils
import cv2
import sys
import time
import datetime

videoFile = "/media/sf_shares/movie_cucumber_1.mp4"
output_folder = "/media/sf_ShareFolder/movie_cucumber"

camera = cv2.VideoCapture(videoFile)
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])

def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image

i = 0
while True:
    (grabbed, img) = camera.read()   

    print("Frame #", i)
    cv2.imwrite(output_folder + "/" + str(i) + ".jpg", img)

    i += 1
