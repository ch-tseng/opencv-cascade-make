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

maxPeopleInSeconds = 30  # secoonds, we will take the max number peoples as correct number in this time range.
peopleCount = []

face_size = (48, 48)
monitor_winSize = (1024, 768)
cam_resolution = (1024, 768)

face_cascade = cv2.CascadeClassifier('palm_20180721.xml')

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])

def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image

def peopleNumerList(numPeople = 0):
    global peopleCount, maxPeopleInSeconds

    if(len(peopleCount)>maxPeopleInSeconds):
        peopleCount.pop(0)

    peopleCount.append(numPeople)

while(camera.isOpened()):
    (grabbed, img) = camera.read()   

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= 1.3,
        minNeighbors=12,
        minSize=face_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    i = 0
    for (x,y,w,h) in faces:
	
        if( (w>face_size[0] and h>face_size[1]) ):
            #roi_color = img[y:y+h, x:x+w]
            #now=datetime.datetime.now()
            #faceName = '%s_%s_%s_%s_%s_%s_%s.jpg' % (now.year, now.month, now.day, now.hour, now.minute, now.second, i)
            #cv2.imwrite(savePath+"/" + faceName, roi_color)
            #print("w:{}, h:{}".format(w,h))
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)            
            i += 1

    r = monitor_winSize[1] / img.shape[1]
    dim = (monitor_winSize[0], int(img.shape[0] * r))
    img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Frame", img2)
    key = cv2.waitKey(1)
    if(key==113):  #q to exit
        break
