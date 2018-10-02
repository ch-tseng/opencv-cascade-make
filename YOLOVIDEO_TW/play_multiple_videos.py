import cv2
import imutils
import numpy as np

FILE_OUTPUT = '/home/digits/Videos/combine_movie1.avi'

width = 2400
height = 800

video1 = "/home/digits/Videos/door_in_1_cascade.avi"
video2 = "/home/digits/Videos/door_in_1_dlib.avi"
video3 = "/home/digits/Videos/door_in_1_Yolo.avi"
caption1 = "Cascade"
caption2 = "Dlib"
caption3 = "YOLO"

cap1 = cv2.VideoCapture(video1)
cap2 = cv2.VideoCapture(video2)
cap3 = cv2.VideoCapture(video3)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 30.0, (width, height))

while True:
    hasFrame1, frame1 = cap1.read()
    hasFrame2, frame2 = cap2.read()
    hasFrame3, frame3 = cap3.read()

    if(hasFrame1 and hasFrame2 and hasFrame3):
        frame1 = imutils.resize(frame1, width=800, height=800)
        frame2 = imutils.resize(frame2, width=800, height=800)
        frame3 = imutils.resize(frame3, width=800, height=800)

        frame = np.hstack((frame1, frame2, frame3))
        cv2.imshow("Frame", frame)
        out.write(frame)
        cv2.waitKey(1)
