import os, glob
import cv2
import dlib
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from align_dlib import AlignDlib
import imutils

#-----------------------------------------------------------
minFaceSize = (25, 25)
sourceFolder = "image_test"
faceDetectType = "cascade"   #cascade or dlib
dlib_detectorRatio = 2
#for Cascade type
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_scale = 1.1
cascade_neighbors = 8

folderCharacter = "/"  # \\ is for windows

#----------------------------------------------------------------
detector = dlib.get_frontal_face_detector()

def getFaces_dlib(img):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector( gray , dlib_detectorRatio)

    i = 0
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            i += 1
    print("total {} faces".format(i))
    return img

def getFaces_cascade(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= cascade_scale,
        minNeighbors=cascade_neighbors,
        minSize=minFaceSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    i = 0
    for (x,y,w,h) in faces:
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            i += 1

    print("total {} faces".format(i))
    return img

for filename in os.listdir(sourceFolder):  
    image_name, image_extension = os.path.splitext(filename)
    image_extension = image_extension.lower()

    if(image_extension == ".jpg" or image_extension==".jpeg" or image_extension==".png" or image_extension==".bmp"):
        image = cv2.imread(sourceFolder + folderCharacter + filename)

        if(faceDetectType=="dlib"):
            faceimg = getFaces_dlib(image)
        else:
            faceimg = getFaces_cascade(image)

        faceimg = imutils.resize(faceimg, width=800)
        cv2.imshow("faces", faceimg)
        cv2.waitKey(0)
