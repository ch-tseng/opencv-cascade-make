import os, glob
import cv2
import dlib
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils

#-----------------------------------------------------------
minFaceSize = (90, 90)
imgOutputType = "jpg"
sourceFolder = "dataset/peoples"
outputFaces = "dataset/faces"
outputFaces_aligned = "dataset/faces_aligned"
faceDetectType = "cascade"   #cascade or dlib
#for Cascade type
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_scale = 1.2
cascade_neighbors = 10

dlib_detectorRatio = 2

folderCharacter = "/"  # \\ is for windows
#----------------------------------------------------------------

def getFaces_dlib(img, folder, imgname):
    global dlib_detectorRatio

    detector = dlib.get_frontal_face_detector()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector( gray , dlib_detectorRatio)

    i = 0
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            face = img[y:y+h, x:x+w]
            imgfile = folder + folderCharacter + imgname + "_" + str(i) + "." + imgOutputType
            cv2.imwrite(imgfile , face)
            i += 1


def getFaces_cascade(img, folder, imgname):
    global minFaceSize, cascade_scale, cascade_neighbors

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
            face = img[y:y+h, x:x+w]
            imgfile = folder + folderCharacter + imgname + "_" + str(i) + "." + imgOutputType
            cv2.imwrite(imgfile , face)
            i += 1

#=====================================================

if not os.path.exists(outputFaces):
    os.makedirs(outputFaces)

if not os.path.exists(outputFaces_aligned):
    os.makedirs(outputFaces_aligned)


for folders in glob.glob(sourceFolder+folderCharacter + "*"):
    print("Load {} ...".format(folders))
    label = os.path.basename(folders)

    for filename in os.listdir(folders):  
        if label is not None:
            image_name, image_extension = os.path.splitext(filename)
            image_extension = image_extension.lower()

            if(image_extension == ".jpg" or image_extension==".jpeg" or image_extension==".png" or image_extension==".bmp"):
                image = cv2.imread(folders + folderCharacter + filename)

                outputFolder = outputFaces + folderCharacter + label
                if not os.path.exists(outputFolder):
                    os.makedirs(outputFolder)

                if(faceDetectType=="cascade"):
                    getFaces_cascade(image, outputFolder, image_name)
                else:
                    getFaces_dlib(image, outputFolder, image_name)

