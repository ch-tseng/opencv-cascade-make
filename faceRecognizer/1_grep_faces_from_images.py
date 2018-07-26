import os, glob
import cv2
import dlib
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from align_dlib import AlignDlib
import imutils

#-----------------------------------------------------------
minFaceSize = (90, 90)
imgOutputType = "jpg"
sourceFolder = "dataset/peoples"
outputFaces = "dataset/faces"
outputFaces_aligned = "dataset/faces_aligned"
faceDetectType = "dlib"   #cascade or dlib
#for Cascade type
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_scale = 1.2
cascade_neighbors = 10

alignType = "imutils"  #dlib or imutils
face_outputsize_aligned = 90  #size for aligned face
dlib_detectorRatio = 2
faceLandmarkModel = "shape_predictor_68_face_landmarks.dat"

folderCharacter = "/"  # \\ is for windows
#----------------------------------------------------------------
detector = dlib.get_frontal_face_detector()
align_dlib = AlignDlib(faceLandmarkModel)
predictor = dlib.shape_predictor(faceLandmarkModel)
fa = FaceAligner(predictor, desiredFaceWidth=face_outputsize_aligned)

def alignFace_imutils(image, imgfile):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    for rect in rects:
        faceAligned = fa.align(image, gray, rect)

        gray2 = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
        rectB = detector( gray2 , 2)

        i = 0 
        for rectFinal in rectB:
            (x2, y2, w2, h2) = rect_to_bb(rectFinal)
            if(w2>minFaceSize[0] and h2>minFaceSize[1]):
                face2 = faceAligned[y2:y2 + h2, x2:x2 + w2]
                #face2 = face2[...,::-1]
                print("aligned:", imgfile)
                cv2.imwrite(imgfile , face2)
                i += 1

def alignFace_dlib(image, imgfile):
    bbs = align_dlib.getAllFaceBoundingBoxes(image)
    i = 0
    for bb in bbs:
        aligned = align_dlib.align(face_outputsize_aligned, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP, scale=0.5)
        if aligned is not None:
            aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY)
            rectB = detector( gray , 2)

            i = 0 
            for rectFinal in rectB:
                (x2, y2, w2, h2) = rect_to_bb(rectFinal)
                if(w2>minFaceSize[0] and h2>minFaceSize[1]):
                    face2 = aligned[y2:y2 + h2, x2:x2 + w2]
                    #face2 = face2[...,::-1]
                    print("aligned:", imgfile)
                    cv2.imwrite(imgfile , face2)
                    i += 1

def getFaces_dlib(img, label, imgname):
    global dlib_detectorRatio

    detector = dlib.get_frontal_face_detector()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector( gray , dlib_detectorRatio)

    i = 0
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        if(w>minFaceSize[0] and h>minFaceSize[1]):
            face = img[y:y+h, x:x+w]
            imgfile = outputFaces + folderCharacter + label + folderCharacter + imgname + "_" + str(i) + "." + imgOutputType
            cv2.imwrite(imgfile , face)
            imgfile = outputFaces_aligned + folderCharacter + label + folderCharacter + imgname + "_" + str(i) + "." + imgOutputType

            if(alignType=="dlib"):
                alignFace_dlib(img, imgfile)
            else:
                alignFace_imutils(img, imgfile)

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

            if(alignType=="dlib"):
                alignFace_dlib(img, imgfile)
            else:
                alignFace_imutils(img, imgfile)

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

                if not os.path.exists(outputFaces+folderCharacter+label):
                    os.makedirs(outputFaces+folderCharacter+label)

                if not os.path.exists(outputFaces_aligned+folderCharacter+label):
                    os.makedirs(outputFaces_aligned+folderCharacter+label)


                if(faceDetectType=="cascade"):
                    getFaces_cascade(image, label, image_name)
                else:
                    getFaces_dlib(image, label, image_name)

