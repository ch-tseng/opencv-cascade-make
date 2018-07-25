import cv2
from align_dlib import AlignDlib
from imutils.face_utils import rect_to_bb
import dlib

faceWidth = 360
faceLandmarkModel = "shape_predictor_68_face_landmarks.dat"
testimgPath = "IMG_7492.jpg"

image = cv2.imread(testimgPath)
align_dlib = AlignDlib(faceLandmarkModel)
bbs = align_dlib.getAllFaceBoundingBoxes(image)
detector = dlib.get_frontal_face_detector()

i = 0
for bb in bbs:
    aligned = align_dlib.align(faceWidth, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP, scale=0.5)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        rectB = detector( gray , 2)
 
        for rectFinal in rectB:
            (x2, y2, w2, h2) = rect_to_bb(rectFinal)
            print((x2, y2, w2, h2))
            face2 = aligned[y2:y2 + h2, x2:x2 + w2]
            face2 = face2[...,::-1]

            cv2.imwrite("dlibface-" + str(i) + ".jpg", face2)
            i += 1
    #cv2.imwrite("dlibface-" + str(i) + ".jpg", aligned)
    #i += 1
