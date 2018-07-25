import os, glob
import cv2
import dlib
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils

faceWidth = 120
imgFileType = "jpg"
peopleFolder = "/home/chtseng/works/face-align/peoples"
outputFaceFolder = "/home/chtseng/works/face-align/faces"
faceLandmarkModel = "shape_predictor_68_face_landmarks.dat"

#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(faceLandmarkModel)
#fa = FaceAligner(predictor, desiredFaceWidth=faceWidth)

def load_images_from_folder(folder, outputFolder):
    global faceLandmarkModel, faceWidth

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(faceLandmarkModel)
    fa = FaceAligner(predictor, desiredFaceWidth=faceWidth)
    labels = []
    images = []

    for folders in glob.glob(folder+"/*"):
        label = os.path.basename(folders)
        print("Load {} ...".format(label))

        if(not os.path.exists(outputFolder + "/" + label)):
            os.mkdir(outputFolder + "/" + label)

        for filename in os.listdir(folders):  
            if label is not None:

                jpgname, file_extension = os.path.splitext(filename)
                if(file_extension.lower() == "." + imgFileType):
                    print("read file: ", os.path.join(folder,folders,filename))
                    img = cv2.imread(os.path.join(folder,folders,filename))

                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        rects = detector(gray, 2)
                        i = 0
                        # loop over the face detections
                        print("find {} faces".format(len(rects)))

                        for rect in rects:
                            # extract the ROI of the *original* face, then align the face
                            # using facial landmarks
                            #(x, y, w, h) = rect_to_bb(rect)
                            #faceOrig = image[y:y + h, x:x + w]
                            faceAligned = fa.align(img, gray, rect)

                            gray2 = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
                            rectB = detector( gray2 , 2)

                            for rectFinal in rectB:
                                (x2, y2, w2, h2) = rect_to_bb(rectFinal)
                                face2 = faceAligned[y2:y2 + h2, x2:x2 + w2]
 
                                #jpgname, file_extension = os.path.splitext(os.path.join(folder,folders,filename))
                                print("write face to ", outputFolder + "/" + label + "/" + jpgname + "-" + str(i) + ".jpg")
                                cv2.imwrite(outputFolder + "/" + label + "/" + jpgname + "-" + str(i) + ".jpg", face2)

                                i += 1



load_images_from_folder(peopleFolder, outputFaceFolder)
