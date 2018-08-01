import cv2
import imutils
import numpy as np
import datetime

faceWidth = 2  # cm
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_scale = 1.2
cascade_neighbors = 10
minFaceSize = (30, 30)
evidenceSavePath = "picEvidence"
folderCharacter = "/"
imgOutputType = "jpg"
cam_id = 0
cam_resolution = (1024,768)

camera = cv2.VideoCapture(cam_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])

facePixel = 0

def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image

def getFaces_cascade(img, rtype):
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
            if(facePixel>0):
                print("Face distance:", (faceWidth * 1.0 / facePixel ) * (w-facePixel) )

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            i += 1

    if(i>0):
        if(rtype==1):
            return (w, h, img)

        else:
            imgname = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            putText(img, datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S"), 5, 40, (255,35,35), thickness=2, size=1)  
            imgfile = evidenceSavePath + folderCharacter + imgname + "." + imgOutputType
            cv2.imwrite(imgfile , img)
            return img

    else:
        if(rtype==1):
            return (0, 0, img)
        else:
            return img

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL) 

while True:

    (grabbed, image) = camera.read()  

    if(facePixel==0):
        (w, h, image2) = getFaces_cascade(image, 1)

        cv2.imshow("Frame", image2)

        if(w>0 and h>0):
            print("Please click o to set the distance.")
            keycode = cv2.waitKey(0) & 0xFF
            print("keycode=",keycode)

            if keycode == ord('o'):
                #scaleRatio = (w * 0.1) / faceWidth
                facePixel = w
                print("facePixel=", facePixel)
        else:
            cv2.waitKey(1)

    else:
        image2 = getFaces_cascade(image, 2)

        cv2.imshow("Frame", image2)
        cv2.waitKey(1)
