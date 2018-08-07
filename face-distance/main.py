import cv2
import imutils
import numpy as np
import datetime

distanceActual = 0  # cm
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cascade_scale = 1.2
cascade_neighbors = 12
minFaceSize = (5, 5)
evidenceSavePath = "picEvidence"
folderCharacter = "/"
imgOutputType = "jpg"
cam_id = 0
cam_resolution = (1080,768)
outputVideo = True
FILE_OUTPUT = 'distance.avi'

#camera = cv2.VideoCapture(cam_id)
camera = cv2.VideoCapture("/media/sf_VMshare/walking-10.mp4")
camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])

facePixel = 0

# Get current width of frame
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width),int(height)))
up_middle_down = 0

def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image

def getFaces_cascade(img, rtype):
    global up_middle_down

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
                distance = round((W * F) / w, 0)
                #pixel_1 = (distanceActual * 1.0 / facePixel) * (w/(w-facePixel+1) )
                #move_cm = (w-facePixel) * pixel_1
                #distance = round(distanceActual - move_cm ,1)
                #print("1 pixel:{}, fixed pixel:{}, face now:{}, moved:{}, distance:{}".format(pixel_1, distanceActual, w, move_cm, distance) )

                if(up_middle_down==0):
                    y_loc = -60
                elif(up_middle_down==1):
                    y_loc = -30
                else:
                    y_loc = 0
                    up_middle_down = 0

                up_middle_down += 1

                putText(img, str(round(distance/100,1)) + "m", int(x+(w/3)), y+y_loc, color=(0,255,0), thickness=3, size=2.6)

            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.circle(img, (int(x+(w/2)), int(y+(h/2))), int(w/2), (255, 255, 255), 2)
            i += 1

    if(i>0):
        if(rtype==1):
            return (w, h, img)

        else:
            imgname = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            #putText(img, datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S"), 5, 40, (255,35,35), thickness=2, size=1)  
            imgfile = evidenceSavePath + folderCharacter + imgname + "." + imgOutputType
            #cv2.imwrite(imgfile , img)
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
            keycode = cv2.waitKey(3000) & 0xFF
            print("keycode=",keycode)

            if keycode == ord('o'):
                #scaleRatio = (w * 0.1) / distanceActual
                facePixel = w
                distanceActual = float(input('Your head width (cm): '))
                P = distanceActual # distance cm
                W = 22 # face width average 20cm
                D = w # facepixels
                F = (P * D) / W

                print("Focal length=", facePixel)
        else:
            cv2.waitKey(1)

    else:
        image2 = getFaces_cascade(image, 2)

        cv2.imshow("Frame", image2)

        if(outputVideo==True):
            out.write(image2)

        cv2.waitKey(1)
