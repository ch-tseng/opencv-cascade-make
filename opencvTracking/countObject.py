from yoloOpencv import opencvYOLO
from objectTracking import trackingObj
import cv2
import imutils

yolo = opencvYOLO(modeltype="yolov3", objnames="../YOLOVIDEO_TW/cfg.pepper/obj.names", 
    weights="../YOLOVIDEO_TW/cfg.pepper/weights/yolov3_40000.weights", 
    cfg="../YOLOVIDEO_TW/cfg.pepper/yolov3.cfg")

videowidth = 1280
vdeoHeigh = 720
countLine_x = 320

#-----------------------------------------------------------
FILE_OUTPUT = '/media/sf_ShareFolder/count.avi'

cap = cv2.VideoCapture("/media/sf_VMshare/countPepper.mp4")

def readFrame(video):
    hasFrame, frame = video.read()
        # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        cv2.waitKey(3000)
        sys.exit()
    else:
        return (hasFrame, frame)

if __name__ == "__main__":

    VIDEO_IN = cv2.VideoCapture("/media/sf_ShareFolder/pepper/p_1.mov")
    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width),int(height)))


    frameID = 0
    countTotal = 0
    while True:
        (hasFrame, frame) = readFrame(VIDEO_IN)
        #Object detect
        yolo.getObject(frame, labelWant=("2_pepper_matured"), drawBox=False)
        print ("Object counts:", yolo.objCounts)

        if(yolo.objCounts>0):
            bboxes = yolo.bbox
            tracker = trackingObj()
            tracker.createTrackers(frame, bboxes, trackerType="MEDIANFLOW")

            success = True
            while success:
                (success, boxes, frame2) = tracker.updateTrackers(frame)
                print(success)

                for id, newbox in enumerate(boxes):
                    centerX = int(newbox[0] + (newbox[2]/2))
                    centerY = int(newbox[1] + (newbox[3]/2))

                    if(centerX<countLine_x):
                        if(tracker.counted[id] == False):
                            countTotal += 1
                            tracker.counted[id] = True

                print("Total:",countTotal)
                cv2.imshow("Frame", imutils.resize(frame2, width=850))
                cv2.waitKey(1)
                (hasFrame, frame) = readFrame(VIDEO_IN)


