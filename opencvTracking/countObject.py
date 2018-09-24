from yoloOpencv import opencvYOLO
from objectTracking import trackingObj
import cv2
import imutils

yolo = opencvYOLO(modeltype="yolov3", objnames="../YOLOVIDEO_TW/cfg.pepper/obj.names", 
    weights="../YOLOVIDEO_TW/cfg.pepper/weights/yolov3_40000.weights", 
    cfg="../YOLOVIDEO_TW/cfg.pepper/yolov3.cfg")

videowidth = 1920
videoheight = 1080
tracking = "MEDIANFLOW"

moveDirection = 3  #0: from top to bottom, 1: from bottom to top, 2: from right to left  3: from left to right

#-----------------------------------------------------------
FILE_OUTPUT = '/media/sf_ShareFolder/p_count_15.avi'

VIDEO_IN = cv2.VideoCapture("/media/sf_ShareFolder/pepper/p_6.m4v")

def readFrame(video):
    hasFrame, frame = video.read()
        # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        cv2.waitKey(3000)
        sys.exit()
    else:
        return (hasFrame, frame)

def printCount(frame, counts):
    cv2.putText(frame, str(counts) + " peppers", (30, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3)

if __name__ == "__main__":

    # Get current width of frame
    width = VIDEO_IN.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    # Get current height of frame
    height = VIDEO_IN.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 30.0, (int(videowidth),int(videoheight)))


    frameID = 0
    countTotal = 0
    while True:
        (hasFrame, frame) = readFrame(VIDEO_IN)
        #Object detect
        yolo.getObject(frame, labelWant=("2_pepper_matured"), drawBox=True)
        printCount(frame, countTotal)

        cv2.imshow("Frame", imutils.resize(frame, width=850))
        out.write(frame)
        cv2.waitKey(1)

        if(yolo.objCounts>0):
            bboxes = yolo.bbox
            tracker = trackingObj()
            '''
            hotBoxes = []
            for hotBox in bboxes:
                centerX = int(hotBox[0] + (hotBox[2]/2))
                centerY = int(hotBox[1] + (hotBox[3]/2))

                if(moveDirection==2):
                    if(centerX<=(videowidth/2)):
                        hotBoxes.append(hotBox)
                elif(moveDirection==3):
                    if(centerX>(videowidth/2)):
                        hotBoxes.append(hotBox)
            '''
            if(len(bboxes)>0):
                tracker.createTrackers(frame, bboxes, bold=3, color=(0,0,255), trackerType=tracking)
                success = True
            else:
                success = False

            while success:
                (hasFrame, frame) = readFrame(VIDEO_IN)
                (success, boxes, frame2) = tracker.updateTrackers(frame)
                print(success)

                for id, newbox in enumerate(boxes):
                    centerX = int(newbox[0] + (newbox[2]/2))
                    centerY = int(newbox[1] + (newbox[3]/2))

                    if(moveDirection==2):
                        countLine_x = int(videowidth/3)
                        if(centerX<=countLine_x):
                            if(tracker.counted[id] == False):
                                countTotal += 1
                                tracker.counted[id] = True
                                tracker.bold = 1
                    elif(moveDirection==3): 
                        countLine_x = videowidth - int(videowidth/3)
                        if(centerX>countLine_x):
                            if(tracker.counted[id] == False):
                                countTotal += 1
                                tracker.counted[id] = True
                                tracker.bold = 1

                print("Total:",countTotal, "Line:", countLine_x)
                printCount(frame, countTotal)
                #cv2.putText(frame, str(countTotal) + " peppers", (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1.6, (0,255,0), 3)

                cv2.imshow("Frame", imutils.resize(frame2, width=850))
                out.write(frame2)
                cv2.waitKey(1)
                #(hasFrame, frame) = readFrame(VIDEO_IN)

