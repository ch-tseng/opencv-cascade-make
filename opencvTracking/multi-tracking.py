import time, sys
import imutils
from pydarknet import Detector, Image
import cv2
from random import randint

FILE_OUTPUT = 'videos/t4.avi'
frames_tracking = 60
trackerType = "CSRT"
detectDirectionPeriod = 5

multiTracker = cv2.MultiTracker_create()
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
justDetected = True
justDetected_wait = 4

def getDirection(x_var, y_var, lastDir):
    direction = ""

    if(abs(x_var)<=1 and abs(y_var)<=1):
        direction = "stop_stop"

    elif(abs(x_var)<=3 and abs(y_var)<=3):
        direction = lastDir

    else:
        if( x_var >= 4 ):
            direction = direction + "right"
        elif( x_var <= -4):
            direction = direction + "left"
        else:
            direction = direction + "stop"

        if( y_var >= 4 ):
            direction = direction + "_down"
        elif( y_var <= -4):
            direction = direction + "_up"
        else:
            direction = direction + "_stop"

    print(x_var, y_var)
    return direction

def getSpeed(x_var, y_var):
    threshold_slow = 10
    threshold_normal = 40
    speed = ""
    fontcolor = (0, 255, 0)
    fontbold = 1

    if(abs(x_var)<threshold_slow or abs(y_var)<threshold_slow):
        speed ="slow"
        fontcolor = (0,255,0)
        fontbold = 1
    if((abs(x_var)>=threshold_slow and abs(x_var)<threshold_normal) or (abs(y_var)>=threshold_slow and abs(y_var)<threshold_normal)):
        speed = "normal"
        fontcolor = (0,255,0)
        fontbold = 1
    if(abs(x_var)>=threshold_normal or abs(y_var)>=threshold_normal):
        speed = "fast"
        fontcolor = (0,0,255)
        fontbold = 2
        print("FAST:", x_var, y_var)

    return (speed, fontcolor, fontbold) 

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
     
    return tracker

def transparentOverlay(src , overlay , pos=(0,0),scale = 1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src

def imgDirection(img, direction, posXY):
    dirImg = cv2.imread("images/"+direction+".png", cv2.IMREAD_UNCHANGED)
    #print("images/"+direction+".png")
    result = transparentOverlay(img, dirImg, posXY, 0.7)
    return result

if __name__ == "__main__":

    net = Detector(bytes("../../darknet/cfg/yolov3.cfg", encoding="utf-8"), 
                   bytes("yolov3.weights", encoding="utf-8"), 0,
                   bytes("coco.data", encoding="utf-8"))

    cap = cv2.VideoCapture("videos/t4.mp4")

    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width),int(height)))
    trackingIndex = 30
    i = 0
    bboxes = []
    colors = []
    counted = []
    objid = []
    lastY = []
    lastX = []
    dirCar = []
    lastDir = []
    objName = []
    speed= ""
    fontcolor = (0, 0, 0)
    fontbold = 1
    waitJustDetected = justDetected_wait + 1

    while True:
        r, frame = cap.read()
        if r:
            i += 1
            trackingIndex += 1
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            if(trackingIndex>=frames_tracking):
                trackingIndex = 31
                del multiTracker
                bboxes = []
                colors = []
                multiTracker = cv2.MultiTracker_create()

                dark_frame = Image(frame)
                results = net.detect(dark_frame)
                del dark_frame

                end_time = time.time()
                #print("{}: Elapsed Time:{}".format(i, end_time-start_time) )

                indexObj = 0
                for cat, score, bounds in results:
                    label = cat.decode('utf-8')
                    print(label)
                    #print("{}:{}".format(cat, score))

                    x, y, w, h = bounds
                    if(label in ("car", "bus", "motorbike", "truck") ):
                        id = str(time.time()) + "_" + str(indexObj)
                        objName.append(label)
                        bboxes.append((x-(w/2), y-(h/2), w, h))
                        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
                        objid.append(id)
                        counted.append(0)
                        #lastY.append(int(y + h/2))
                        #lastX.append(int(x + w/2))
                        lastY.append(0)
                        lastX.append(0)
                        lastDir.append("")
                        dirCar.append("stop_stop")

                        #cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,255,0), 3)
                        #print("add text: ",label)
                        #cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0,255,0))
                        indexObj += 1

                for bbox in bboxes:
                    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

                justDetected = True
                waitJustDetected = 0

        success, boxes = multiTracker.update(frame)

        for id, newbox in enumerate(boxes):
            if(justDetected==True):
                lastX[id] = int(newbox[0]+(newbox[2]/2))
                lastY[id] = int(newbox[1]+(newbox[3]/2))

            else:
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                #cv2.rectangle(frame, p1, p2, colors[id], 2, 1)
                #cv2.putText(frame, objid[id], (int(newbox[0]), int(newbox[1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, colors[id])

                counted[id] = 1  #count numbers of the object

                #if(i % detectDirectionPeriod == 0):
                #if(waitJustDetected>justDetected_wait):
                direction = ""
                x_var = int(newbox[0]+(newbox[2]/2)) - lastX[id]
                y_var = int(newbox[1]+(newbox[3]/2)) - lastY[id]

                print("now_x:{}, now_y:{}, last_x:{}, last_y:{}, x_var:{}, y_var:{}".format(int(newbox[0]+(newbox[2]/2)), int(newbox[1]+(newbox[3]/2)), lastX[id], lastY[id], x_var, y_var) )
                dirCar[id] = getDirection(x_var, y_var, lastDir[id])
                lastDir[id] = dirCar[id]

                (speed, fontcolor, fontbold) = getSpeed(x_var, y_var)

                lastX[id] = int(newbox[0]+(newbox[2]/2))
                lastY[id] = int(newbox[1]+(newbox[3]/2))

                #print(waitJustDetected, justDetected_wait)
                #if(waitJustDetected>justDetected_wait):
                if(dirCar[id] != "stop_stop"):
                    print("images/"+objName[id]+".png")
                    objImg = cv2.imread("images/"+objName[id]+".png", cv2.IMREAD_UNCHANGED)
                    frame = transparentOverlay(frame, objImg, (int(newbox[0]-20), int(newbox[1])), 0.7)
                    cv2.putText(frame, speed, (int(newbox[0]), int(newbox[1])), cv2.FONT_HERSHEY_COMPLEX, 0.9, fontcolor, fontbold)
                    if(dirCar[id]!=""):
                        frame = imgDirection(frame, dirCar[id], p1)

                #else:
                #    waitJustDetected += 1
        else:
            justDetected = False

        #frame = imutils.resize(frame, width=640)
        #print("Frame #{}, justDetected:{}, trackingIndex:{}, bbox:{}, obj count:{}".format(i, justDetected, trackingIndex, len(bboxes), len(boxes)) )
        cv2.imshow("preview", frame)

        # Saves for video
        out.write(frame)
        if(trackingIndex==31):
            k = cv2.waitKey(1)
        else:
            k = cv2.waitKey(1)

        if k == 0xFF & ord("q"):
            out.release()
            break
