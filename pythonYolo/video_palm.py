import time
import imutils
from pydarknet import Detector, Image
import cv2

FILE_OUTPUT = 'palm.avi'

if __name__ == "__main__":

    net = Detector(bytes("cfg.palm/yolov3.cfg", encoding="utf-8"), bytes("cfg.palm/yolov3_9900.weights", encoding="utf-8"), 0,
                   bytes("cfg.palm/obj.data", encoding="utf-8"))

    cap = cv2.VideoCapture("cfg.palm/IMG_3571.m4v")
    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width),int(height)))
    i = 0

    while True:
        r, frame = cap.read()
        if r:
            i += 1
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            print("{}: Elapsed Time:{}".format(i, end_time-start_time) )

            for cat, score, bounds in results:
                label = cat.decode('utf-8')
                print("{}:{}".format(cat, score))

                if(label=="palm_0"):
                    label = "0"
                elif(label=="palm_1"):
                    label = "1"
                elif(label=="palm_2"):
                    label = "2"
                elif(label=="palm_3"):
                    label = "3"
                elif(label=="palm_4"):
                    label = "4"
                elif(label=="palm_5"):
                    label = "5"


                if(label!="head"):
                    x, y, w, h = bounds
                    cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,255,0))
                    cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 255))

            # Saves for video
            out.write(frame)
            #frame = imutils.resize(frame, width=640)
            #cv2.imshow("preview", frame)


        #k = cv2.waitKey(1)
        #if k == 0xFF & ord("q"):
        #    out.release()
        #    break
