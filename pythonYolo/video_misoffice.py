import time
import imutils
from pydarknet import Detector, Image
import cv2

if __name__ == "__main__":

    net = Detector(bytes("/home/chtseng/works/opencv-cascade/yolo-v2/cfg/tiny-yolo.cfg", encoding="utf-8"), bytes("/home/chtseng/works/opencv-cascade/yolo-v2/cfg/tiny-yolo_9900.weights", encoding="utf-8"), 0,
                   bytes("/home/chtseng/works/opencv-cascade/yolo-v2/cfg/obj.data", encoding="utf-8"))

    cap = cv2.VideoCapture("/media/sf_VMshare/Videos/misoffice_3x.mp4")

    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            print("Elapsed Time:",end_time-start_time)

            for cat, score, bounds in results:
                label = cat.decode('utf-8')
                print("{}:{}".format(cat, score))

                if(label!="head"):
                    x, y, w, h = bounds
                    cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                    cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

            frame = imutils.resize(frame, width=640)
            cv2.imshow("preview", frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break
