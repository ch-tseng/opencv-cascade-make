import time
import imutils
from pydarknet import Detector, Image
import cv2

FILE_OUTPUT = 'cucumber.avi'

if __name__ == "__main__":

    net = Detector(bytes("../yolo-v3/cfg.cucumber/yolov3.cfg", encoding="utf-8"), 
                   bytes("../yolo-v3/cfg.cucumber/weights/yolov3_30000.weights", encoding="utf-8"), 0,
                   bytes("../yolo-v3/cfg.cucumber/obj.data", encoding="utf-8"))

    cap = cv2.VideoCapture("/home/digits/Videos/cucumber_2018_2.mov")

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

                if(label=="0_cucumber_flower"):
                    labelName = "flower"
                    labelColor = (255, 255, 255)
                elif(label=="2_cucumber_matured"):
                    labelName = "cucumber"
                    labelColor = (193, 161, 31)
                else:
                    labelName = "unknow"
                    labelColor = (255, 255, 255)


                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),labelColor, 3)

                boundbox = cv2.imread("cucumber_images/"+label+".jpg")

                try:
                    frame[ int(y-h/2):int(y-h/2)+boundbox.shape[0], int(x-w/2):int(x-w/2)+boundbox.shape[1]] = boundbox
                    print("read:",label+".jpg")

                except:
                    print("add text: ",labelName)
                    cv2.putText(frame, labelName, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, labelColor)

            # Saves for video
            out.write(frame)
            #frame = imutils.resize(frame, width=640)
            cv2.imshow("preview", frame)


        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            out.release()
            break
