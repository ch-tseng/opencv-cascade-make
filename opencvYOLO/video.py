import cv2
import numpy as np
FILE_OUTPUT = 'output/loofah.avi'

cap = cv2.VideoCapture('/media/sf_shares/loofah.mov')
# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 30.0, (int(width),int(height)))

model_path = 'cfg.loofah/yolov3-tiny_10000.weights'
cfg_path = 'cfg.loofah/yolov3-tiny.cfg'
yolo_net = cv2.dnn.readNet(model_path, cfg_path, 'darknet')
i = 0

while True:
    flag, img = cap.read()
    print("frame #{}".format(i))
    if flag:
        rows = img.shape[0]
        cols = img.shape[1]

        yolo_net.setInput(cv2.dnn.blobFromImage(img, 1, (416, 416), (127.5, 127.5, 127.5), False, False))
        cvOut = yolo_net.forward()
        for detection in cvOut:
            confidence = np.max(detection[5:])
            if confidence > 0:
                classIndex = np.argwhere(detection == confidence)[0][0] - 5
                x_center = detection[0] * cols
                y_center = detection[1] * rows
                print("x_center, y_center", x_center, y_center)
                width = detection[2] * cols
                height = detection[3] * rows
                print("width, height", width, height)
                start = (int(x_center - width/2), int(y_center - height/2))
                end = (int(x_center + width/2), int(y_center + height/2))
                print("start, end", start, end)
                cv2.rectangle(img, start, end ,(0, 0, 255), thickness=3)

        if(len(cvOut)>0): cv2.imwrite('output/show'+str(i)+".jpg", img)
        out.write(img)
        i +=1

    else:
        break
