from pydarknet import Detector, Image
import imutils
import cv2

video_file = "cfg.loofah/IMG_6850.m4v"
output_video = "loofah.avi"

def yoloPython(img):
    img2 = Image(img)
    results = net.detect(img2)

    for cat, score, bounds in results:
        cat = cat.decode("utf-8")
        if(cat == "flower"):
            boundcolor = (2, 233, 247)
        elif(cat == "loofah"):
            boundcolor = (1, 251, 42)
        elif(cat == "loofah_wrapped"):
            boundcolor = (1, 251, 42)
        elif(cat == "loofah_under"):
            boundcolor = (1, 251, 42)

        x, y, w, h = bounds
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), 
            int(y + h / 2)), boundcolor, thickness=3)

        boundbox = cv2.imread("images/"+cat+".png")
        print("read:","images/"+cat+".png")
        #img[ int(y-h/2):int(y-h/2)+boundbox.shape[0], int(x-w/2):int(x-w/2)+boundbox.shape[1]] = boundbox
        start_y = int(y - (h/2))
        end_y = start_y + boundbox.shape[0]
        start_x = int(x - (w/2))
        end_x = start_x + boundbox.shape[1]
        print(start_y, end_y, start_x, end_x)
        if(start_y<0): start_y=0
        if(start_x<0): start_x=0
        boundbox = imutils.resize(boundbox, width=end_x-start_x, height=end_y-start_y)
        img[ start_y:end_y, start_x:end_x] = boundbox

    return img

net = Detector(bytes("cfg.loofah/yolov3-tiny.cfg", encoding="utf-8"),
    bytes("cfg.loofah/yolov3-tiny_10000.weights", encoding="utf-8"), 0,
    bytes("cfg.loofah/obj.data",encoding="utf-8"))

cap = cv2.VideoCapture(video_file)
# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video,fourcc, 30.0, (int(width),int(height)))
i = 0

while True:
    r, img = cap.read()
    img = yoloPython(img)
    out.write(img)
