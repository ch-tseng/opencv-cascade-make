from pydarknet import Detector, Image
import imutils
import cv2

video_file = "/media/sf_VMshare/IMG_7155.MOV"
output_video = "loofah.avi"

def yoloPython(img):
    img2 = Image(img)
    results = net.detect(img2)

    for cat, score, bounds in results:
        cat = cat.decode("utf-8")
        if(cat == "flower"):
            boundcolor = (252, 172, 1)
        elif(cat == "loofah"):
            boundcolor = (1, 237, 252)
        elif(cat == "loofah_wrapped"):
            boundcolor = (1, 237, 252)
        elif(cat == "loofah_under"):
            boundcolor = (1, 237, 252)

        x, y, w, h = bounds
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), 
            int(y + h / 2)), boundcolor, thickness=3)

        boundbox = cv2.imread("images/"+cat+".png")
        print("read:","images/"+cat+".png")
        print("boundbox:", boundbox.shape)
        start_x=int(x - w / 2)
        start_y=int(y - h / 2)-boundbox.shape[0]
        end_x=start_x+boundbox.shape[1]
        end_y=start_y+boundbox.shape[0]

        if(start_y<0): start_y=0
        if(start_x<0): start_x=0
        print("start_x={}, start_y={}, end_x={}, end_y={}".format(start_x,start_y,end_x,end_y))
        #boundbox = imutils.resize(boundbox, width=newshape[1], height=newshape[0])
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
