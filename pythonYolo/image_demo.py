from pydarknet import Detector, Image
import argparse
import cv2
import os
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the source image")
args = vars(ap.parse_args())

if __name__ == "__main__":
    # net = Detector(bytes("cfg/densenet201.cfg", encoding="utf-8"), bytes("densenet201.weights", encoding="utf-8"), 0, bytes("cfg/imagenet1k.data",encoding="utf-8"))

    net = Detector(bytes("/home/chtseng/works/opencv-cascade/yolo-v3/cfg/yolov3.cfg", encoding="utf-8"), bytes("/home/chtseng/works/opencv-cascade/yolo-v3/cfg/yolov3_11600.weights", encoding="utf-8"), 0, bytes("/home/chtseng/works/opencv-cascade/yolo-v3/cfg/obj.data",encoding="utf-8"))

    img = cv2.imread(args["image"])

    img2 = Image(img)

    # r = net.classify(img2)
    results = net.detect(img2)
    print(results)

    for cat, score, bounds in results:
        x, y, w, h = bounds
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
        cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))

    img = imutils.resize(img, width=700)
    cv2.imshow("output", img)
    # img2 = pydarknet.load_image(img)

    cv2.waitKey(0)
