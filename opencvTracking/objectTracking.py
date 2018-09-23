import time
import cv2
import numpy as np

class trackingObj():
    def __init__(self):
        self.time = time.time()
        self.label = []
        self.bbox = []
        self.direction = []
        self.counted = []
        self.bbxBold = []
        self.bbxColor = []
        self.bbxSpeed = []

    def updateCenterXY(id, XY=(0,0)):
        self.lastXY[id] = self.nowXY[id]
        self.historyXY.append(self.nowXY[id])
        self.nowXY[id] = XY

    def updateDirection(id, direction):
        self.lastDir[id] = self.direction[id]
        self.direction[id] = direction

class opencvYOLO():
    def __init__(self, modeltype="yolov3", objnames="coco.names", weights="yolov3.weights", cfg="yolov3.cfg"):
        self.modeltype = modeltype
        self.score = 0.5
        self.nms = 0.6

        if(modeltype=="yolov3"):
            self.inpWidth = 608
            self.inpHeight = 608
        else:  # yolov3-tiny
            self.inpWidth = 416
            self.inpHeight = 416

        self.classes = None
        with open(objnames, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        dnn = cv2.dnn.readNetFromDarknet(cfg, weights)
        dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.net = dnn

    def setScore(self, score=0.5):
        self.score = score

    def setNMS(self, nms=0.6):
        self.nms = nms

    # Get the names of the output layers
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
 
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.score:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score, self.nms)
        self.indices = indices

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

        self.bbox = boxes
        self.classIds = classIds
        self.scores = confidences

    # Draw the predicted bounding box
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    def getObject(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
        # Sets the input to the network
        net = self.net
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(self.getOutputsNames(net))
        # Remove the bounding boxes with low confidence
        self.postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the 
        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

    def listLabels(self):
        for i in self.indices:
            i = i[0]
            box = self.bbox[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            classes = self.classes
            print("Label:{}, score:{}, left:{}, top:{}, right:{}, bottom:{}".format(classes[self.classIds[i]], self.scores[i], left, top, left + width, top + height) )

