import time
import cv2
import numpy as np

class trackingObj():
    def __init__(self):
        self.multiTracker = cv2.MultiTracker_create()

        self.time = time.time()
        self.label = []
        self.bbox = []
        self.direction = []
        self.counted = []
        self.bbxBold = []
        self.bbxColor = []
        self.bbxSpeed = []

    def createTrackerByName(self, trackerType):
        trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

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

    def createTrackers(self, frame, bboxes, trackerType="CSRT"):
        trackers = self.multiTracker
        self.bboxes = []
        self.color = []
        self.counted = []
        self.bold = []

        for bbox in bboxes:
            #print(bbox)
            trackers.add(self.createTrackerByName(trackerType), frame, bbox)
            self.bboxes.append(bbox)
            self.color.append((0,0,255))
            self.counted.append(False)
            self.bold.append(1)

        self.multiTracker = trackers

    def updateTrackers(self, frame):
        success, boxes = self.multiTracker.update(frame)

        for id, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

            if(self.counted[id]==True):
                self.color[id] = (255, 0, 0)
            else:
                self.color[id] = (0,0,255)

            cv2.rectangle(frame, p1, p2, self.color[id], self.bold[id], 1)

        return (success, boxes, frame)

    def updateCenterXY(id, XY=(0,0)):
        self.lastXY[id] = self.nowXY[id]
        self.historyXY.append(self.nowXY[id])
        self.nowXY[id] = XY

    def updateDirection(id, direction):
        self.lastDir[id] = self.direction[id]
        self.direction[id] = direction

