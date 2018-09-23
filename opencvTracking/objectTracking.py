import time

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
