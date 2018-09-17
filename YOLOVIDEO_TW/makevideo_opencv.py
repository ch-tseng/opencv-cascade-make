import os, time
import argparse
import cv2 as cv
import numpy as np
import webcolors

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 608       #Width of network's input image
inpHeight = 608      #Height of network's input image

FILE_OUTPUT = '/media/sf_shares/output/pepper_1.avi'
# Load names of classes
classesFile = "cfg.pepper/obj.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
 
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "cfg.pepper/yolov3.cfg";
modelWeights = "cfg.pepper/weights/yolov3_40000.weights";
 
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

outputFile = FILE_OUTPUT
parser = argparse.ArgumentParser(description="Do you wish to scan for live hosts or conduct a port scan?")
parser.add_argument("-i", dest='image', action='store', help='Image')
parser.add_argument("-v", dest='video', action='store',help='Video file')

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def getROI_Color(roi):
    mean_blue = np.mean(roi[:,:,0])
    mean_green = np.mean(roi[:,:,1])
    mean_red = np.mean(roi[:,:,2])

    actual_name, closest_name = get_colour_name((mean_red, mean_green, mean_blue))

    #actual_name, closest_name = get_colour_name((mean_blue, mean_green, mean_red))

    return (actual_name, closest_name, (mean_blue, mean_green, mean_red))

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, orgFrame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
 
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
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
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, orgFrame)

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, orgFrame):
    if(classes[classId]=="0_pepper_flower"):
        labelName = "flower"
        labelColor = (255, 255, 255)
    elif(classes[classId]=="1_pepper_young"):
        labelName = "young"
        labelColor = (193, 161, 31)
    elif(classes[classId]=="2_pepper_matured"):
        labelName = "Pepper"
        labelColor = (12, 255, 240)
    else:
        labelName = "unknow"
        labelColor = (255, 255, 255)

    boundbox = cv.imread("cfg.pepper/images/"+classes[classId]+".jpg")
    print("boundbox:", boundbox.shape)
    start_x=left
    start_y=top
    end_x=start_x+boundbox.shape[1]
    end_y=start_y+boundbox.shape[0]
    center_x = left + int((right-left)/2)
    center_y = top + int((bottom-top)/2)
    print("(end_x-start_x)={}, (end_y-start_y)={}, img.shape[1]={}, img.shape[0]={}".format((end_x-start_x),(end_y-start_y),boundbox.shape[1],boundbox.shape[0]))
    #(color1, color2, colorDetect) = getROI_Color(frame[ start_y:end_y, start_x:end_x])
    if(center_x-10<0):
        roi_x1 = 0
    else:
        roi_x1= center_x - 10

    if(center_y-10<0): 
        roi_y1 = 0
    else:
        roi_y1= center_y - 10

    if(center_x+10>frame.shape[1]): 
        roi_x2 = frame.shape[1]
    else:
        roi_x2= center_x + 10

    if(center_y+10>frame.shape[0]): 
        roi_y2 = frame.shape[0]
    else:
        roi_y2= center_y + 10

    (color1, color2, colorDetect) = getROI_Color(orgFrame[ roi_y1:roi_y2, roi_x1:roi_x2])
    print("color1:{}, color2:{}".format(color1, color2))
    cv.imwrite("/media/sf_shares/output/colors/" + color2 + "_" + str(time.time()) + ".jpg", orgFrame[ top:bottom, left:right])

    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), colorDetect, 3)

    #cv.putText(frame, color2, (center_x, center_y), cv.FONT_HERSHEY_COMPLEX, 1.8, colorDetect, 2)

    try:
        frame[ start_y:end_y, start_x:end_x] = boundbox
        print("read:","images/"+classes[classId]+".jpg")

    except:
        print("add text: ",labelName)
        cv.putText(frame, labelName, (center_x, center_y), cv.FONT_HERSHEY_COMPLEX, 1.6, labelColor, 2)


    #label = '%.2f' % conf
    # Get the label for the class name and its confidence
    #if classes:
    #    assert(classId < len(classes))
    #    label = '%s:%s' % (classes[classId], label)
 
    #Display the label at the top of the bounding box
    #labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 5, 2)
    #top = max(top, labelSize[1])

    #print(label, labelName)
    #cv.putText(frame, labelName, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1.5, labelColor, 2)

args = parser.parse_args()
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)
 
# Get the video writer initialized to save the output video
if (not args.image):
    #vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(FILE_OUTPUT,fourcc, 30.0, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

i = 0
while cv.waitKey(1) < 0:
     
    # get frame from the video
    hasFrame, frame = cap.read()
    i += 1 
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    orgFrame = frame.copy()
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
 
    # Sets the input to the network
    net.setInput(blob)
 
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
 
    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, orgFrame)
 
    # Put efficiency information. The function getPerfProfile returns the 
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

    #cv.putText(frame, label, (30, 40), cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
 
    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        print("Frame #", i)
        #vid_writer.write(frame.astype(np.uint8))
        out.write(frame)
        #cv.imshow("frame", frame)
        cv.waitKey(1)



