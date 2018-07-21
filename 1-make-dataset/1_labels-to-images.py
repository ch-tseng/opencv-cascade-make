import glob, os
import os.path
import time
import cv2
from xml.dom import minidom
from os.path import basename
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

folderCharacter = "/"  # \\ is for windows
xmlFolder = "/home/chtseng/works/opencv-cascade/dataset/palm/v1/labels"
imgFolder = "/home/chtseng/works/opencv-cascade/dataset/palm/v1/images"
labelName = "palm"
saveROIsPath = "dataset/positives"
imageKeepType = "jpg"
generateNegativeSource = True
negSourceOutput = "dataset/neg-Source"
#useAugmentation = True
#augCounts = 5  #create how many images for 1 ?

totalLabels = 0
wLabels = 0
hLabels = 0

def saveROI( roiSavePath, imgFolder, xmlFilepath, labelGrep="", generateNeg=False):
    global totalLabels, wLabels, hLabels, negSourceOutput
    
    xml_filename, xml_file_extension = os.path.splitext(xmlFilepath)
    xml_filename = basename(xml_filename)

    labelXML = minidom.parse(xmlFilepath)
    labelName = []
    labelXstart = []
    labelYstart = []
    labelW = []
    labelH = []
    totalW = 0
    totalH = 0
    countLabels = 0

    tmpArrays = labelXML.getElementsByTagName("filename")
    for elem in tmpArrays:
        filenameImage = elem.firstChild.data
    #print ("Image file: " + filenameImage)

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        labelName.append(str(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmin")
    for elem in tmpArrays:
        labelXstart.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymin")
    for elem in tmpArrays:
        labelYstart.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmax")
    for elem in tmpArrays:
        labelW.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymax")
    for elem in tmpArrays:
        labelH.append(int(elem.firstChild.data))
        
    image = cv2.imread(imgFolder + "/" + filenameImage)
    image2 = image.copy()
    filepath = imgFolder
    filename = filenameImage

    for i in range(0, len(labelName)):
        if(labelGrep=="" or labelGrep==labelName[i]):
            countLabels += 1
            totalW = totalW + int(labelW[i]-labelXstart[i])
            totalH = totalH + int(labelH[i]-labelYstart[i])
            
            #roi = roi[...,::-1]
            #get the label image from the source image
            roi = image[labelYstart[i]:labelH[i], labelXstart[i]:labelW[i]]
            
            roiFile = roiSavePath + folderCharacter + xml_filename + '_' + str(countLabels)+"."+imageKeepType
            cv2.imwrite(roiFile, roi)

            if(generateNeg==True):
                cv2.rectangle(image, (labelXstart[i], labelYstart[i]), 
                    (labelXstart[i]+int(labelW[i]-labelXstart[i]), labelYstart[i]+int(labelH[i]-labelYstart[i])), (0,0,0), -1)

            '''
            if(useAugmentation==True):
                roi = roi[...,::-1]
                #roi_augFile = roiSavePath + folderCharacter + "aug_" + xml_filename + '_' + str(countLabels)
                #Image augmentation
                datagen = ImageDataGenerator(
                    zca_whitening=False,
                    rotation_range=180,
                    #width_shift_range=0.2,
                    #height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest")

            
                x = img_to_array(roi)   # this is a Numpy array with shape (3, 150, 150)
                x = x.reshape (( 1 ,) + x.shape)   # this is a Numpy array with shape (1, 3, 150, 150)
                i =  0
                for batch in datagen.flow(x, batch_size = 1 ,
                    save_to_dir = roiSavePath, save_prefix = "aug_" + xml_filename, save_format = 'jpg' ):

                    i +=  1
                    if i >  augCounts:
                        break   # otherwise the generator would loop indefinitely
               '''
    if(generateNeg==True):
       negFile = negSourceOutput + folderCharacter + xml_filename + '_' + str(countLabels)+"."+imageKeepType
       cv2.imwrite(negFile, image)



    wLabels += totalW
    hLabels += totalH
    totalLabels += countLabels

    #if(countLabels>0): print("Average W, H: {}, {}".format(int(totalW/countLabels), int(totalH/countLabels)) )
    print("    find {}/{} labels.".format(countLabels, totalLabels) )



#Create all required folders
if not os.path.exists(saveROIsPath):
    os.makedirs(saveROIsPath)

if not os.path.exists(saveROIsPath + folderCharacter):
    os.makedirs(saveROIsPath + folderCharacter)

if(generateNegativeSource == True):
    if not os.path.exists(negSourceOutput):
        os.makedirs(negSourceOutput)

#make positive images
fileCount = 0
for file in os.listdir(xmlFolder):
    filename, file_extension = os.path.splitext(file)
    if(file_extension==".xml"):
        fileCount += 1
        print("processing XML: {}".format(filename))
        
        xmlfile = xmlFolder + folderCharacter + file        

        saveROI(saveROIsPath, imgFolder, xmlfile, labelName, generateNegativeSource)


avgW = round(wLabels/totalLabels, 1)
avgH = round(hLabels/totalLabels,1)

with open(saveROIsPath + folderCharacter + "desc.txt", 'a') as the_file:
    the_file.write("{} XML file processed \n".format(fileCount))
    the_file.write("Total labels: {} \n".format(totalLabels))
    the_file.write("Average W:H = {}:{} \n".format(avgW, avgH))	

print("----> Average W:H = {}:{}".format(avgW, avgH ))
