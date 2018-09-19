import os

classes = 2
classList = { "0_cucumber_flower": 0, "2_cucumber_matured":1 }  #Same with you defined in 1_labels_to_yolo_format.py
folderCharacter = "/"  # \\ is for windows
cfgFolder = "cfg.cucumber"
cfg_obj_names = "obj.names"
cfg_obj_data = "obj.data"

if not os.path.exists(cfgFolder):
    os.makedirs(cfgFolder)

if not os.path.exists(cfgFolder + "/weights"):
    os.makedirs(cfgFolder + "/weights")


with open(cfgFolder + folderCharacter + cfg_obj_data, 'w') as the_file:
    the_file.write("classes= " + str(classes) + "\n")
    the_file.write("train  = " + cfgFolder + folderCharacter + "train.txt\n")
    the_file.write("valid  = " + cfgFolder + folderCharacter + "test.txt\n")
    the_file.write("names = " + cfgFolder + folderCharacter + "obj.names\n")
    the_file.write("backup = " + cfgFolder + folderCharacter + "weights/")

the_file.close()

with open(cfgFolder + folderCharacter + cfg_obj_names, 'w') as the_file:
    for className in classList:
        the_file.write(className + "\n")

the_file.close()
