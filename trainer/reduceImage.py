import cv2
import os
import sys

unaltered = "C:/Users/Nathan/Desktop/Tennis/Unaltered"
reduced = "C:/Users/Nathan/Desktop/Tennis/Reduced"

#Get number of files in directory
path, dirs, files = os.walk(unaltered).next()
fileCount = len(files)

#For each file in directory
for i, filename in enumerate(os.listdir(unaltered)):
    #Read in image
    img = cv2.imread(os.path.join(unaltered,filename))
    #If not at end of folder
    if img is not None:
        #Get height and width, then print relevant info
        height, width, channels = img.shape
        sys.stdout.write("(" + str(i+1) + "/" + str(fileCount) + ") - " + str(filename) + " - " + str(height) + "x" + str(width) + "\n")

        #Crop according to which dimension is larger
        if height > width:
            crop_img = img[((height - width) / 2):(((height - width) / 2) + width), 0:width]
        else:
            crop_img = img[0:height, ((width - height) / 2):(((width - height) / 2) + height)]

        #Resize the cropped image
        resized_image = cv2.resize(crop_img, (32, 32))
        #Write cropped image to Reduced folder
        cv2.imwrite(reduced + "/" + filename, resized_image)

cv2.waitKey(0)