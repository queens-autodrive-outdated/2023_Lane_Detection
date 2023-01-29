import cv2
import os

from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# Change model value to choose between CULane and TU Simple datasets
TUSIMPLE = 0
CULANE = 1

model=CULANE

if model:
    model_path = "models/culane_18.pth"
    model_type = ModelType.CULANE
elif not model:
    model_path = "models/tusimple_18.pth"
    model_type = ModelType.TUSIMPLE
    
use_gpu = False     # To use gpu, must install Cuda and Pytorch with Cuda enabled



# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)


def detectLanes(img, fileName):
    

    # Detect the lanes
    output_img = lane_detector.detect_lanes(img)

    # Draw estimated depth
    cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL) 
    cv2.imshow("Detected lanes", output_img)
    cv2.imwrite("Output/"+fileName, output_img)
    cv2.waitKey(0)

    cv2.imwrite("output.jpg",output_img)




# Loop through images in directory
image_directory = "Test_Images"
for file in os.listdir(image_directory):
    # Read RGB images
    img=os.path.join(image_directory, file)
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1280,720), interpolation = cv2.INTER_AREA)
    
    detectLanes(img, file)






