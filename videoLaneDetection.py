import cv2, os
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
    
use_gpu = True     # To use gpu, must install Cuda and Pytorch with Cuda enabled


# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

directory="Test_Videos"
for file in os.listdir(directory):

    # Initialize video
    videoPath=os.path.join(directory, file)
    cap = cv2.VideoCapture(videoPath)

    # For Video output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
    size = (width, height)
    out = cv2.VideoWriter('Output/'+file, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    
    cv2.namedWindow("Detected lanes", cv2.WINDOW_AUTOSIZE)	
    while cap.isOpened():
        try:
            ret, frame = cap.read() # Read frame from the video
        except:
            continue
        if ret:
            output_img=lane_detector.detect_lanes(frame) # Detect Lanes
            output_img=cv2.resize(output_img, size)
            cv2.imshow("Detected lanes", output_img)
            out.write(output_img)
        else:
            break   
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Detected lanes', 0)==-1:
            break
    
    
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

