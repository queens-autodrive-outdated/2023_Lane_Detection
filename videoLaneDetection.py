import cv2, os
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import numpy as np


model_path = "models/culane_18.pth"
model_type = ModelType.CULANE
use_gpu = True     # To use gpu, must install Cuda and Pytorch with Cuda enabled

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

directory="Test_Videos/To_Run"
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
    import os
    outdir = "Outputs"
    os.makedirs(outdir, exist_ok = True)
    idx = 0
    while cap.isOpened():
        try:
            ret, frame = cap.read() # Read frame from the video
        except:
            continue
        if ret:
            lane_points, lane_detected, cfg=lane_detector.detect_lanes(frame) # Detect Lanes

            output_img =lane_detector.draw_lanes(frame, lane_points, lane_detected, cfg, True)

            cv2.imshow("Detected lanes", out_stacked)
            out.write(out_stacked)
        else:
            break   
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Detected lanes', 0)==-1:
            break
    
    
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

