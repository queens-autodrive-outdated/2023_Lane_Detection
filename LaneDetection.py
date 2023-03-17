import cv2, os
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import numpy as np

class LaneDetector:
    def __init__(self):
        model_path = "model/LaneDetection/culane_18.pth"
        model_type = ModelType.CULANE
        use_gpu = True

        # Initialize lane detector
        self.lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

        # Initialize Camera
        self.cap = cv2.VideoCapture(0) # Get frame from camera
        self.image_width=1280
        self.image_height=720
        self.size=(self.image_width, self.image_height)

        self.lane_points=[]
        self.lane_detected=[]
        self.model_config=None

    def set_image_size(self, w, h):
        self.image_width=w
        self.image_height=h
        self.size=(self.image_width, self.image_height)

    def detect_lanes(self, frame):
        points, laneDetected, cfg = self.lane_detector.detect_lanes(frame) # Detect Lanes
        self.lane_points=points
        self.lane_detected=laneDetected
        self.model_config=cfg
        
    def output_lines_on_img(self,frame):
        output_img=self.lane_detector.draw_lanes(frame, self.lane_points, self.lane_detected, self.model_config, True)
        return output_img

    def video_visualization(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
        self.image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
        self.size = (self.image_width, self.image_height)
        
        cv2.namedWindow("Detected lanes", cv2.WINDOW_AUTOSIZE)

        while cap.isOpened():
            try:
                ret, frame = cap.read() # Read frame from the video
            except:
                continue
            if ret:
                self.detect_lanes(frame)
                out_img=self.output_lines_on_img(frame)
                cv2.imshow("Detected lanes", out_img)
            else:
                break   
            if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Detected lanes', 0)==-1:
                break
        cap.release()
        cv2.destroyAllWindows()

        









