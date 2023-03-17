from LaneDetection import LaneDetector
import os

detector = LaneDetector()
directory="Test_Videos/To_Run"
for file in os.listdir(directory):
    # Initialize video
    videoPath=os.path.join(directory, file)
    detector.video_visualization(videoPath)