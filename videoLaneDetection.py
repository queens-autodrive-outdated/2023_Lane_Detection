import cv2, os
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import numpy as np


model_path = "models/culane_18.pth"
model_type = ModelType.CULANE
use_gpu = True     # To use gpu, must install Cuda and Pytorch with Cuda enabled

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

#################
def moving_avg(new_arr, old_arr, new_weight = 0.05):
    return new_arr * new_weight + old_arr * (1-new_weight)

class MvArray():
    def __init__(self, initial_array, alpha):
        self.array = initial_array
        self.cnt = 0
        self.alpha = alpha
    
    def update(self, new_array):
        c = max(self.alpha, 1 / (self.cnt + 1))
        self.array = moving_avg(new_array, self.array, new_weight= c)
        self.cnt += 1
        return self.array

class Line():
    def __init__(self, lines_alpha = 0.05, limits_alpha = 0.05, n_degrees = 3):
        self.n_degrees = n_degrees
        self.cnt = 0
        self.limits_alpha = limits_alpha
        self.lines_alpha = lines_alpha
        
        
    def update(self, new_line):
        if self.cnt == 0 :
            
            if len(new_line):
            
                p = np.polyfit(new_line[:,0], new_line[:,1], self.n_degrees)
                (x_min, _), (x_max, _) = line1.min(0), line1.max(0)

                self.limits = MvArray(np.array([x_min, x_max]), self.limits_alpha)
                x_min, x_max = self.limits.array

                self.axis = np.linspace(x_min, x_max, 10)
                npred_line = np.polyval(p, self.axis) #predicted line
                pred_line = npred_line

                self.predicted_line = MvArray(pred_line, self.lines_alpha)

                self.cnt += 1
        
        else:
            if len(new_line):
                p = np.polyfit(new_line[:,0], new_line[:,1], self.n_degrees)
                (nx_min, _), (nx_max, _) = new_line.min(0), new_line.max(0)

                self.limits.update(np.array([nx_min, nx_max]))
                x_min, x_max = self.limits.array

                self.axis = np.linspace(x_min, x_max, 10)
                npred_line = np.polyval(p, self.axis)

                self.predicted_line.update(npred_line)
                self.cnt += 1
            
    def get_line(self):
        try:
            return np.array([self.axis, self.predicted_line.array]).T
        except:
            return np.array(np.zeros((10,2)))
#### END OF CLASS LINE


# For moving average buffer
BUFFERSIZE=20
bufferPos=0

# Store values of lines
list_lane0 = []
list_lane1 = []
list_lane2 = []
list_lane3 = []

# create an instance for each line
myline0 = Line(lines_alpha = 0.05, limits_alpha = 0.05, n_degrees = 3)
myline1 = Line(lines_alpha = 0.05, limits_alpha = 0.05, n_degrees = 3)
myline2 = Line(lines_alpha = 0.05, limits_alpha = 0.05, n_degrees = 3)
myline3 = Line(lines_alpha = 0.05, limits_alpha = 0.05, n_degrees = 3)

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
            pline=[None]*4

            print(idx)
            if len(list_lane1)<BUFFERSIZE:
                print("if")
                list_lane0.append(lane_points[0])
                list_lane1.append(lane_points[1])
                list_lane2.append(lane_points[2])
                list_lane3.append(lane_points[3])
                pline=lane_points
                    
            else:
                print("Else")
                list_lane0[idx]=lane_points[0]
                list_lane1[idx]=lane_points[1]
                list_lane2[idx]=lane_points[2]
                list_lane3[idx]=lane_points[3]
                # pline=lane_points

                # Do point manipulation
                for i,(line0,line1,line2, line3) in enumerate(zip(list_lane0, list_lane1, list_lane2, list_lane3)): 
    
                    myline0.update(line0) # for each frame, update the line point values
                    myline1.update(line1) # for each frame, update the line point values
                    myline2.update(line2) # for each frame, update the line point values
                    myline3.update(line3) # for each frame, update the line point values
                    
                    pline[0] = myline0.get_line() #this code gets the line points from the Line object
                    pline[1] = myline1.get_line()
                    pline[2] = myline2.get_line()
                    pline[3] = myline3.get_line()

                # Increment idx or reset to beginning
                idx=(idx+1)%BUFFERSIZE
                
            output_img=lane_detector.draw_lanes(frame, pline, lane_detected, cfg, True)

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

