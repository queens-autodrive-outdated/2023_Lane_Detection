# Lane Detection: Current Progress

![Campus](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgif1.gif) 

More demos at bottom of page

# Requirements

 * **OpenCV**, **Scikit-learn** and **pytorch**.
 
# Installation
Create Conda Environment:
```
conda create -n lane_det python=3.8 -y
conda activate lane_det
```

Install required packages:
```
pip install -r requirements.txt
```

#### Pytorch:
Check the [Pytorch website](https://pytorch.org/) to find the best method to install Pytorch in your computer.

To use without a GPU, normal versions of Pytorch will work. If setting the *use_gpu* flag to true, you must
install CUDA before installing the CUDA enabled versions of Pytorch from the [Pytorch website](https://pytorch.org/)

## Pretrained model

The pretrained models (CULane) must be downloaded from [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) and placed in the model/LaneDetection folder for the code to run. [DOWNLOAD THIS FILE - CULANE PRETRAINED MODEL](https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view)

# Current Input and Output

* **Input**: RGB image of size 1280 x 720 pixels.
* **Output**: Keypoints for a maximum of 4 lanes (left-most lane, left lane, right lane, and right-most lane).
 



# Running the Code

 **Image inference**: 
 Takes in provided images and displays the detected lane on the image. Place all test images in the **Test_Images** folder. The output of each photo is saved in **Output**
 
 ```
 python imageLaneDetection.py 
 ```
 
**Video inference**: 
Takes .mp4 videos saved in the **Test_Videos** folder and detects the lane on each frame. The output for each video is saved in **Output** as an .mp4
 
 ```
 python videoLaneDetection.py
 ```

**Webcam inference**: *Not Tested or editted from original*
 
 ```
 python webcamLaneDetection.py
 ```

## Kingston Demos
![Campus](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgif1.gif) 
![Union](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgif2.gif)
![Union&SJA](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgif3.gif) 
![Johnson](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgif4.gif)
