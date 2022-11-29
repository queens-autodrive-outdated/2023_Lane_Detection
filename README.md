# Lane Detection: Current Progress

![!Lane detection demo 1](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgit1.gif)
![!demo 2](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgit2.gif)
![!demo 3](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgit3.gif)
![!demo 4](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgit4.gif)

# Requirements

 * **OpenCV**, **Scikit-learn** and **pytorch**.
 
# Installation
```
pip install -r requirements
```

#### Pytorch:
Check the [Pytorch website](https://pytorch.org/) to find the best method to install Pytorch in your computer.

To use without a GPU, normal versions of Pytorch will work. If setting the *use_gpu* flag to true, you must
install CUDA before installing the CUDA enabled versions of Pytorch from the [Pytorch website](https://pytorch.org/)

## Pretrained model

The pretrained models (TuSimple and CULane) must be downloaded from [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) and placed in the models folder for the code to run. 

In the code for detection, you can change the model between the pretrained models using the CULane and TuSimple datasets. Currently using the r18 verions.
We will need to figure out how to implement newer/different models, likely to be based on [Ultra-Fast-Lane-Detection-V2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) or [CLRNet](https://github.com/Turoad/CLRNet) to improve results


# Current Input and Output

* **Input**: RGB image of size 1280 x 720 pixels.
* **Output**: Keypoints for a maximum of 4 lanes (left-most lane, left lane, right lane, and right-most lane).
 



# Examples

 **Image inference**: Takes in provided images and displays the detected lane on the image. Place all test images in the **Test_Images** folder. The output of each photo is saved in **Output**
 
 ```
 python imageLaneDetection.py 
 ```
 
**Video inference**: Takes .mp4 videos saved in the **Test_Videos** folder and detects the lane on each frame. The output for each video is saved in **Output** as an .mp4
 
 ```
 python videoLaneDetection.py
 ```

**Webcam inference**: *Not Tested or editted from original*
 
 ```
 python webcamLaneDetection.py
 ```

 

 # Progress

 ![!demo 2](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgit2.gif)
![!demo 3](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgit3.gif)
![!demo 4](https://github.com/ryanbaker77/lane-detection/blob/main/demos/LDgit4.gif)
