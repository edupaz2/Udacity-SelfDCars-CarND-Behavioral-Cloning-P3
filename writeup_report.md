# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[cnn]: ./cnn-architecture.png "NVIDIA CNN"
[center]: ./center.jpg "center"
[center_view]: ./center_view.jpg "center view"
[left_view]: ./left_view.jpg "left view"
[right_view]: ./right_view.jpg "right view"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Network Architecture proposed by NVIDIA (more at https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

You can check the implementation at model.py, line 72, Nvidia function.

The model consists of a:
* Normalization layer, using a Keras lambda layer.
* 5 Convolutional layers, using 5x5 and 3x3 kernels and depths between 24, 36, 48 and 64, and including RELU layers to introduce nonlinearity.
* 4 Fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 146).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road introducing a correction steering factor of 2.0 from the center lane steering value. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to feed the model with the best training data I could get.

My first step was implementing the LeNet architecture. Then I proceed to use the simulator in order to get data to feed my model with. I applied normalization to the data, use the left and right cameras images, and generate new data by mirroring the current images.

After checking the simulator started to "drive", I implemented the convolution neural network proposed by the Nvidia team. This model performed very good at the very first attempt.

My next step was to gather small pieces of data at the points where the model was not behaving properly, for example: at the bridge, and at some parts of the track where the surface changed.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set, being the validation a 20% of the training set.

I must say the sample data was very helpful too.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][cnn]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I added the sample data provided in the repository, and use left and right sides too (using a steering correction factor of 2.0).

![alt text][left_view]
![alt text][center_view]
![alt text][right_view]

After that I augmented the data by flipping the current images and angles (horizontal mirror).
Next step was adding small records of data where the vehicle was failing to track the lane properly.

Then I repeated this process on track two in order to get more data points, but the model behaved very bad, so I discarded that data. I guess this new data will need more preprocessing.

After the collection process, my numbers were:

Datasets size: 25002.
Training set: 20001
Validation set: 5001 (20% of the total)