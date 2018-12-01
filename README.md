# **Behavioral Cloning** 

## Information

### This is Udacity CarND project. 
[CarND-Behavioral-Cloning](https://github.com/udacity/CarND-Behavioral-Cloning-P3.git)

The code and related document is in [this repo](https://github.com/ChengZhongShen/Behavioral_Cloning.git)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/figure_1.png "Traning Loss"
 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for loading the images, training and saving the convolution neural network. 
the main function 'train_model()' finsish the acutual work.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is in the function 'model_navidia2b()'.(line 428-500)
This model use [Navidia]( https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) End to End training model.

It consists of 7 conv layer and 4 fully connect layer.

#### 2. Attempts to reduce overfitting in the model

The paper not mentioned about the activation and drop. I use RELU as activation fucntion and add 0.2 drop at conv layer and 0.5 drop at fully connect layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting. It use track 1 data provide by Udacity and Track 2 data collect on local enviroment.
In line 574, train_model function indicate the data source and data handleing and training parameters.
```python
    train_model(data_path=('./data/', './data_t2_2/', './data_t2_reverse/'), sample_rate=1, data_sub_read=False, data_sample_balance=0.01, sides_image=True, sides_image_offset=0.3, data_augmentation=3, 
                model='nvidia2b', input_color='YUV', learn_par=(0.001, 256, 30))
```

#### 3. Model parameter tuning

The model used an adam optimizer, mse cost function and a callback fucntion for earlystopping.
line(489, 495)
Use the adam optimizer default setting.
Change the batch_size from default 32 to 256.
Epoch = 30, but has callback earlystopping to monitor. 

#### 4. Appropriate training data

The simulator could provide the center/left/right camera image @160X320 RGB.
Three image used. Left/right image was adjust by offset compare with center image.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

#### Track 1

Training the model use the Udacity data. Load the center image and right/left image same time.

Serveral adjust points:
* the left/right image offset
* center image straight/curve data ration
* data augmentation

After try the left/right offset 0.06, 0.11, 0.22, 0.3, find 0.11-0.22 is suitable for the track 1.
The Udacity provided data include 8036 samples. More than half of the samples, the steering value is <0.02. After several try, keep 20% of the 'zero' data is a reaonable ration could keep a balance of straight/curve data.
To generate more data, randomly ratate and shift the image. (1x augmentation seems engough for track 1)

after handle the imput data, the model work well, try drive the car use the model in differet speed.

[10MPH](./video/t1_10.mp4)

[20MPH](./video/t1_20.mp4)

[30MPH](./video/t1_30.mp4), the screen shot could find [here](https://youtu.be/3jx54JCnSpU).


#### Track 2

Track 2 need more data.
Collect the track 2 data two loop in clockwise and anticlockwise.
Use the same stragety of track 1, adjust the left/right offset, center image straight/curve data ration and data augmentation.

The model could work at 10MPH, but fail at 20MPH.

[10MPH](./video/t2_10.mp4),  the screen shot could find [here](https://youtu.be/aeEd6IE6lZk).

[15MPH](./vedio/t2_15.mp4)

[20MPH](./vedio/t2_20.mp4), failed.

The training of Track 2 use a EarlyStopping call back. 30 epochs, stop at 20 epochs.
![image8]



