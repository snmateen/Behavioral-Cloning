# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./examples/center_500x200.jpg "Center Image"
[image3]: ./examples/rsz_center_2017_04_25_01_34_52_687.jpg "Recovery Image"
[image4]: ./examples/rsz_center_2017_04_25_01_34_51_504.jpg "Recovery Image"
[image5]: ./examples/rsz_center_2017_04_25_01_34_52_197.jpg "Recovery Image"
[image6]: ./examples/center_2017_04_25_01_00_19_703.jpg "Normal Image"
[image7]: ./examples/flipped_center.jpg "Flipped Image"
[image8]: ./examples/noised.jpg "Noised Image"
[image9]: ./examples/random_shear.jpg "Sheared Image"

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 - video capture of the autonomous mode using the trained model
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

1. An appropriate model architecture has been employed

* My model consists of a convolution neural network with 5x5 / 3x3 filter sizes and depths between 24 and 64 (model.py lines 99-103) 

```python
# add 5 convolution layers
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
```

* The model includes RELU layers to introduce nonlinearity (code line 99-103), and the data is normalized in the model using a Keras lambda layer (code line 93). 

```python
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(rows,cols,depth)))
```

2. Attempts to reduce overfitting in the model

* The model contains dropout layers in order to reduce overfitting (model.py lines 111 and 114). 

```python
# add 3 fully connected layers
# with 50% drop out to avoid overfitting
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
```

* The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 124-128). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

```python
# train and validate
# total number of epochs = 5
model.fit_generator(train_generator
                    , samples_per_epoch= 12 * len(train_samples)
                    , validation_data=validation_generator
                    , nb_val_samples=len(validation_samples)
                    , nb_epoch=5)
```

3. Model parameter tuning

* The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 124-128).

4. Appropriate training data

* Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Data was collected driving the car for 3 loops in counter clockwise direction and 2 loops in clockwise directions.

_For details about how I created the training data, see the next section._

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with simple such as LeNet architecture and move on to using NVIDIA self driving car architecture which can be found [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

My first step was to use a convolution neural network model similar to the LeNet architecture, more information can be found [here](http://yann.lecun.com/exdb/lenet/), I thought this model might be appropriate because it gives a good starting point for training self driving car model using camera images.

As a next step I moved onto implementing NVIDIA self driving car convolution neural network architecture with few additional changes in preprocessing such as cropping the image to remove top 65 and bottom 25 pixels as they add more of noise to model rather than useful information.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that Dropout layers are added in the fully connected layers with 50% drop out.

Then I added few more images as input to generalize the model well.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as, around the corners and near the bridge, to improve the driving behavior in these cases, I added few more training data that show recovering from the sides.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 90-117) consisted of a convolution neural network with the following layers and layer sizes 

3 Strided convolution layers with 2x2 strides
- convolution layer of 24 depth with 5x5 filter
- convolution layer of 36 depth with 5x5 filter
- convolution layer of 48 depth with 5x5 filter

2 Non Strided convolution layers
- convolution layer of 64 depth with 3x3 filter
- convolution layer of 64 depth with 3x3 filter

- Flatten layer

3 fully connected layers (with dropout layers after 1st and 3rd fully connected layer)
- 100 neurons
- 50% drop out layer
- 50 neurons
- 10 neurons
- 50% drop out layer


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center to road if drifts off and get off the road. These images show what a recovery looks like starting from left side of the road to the center :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process in clockwise direction to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help in generalizing the model well. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

And augmentation step also contains random shearing and noise addition

![alt text][image8]
![alt text][image9]


After the collection process, I had 41361 number of data points. I then preprocessed this data by above mentioned augmentation technique which increased the data points to 124083.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by mean squared error loss. 

```sh
$ python model.py

Using TensorFlow backend.
length of samples 13787
Epoch 1/5
99261/99261 [==============================] - 1101s - loss: 0.0217 - val_loss: 0.0111
Epoch 2/5
99261/99261 [==============================] - 1095s - loss: 0.0153 - val_loss: 0.0119
Epoch 3/5
99261/99261 [==============================] - 1100s - loss: 0.0130 - val_loss: 0.0064
Epoch 4/5
99261/99261 [==============================] - 1102s - loss: 0.0113 - val_loss: 0.0069
Epoch 5/5
99261/99261 [==============================] - 1104s - loss: 0.0105 - val_loss: 0.0111
```

I used an adam optimizer so that manually training the learning rate wasn't necessary.
