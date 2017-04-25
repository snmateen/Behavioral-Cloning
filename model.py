# load modules
import numpy as np
import os, csv 
import matplotlib.image as mpimg
import numpy as np

# load keras modules
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, MaxPooling2D
from keras.preprocessing.image import random_shear

# read lines from the driving log file
samples = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)

	for line in reader:
		samples.append(line)

print("length of samples", len(samples))

# load sklearn modules for train test split
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# dimensions
rows, cols, depth = 160, 320, 3 
batch_size = 64
corrections = [0, 0.2, -0.2]

from sklearn.utils import shuffle
def generator(samples, batch_size=32):
	"""
	This function defination generates training features and lables for given samples and batch size
	samples - list of lists that stores images, steering angles, speed etc.,

	parse through the list and read images (center, left and right camera) and steering angles with (0, 0.2, -0.2) corrections

	images are augmented using following methods
	- image flip
	- addition of noise
	- random shear
	"""
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:

				for i, correction in zip(range(3), corrections):

					# read image and steering angle + correction (depending upon center, left or right camera image)
					name = './data/IMG/' + batch_sample[i].split('\\')[-1]
					image = mpimg.imread(name)
					angle = float(batch_sample[3]) + correction

					# original image
					images.append(image)
					angles.append(angle)

					# augmentation
					# flip image
					images.append(np.fliplr(image))
					angles.append(angle * -1.0)

					# add noise
					images.append(image - 0.35)
					angles.append(angle)

					# random shear
					images.append(random_shear(image, np.random.randint(10)))
					angles.append(angle)
			
			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)

			yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(rows,cols,depth)))

# crop top 65 and bottom 25 pixels
model.add(Cropping2D(cropping=((65,25), (0,0))))

# add 5 convolution layers
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))

# flatten
model.add(Flatten())

# add 3 fully connected layers
# with 50% drop out to avoid overfitting
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))

# linear activated final layer
model.add(Dense(1))

# compile the model
model.compile(loss='mse', optimizer='adam')

# train and validate
# total number of epochs = 5
model.fit_generator(train_generator
                    , samples_per_epoch= 12 * len(train_samples)
                    , validation_data=validation_generator
                    , nb_val_samples=len(validation_samples)
                    , nb_epoch=2)

# save the model
model.save('model.h5')
