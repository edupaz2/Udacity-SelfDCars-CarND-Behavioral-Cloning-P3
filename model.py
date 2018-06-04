import csv
import os
import cv2
import numpy as np
import tensorflow as tf

# Constant values
EPOCHS = 7
BATCH_SIZE=128

def composeNewFilePath(new_path, oldfilepath):
	return new_path + oldfilepath.split('\\')[-1]

def extractDataSet(datasetpath):
	print('Extracting dataset:', datasetpath)
	lines = []
	try:
		with open(datasetpath+'/driving_log.csv') as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				lines.append(line)
	except FileNotFoundError:
		pass

	print(len(lines),'lines read.')
	paths = []
	angles = []
	current_outpath = datasetpath + '/IMG/'
	for line in lines:
		steering_center = float(line[3])

		# create adjusted steering measurements for the side camera images
		correction = 0.2 # this is a parameter to tune
		steering_left = steering_center + correction
		steering_right = steering_center - correction

		# read in images from center, left and right cameras
		img_center = composeNewFilePath(current_outpath, line[0])
		img_left = composeNewFilePath(current_outpath, line[1])
		img_right = composeNewFilePath(current_outpath, line[2])

		# add images and angles to data set
		paths.append(img_center)
		paths.append(img_left)
		paths.append(img_right)

		angles.append(steering_center)
		angles.append(steering_left)
		angles.append(steering_right)

	return paths, angles

from keras.models import Sequential
from keras.layers import Lambda, Dense, Activation, Flatten, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def LeNet():
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(6, 5, 5, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6, 5, 5, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model

def Nvidia():
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
	model.add(Convolution2D(64,3,3,activation='relu'))
	model.add(Convolution2D(64,3,3,activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

from sklearn.utils import shuffle
def myGenerator(set_x, set_y, batch_size=BATCH_SIZE):
	while True:
		shuffle_x, shuffle_y = shuffle(set_x, set_y)
		for offset in range(0, len(set_x), batch_size):
			batch_x, batch_y = set_x[offset:offset+batch_size], set_y[offset:offset+batch_size]
			# Read and preprocess images in here for efficiency
			batch_images = []
			batch_angles = []
			for i in range(len(batch_x)):
				img = cv2.imread(batch_x[i])
				angle = batch_y[i]
				batch_images.append(img)
				batch_angles.append(angle)
				# Augment data by mirroring:
				batch_images.append(cv2.flip(img, 1))
				batch_angles.append(angle*-1.0)
				
			yield (np.array(batch_images), np.array(batch_angles))

### Main
images_path = []
steering_angles = []

# Extract the datasets. Generating two lists:
# images_path with the filepath of each image
# steering_angles with the steering angle of each image
print('-- Extracting datasets.')
for dataset in os.listdir('data'):
	new_paths, new_steering_angles = extractDataSet('data/'+dataset)
	images_path.extend(new_paths)
	steering_angles.extend(new_steering_angles)

	print(dataset, len(new_paths), len(new_steering_angles))

print('-- Generating training, validation sets.')

## Split validation dataset off from training dataset
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(images_path, steering_angles, test_size=0.2)

## Samples length adjustment
samples_per_epoch = int(len(X_train)/EPOCHS)
total_batch_size = samples_per_epoch*EPOCHS

train_generator = myGenerator(X_train, y_train)
valid_generator = myGenerator(X_valid, y_valid)

print('Datasets size:', len(images_path))
print('Train samples:', len(X_train))
print('Valid samples:', len(X_valid))
print('samples_per_epoch:', samples_per_epoch)
print('epochs:', EPOCHS)
print('total_batch:', total_batch_size)

# MODEL
print('-- Training')
model = Nvidia()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, validation_data=valid_generator, nb_val_samples=len(X_valid), nb_epoch=EPOCHS, verbose=1)
model.save('model.h5')

# Visualize loss
print('-- Visualize the loss')
from keras.models import Model
import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

print('Loss:', history_object.history['loss'])
print('Val_Loss:', history_object.history['val_loss'])
