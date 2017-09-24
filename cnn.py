#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:39:02 2017

@author: elementique
"""

# make theano to use GPU
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import theano
import keras

# Part 1 Building Convolutional neural network
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# Most common approach is to use 32 feature detectors/filters in CNN
# We use filters 3x3 for convolution
# We use input shape of 64x64 pixels and 3 channels. Our Images are 256x256 pixels RGB, but due to memory consumption we should reduce it to 64
# We use the rectifier activation funciton to remove negative pixel values in the feature map and get nonlinearity
classifier.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# We redeuce the complexity of our model without reducing its performance
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#we add second convolutional layers, and max pooling afterwards
classifier.add(Conv2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#we can also add third convolutional layer an double the number of feature detectors/filters

# Step 3 - Flattening - We have created our input layer! :)

classifier.add(Flatten())

# Step 4 - Full connection. 

#128 hidden nodes in the hidden layer is the common number that shows good performance
# Sigmoid activation function for binary outcome in the outcome layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# Keras: Image preprocessing/argumentation to avoid overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', 
                                                 target_size=(64, 64), 
                                                 batch_size=32, 
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
# prefered steps_per_epoch = 8000
# prefered epochs = 25 or more 
# For testing pusposes I fewer steps to be less time-consuming
classifier.fit_generator(training_set,
                    steps_per_epoch=1000,
                    epochs=5,
                    validation_data=test_set,
                    validation_steps=2000)

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
#As we have colored image, we convert image to an array 64x64x3, where 3 is the for R, G and B channels
test_image = image.img_to_array(test_image) 
#We add one more dimension to our 3-dimensional array
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:  # - check the result (first line and first column in result array) 
    prediction = 'dog'
    print('It is a', prediction,'!')
else:
    prediction = 'cat'
    print('It is a', prediction,'!')
