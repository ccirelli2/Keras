# KERAS -----------------------------------------------------------
'''
Tutorial            https://elitedatascience.com/keras-tutorial-deep-learning-in-python
Documentation:      https://keras.io/

Data type:          int to float32
                    For all weights and neuron activations, if you are using a 
                    method based on backpropagation for training updates, 
                    then you need a data type that approximates real numbers, 
                    so that you can apply fractional updates based on differentiation. 
                    Best weight values are often going to be fractional, non-whole 
                    numbers. Non-linearities such as sigmoid are also going to 
                    output floats. So after the input layer you have matrices of 
                    float values anyway. There is not much speed advantage 
                    multiplying integer matrix with float one (possibly even 
                    slightly slower, depending on type casting mechanism). 
                    So the input may as well be float.
                    https://datascience.stackexchange.com/questions/13636/
                    neural-network-data-type-conversion-float-from-int
'''

# IMPORT PYTHON MODULES -------------------------------------------
import numpy as np
np.random.seed(123)

# Import Feed Forward CNN
from keras.models import Sequential

# Import Core Layer Types (core because they are used in almost every NN)
from keras.layers import Dense, Dropout, Activation, Flatten

# Import Keras CNN Layers 
from keras.layers import Convolution2D, MaxPooling2D

# Import Utilities
from keras.utils import np_utils

# Import Matplotlib
from matplotlib import pyplot as plt

# Import Opencv
import cv2

# LOAD DATASET -----------------------------------------------------

# MNIST dataset
'https://en.wikipedia.org/wiki/MNIST_database'
from keras.datasets import mnist

# Load Train / Test SPlit
'Why tuples?'
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print Shape (60,000 samples, 28x28 pixel images)
'''
print('Training set shape => {}'.format(X_train.shape))
'''

# Show Image
'''
cv2.imshow('Training Example', X_train[0])
cv2.waitKey(0)
'''


# PREPROCESS DATA --------------------------------------------------
'''
Docs:       When using Theano backend, you must explicitly declare a dimension for the depth
            of the input image. Full color with all 3 RGB channels will have depth of 3. 
            The data from MNIST will have depth of 1. 

np.reshape  (array to reshape, newshape?, depth, row, col)
'''

# Reshape Training Data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test  = X_test.reshape(X_test.shape[0], 1, 28, 28)
'X_train.shape = (60000, 1, 28, 28)'

# Convert to Float32i (notice that the current data type is numpy.unit8')
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')

# Normalize Data (Point?)
X_train /= 255
X_test  /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
'Is this because the Softmax will return a 10x1 matrices?'
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# DECLARE MODEL ARCHITECTURE ----------------------------------------

# Define Model
'''
core.layers     https://keras.io/layers/core/
num.convs       https://keras.io/layers/convolutional/
input_shape     should be the shape of one input.  In this case its 1,28,28
32              Refers to the number of convolutions
3,3             Number of rows and cols of the kernel used for the convolution
                So we are going from a 28x28 to 3x3 matrix?

'''
model = Sequential()
model.add(Convolution2D(32, 3, 3, activations='relu', input_shape(1,28,28)))


# Add More Layers
model.add(Convolution2D(32, 3, 3, activation='relu')) # why do we add another convolution of 32?
model.add(MaxPooling2D(pool_size=(2,2)))              # so we are going to get a 2x2 matrix?
model.add(Dropout(0,25))                              # When we add layers, are they sequential?












