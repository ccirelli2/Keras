# TRAIN CONVOLUTIONAL NEURAL NETWORK -------------------------------
'''
categorical cross entropy
                https://peltarion.com/knowledge-center/documentation/modeling-view/
                build-an-ai-model/loss-functions/categorical-crossentropy


Example From Keras Docs
                https://keras.io/examples/mnist_cnn/
'''



# LOAD LIBRARIES ---- -----------------------------------------------

# Python
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# CNN Modules
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# PERCEPTRON FEED FORWARD MODEL ------------------------------------------------------

def perceptron_model():
    seed = 7
    np.random.seed(seed)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    # Change Matrix To An Array
    num_pixels  = X_train.shape[1] * X_train.shape[2]
    X_train     = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    y_train     = y_train.reshape(y_train.shape[0], num_pixels).astype('float32')

    # Normalize Pixels 
    X_train     = X_train / 255
    X_test      = X_test / 255

    # Output Variable (Target, from 0-9) - Change to Binary Matrix
    y_train     = np_utils.to_categorical(y_train)
    y_test      = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    
    # create model
    model   = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    # Compile Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=10, batch_size=200, 
                    verbose=2)
    # Return Model
    return model


# CONVOLUTIONAL NEURAL NETWORK MODEL ---------------------------------------------

def cnn():

    # Load Data
    seed = 7
    np.random.seed(seed)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Data Shape
    ''' Layers used for two dimensional convolutions expect pixel values with
        the dimensions [channels][width][height].  In the case of MNIST the data
        is already gray scale, so channels = 1'''
    
    # (num_imgs, 1 dim, 28 rows, 28 cols)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    
    # Normalize inputs
    X_train = X_train / 255
    X_test  = X_test / 255

    # One Hot Encode Target
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # Build Model
    ''' Border_mode:  https://datascience.stackexchange.com/questions/11840/
        border-mode-for-convolutional-layers-in-keras
        **input_shape: (batch, steps, channels)
        '''
    model   = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(28,28, 1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fit Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, 
                batch_size=200, verbose=2)
    scores  = model.evaluate(X_test, y_test, verbose=0)
    print(scores)


cnn()




