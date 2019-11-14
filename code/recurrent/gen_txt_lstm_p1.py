# DOCUMENTATION -------------------------------------------
'''
Url:        https://keras.io/examples/lstm_text_generation/
Tutorial:   https://machinelearningmastery.com/
            text-generation-lstm-recurrent-neural-networks-python-keras/
Obj:        We are going to learn the dependencies between characters and 
            the conditional probabilities of characters in sequences so that 
            we can in turn generate wholly new and original sequences of characters.
'''

# IMPORT LIBRARIES ----------------------------------------

# Python
from __future__ import print_function
import numpy as np
import pandas as pd
import random
import os
import io
import sys


# Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# Project
import module_gen_txt as m1



# LOAD DATA -----------------------------------------------

# Load Text File
afile       = r'alice_wonderland.txt'
dir_data    = r'/home/ccirelli2/Desktop/repositories/Keras/data'
text        = open(dir_data + '/' + afile).read().lower().replace('\n', '')

# Enumerate Text
'''
Step1:      Create a list of each unique character in the text
Step2:      Create a mapping of each character in the tex to an integer
chars:      may require clean up. 
step3:      Create diction of int, character
'''
chars       = sorted(list(set(text)))
char_to_int = dict((c,i) for i, c in enumerate(chars))

# CREATEA TRAINING DATA
'''
Process:    Split text up into sequences with a fixed lenth of 100 characters (arbitrary)
            Therefore, each training patter of the network is comprised of 100 time steps
            of one character (X) followed by one character output (y).
            Each training pattern of the network is comprised of 100 time steps of one 
            character (X) followed by one character output (y). When creating these sequences, 
            we slide this window along the whole book one character at a time, 
            allowing each character a chance to be learned from the 100 characters that 
            preceded it (except the first 100 characters of course)

Thoughts:   So essentially its learning to predict the next sequence in the text?
            Also, the expectation is that it will output a sequence of 100 characters?

'''

n_chars = len(text)
n_vocab = len(chars)

# Create Sequences 
'''
Process:    We will create sequences of 100 characters in length, converting each token
            to a integer
'''
seq_length  = 100
dataX       = []
dataY       = []
Count       = 0
# Iterate over entire text w/ a step 1.  Stop at n_chars minus seq length
for i in range(0, n_chars - seq_length, 1):
    seq_in  = text[i : i + seq_length] # this creates a seq of len 100
    seq_out = text[i + seq_length]     # this outputs the next char in the sequence
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out]) # does not require list comp as its a single char

n_patterns   = len(dataX)


# TRANSFORM DATA TO FIT SEQUENCE REQUIRED BY KERAS ----------------------------- 
'''
input_shape:    [samples, time steps, features]
dataX:          X data set of sequences
n_patters:      Looks like the number of sequences in training set
seq_lenth:      fixed length of each sequence
1:              dimension of input vector'''

X = np.reshape(dataX, (n_patterns, seq_length, 1))

# Normalize X
''' This is probably important as we are using whole integers whose magnitude has 
    no real relationship with the words. 
'''
X = X/ float(n_vocab)

# One Hot Encode The Output Variable
''' Need to one hot encode so that the system can predict the probability of the number
'''
y = np_utils.to_categorical(dataY)



# DEFINE LSTM MODEL -------------------------------------------------------------
''' 256:        Number of memory units
    Training:   There is no test data since we are modeling the entire dataset to get the
                probability of each character in a sequence. 
                Seeking a balance between generalization, overfitting but short of
                memorization
'''

model   = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0,2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define Checkpoints
'''
Model Checkpoingt:  Used to record all the network weights to file each time an improvement
                    in loss is observed. 
'''
filepath    ="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint  = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='mim')
callbacks_list = [checkpoint]

# Fit Model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)









