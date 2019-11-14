# DOCUMENTATION -------------------------------------------
'''
Url:        https://keras.io/examples/lstm_text_generation/
Tutorial:   https://machinelearningmastery.com/
            text-generation-lstm-recurrent-neural-networks-python-keras/
Obj:        We are going to learn the dependencies between characters and 
            the conditional probabilities of characters in sequences so that 
            we can in turn generate wholly new and original sequences of characters.
'''

# IMPORT LIBRARIES ------------------------------------------------

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


# LOAD DATA --------------------------------------------------------

# Load Text File
afile       = r'alice_wonderland.txt'
dir_data    = r'/home/ccirelli2/Desktop/repositories/Keras/data'
raw_text        = open(dir_data + '/' + afile).read().lower().replace('\n', '')


# MODEL -----------------------------------------------------------
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# LOADING WEIGHTS -------------------------------------------------
filename    = 'weights-improvement-01-2.9477.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# Create Reverse Mapping of Characters (Integers to Characters)
int_to_char = dict((i,c) for i, c in enumerate(chars))



# MAKE PREDICTIONS -----------------------------------------------
'''
Process:        First start off with a seed sequence as input
                Then, generate the next character, update the seed sequence to add the
                generated character
                tip off the first character of the sequence
'''

# pick a random seed
'''dataX right now is a list of lists within which are enumerated words
'''
start   = np.random.randint(0, len(dataX)-1)
pattern  = dataX[start] # random start to sequence

# Decode first sequence 
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")













