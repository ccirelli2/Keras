# DOCUMENTATION -----------------------------------------
'''
Time series 
predictions:    Has a sequence dependency among the input 
                variables. 
LSTM:           RNN that can learn these dependencies. 
Data:           LSTMs are sensitive to scal fo the input data. 
                Therefore, should rescale data. 
Input:          Must be [samples, time steps, features].  
                airlines dataset is in the form [samples, features]

'''


# LOAD PACKAGES -------------------------------------------

# Python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math


# Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



# DATA --------------------------------------------------------

# Create df w/ Target Col
def create_dataset(df, look_back=1):
    dataX, dataY = [], []
    for i in range(len(df)-look_back -1):
        a = df[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(df[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# Set Seed
seed = 7
np.random.seed(seed)

# Load Data
afile       = r'international_airline_passengers.csv'
dir_data    = r'/home/ccirelli2/Desktop/repositories/Keras/data'
df = pd.read_csv(dir_data + '/' + afile, sep=';', usecols=[1], skiprows=1, engine='python').values
df = df.astype('float32')

# Normalize Dataset

scaler  = MinMaxScaler(feature_range=(0,1))
df      = scaler.fit_transform(df)

# Split Train / Test
train_size  = int(len(df) * 0.67)
test_size   = len(df) - train_size
train, test = df[: train_size, :], df[train_size:,:] 

# Reshape into X=t and Y=t+1
look_back   = 1
trainX, trainY  = create_dataset(train, look_back)
testX, testY    = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX  = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
test    = np.reshape(testX,  (testX.shape[0], 1, testX.shape[1]))


# CREATE & FIT LSTM MODEL ---------------------------------------
'''LSTM(4 <= I think this refers to the number of LSTM neurons. 
'''
model   = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=10, batch_size=1, verbose=2)

# Make Predictions
trainPredict    = model.predict(trainX)











































