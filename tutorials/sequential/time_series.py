# DOCUMENTATION --------------------------------------------







# LOAD PACKAGES -------------------------------------------

# Python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Keras
from keras.models import Sequential
from keras.layers import Dense


# LOAD DATA -----------------------------------------------
afile       = r'international_airline_passengers.csv'
dir_data    = r'/home/ccirelli2/Desktop/repositories/Keras/data'
df = pd.read_csv(dir_data + '/' + afile, sep=';', usecols=[1], skiprows=1, engine='python').values
df = df.astype('float32')

# Train / Test Split
train_size = int(len(df) * 0.67)
test_size = len(df) - train_size
train, test = df[0:train_size, :], df[train_size:len(df), :]

# Create df w/ Target Col
def create_dataset(df, look_back=1):
    dataX, dataY = [], []
    for i in range(len(df)-look_back -1):
        a = df[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(df[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back =1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# CREATE MODEL -------------------------------------------

model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=0)

# Estimate Performance
trainScore  = model.evaluate(trainX, trainY, verbose=0)
print('Train Score => {}'.format(trainScore))

testScore   = model.evaluate(testX, testY, verbose=0)
print('Test Score  => {}'.format(testScore))


# Generate Prediction
model_predict = model.predict(trainX)
df_prediction = pd.DataFrame({})

plt.plot(model_predict)
plt.plot(testX)
plt.show()




















