# DOCUMENTATION ---------------------------------------------------------
'''
Description:        Tutorial from book "Develop Deep Learning Models w/ Keras" Jason Browlee
Dataset:            Pima Indians, onset of diabetes, binary classification problem. 
Dense:              Class used to define fully connected layers
                    First argument is the number of layers (Dense(12,


'''

# LOAD LIBRARIES ---------------------------------------------------------

# Python libraries
import numpy as np
import pandas as pd


# Keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Sklearn
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import LabelEncoder




# LOAD DATASET ---------------------------------------------------------

# Set Seed
seed = 7
np.random.seed(seed)

# Dataset
dir_data    = r'/home/ccirelli2/Desktop/repositories/Keras/data'
afile       = r'pima-indians-diabetes.csv' 
dataset     = pd.read_csv(dir_data + '/' + 'ionosphere.csv', header=None).values

# X / Y Split
X = dataset[:, 0:34].astype(np.float64)
Y = dataset[:, 34]

# Encode Target
encoder     = LabelEncoder()
encoder.fit(Y)
Y           = encoder.transform(Y)



# CREATE MODEL ---------------------------------------------------------
def perceptron(X, Y):

    # Model (input_dim refers to nu cols)
    model = Sequential()
    model.add(Dense(34, input_dim = 34, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Features
    epochs  = 50
    lr      = 0.1
    decay   = lr / epochs
    momentum= 0.8

    # Stochastic Gradient Decent
    sgd = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)
    
    # Module Compile
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Model Fit
    model.fit(X, Y, validation_split=0.33, nb_epoch=epochs, batch_size=28, verbose=2)

    # Return model
    return model



