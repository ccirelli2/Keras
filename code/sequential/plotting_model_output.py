# DOCUMENTATION ------------------------------------------------
'''
History Callback        Records training metrics for each epoch
                        Returned from calls to the fit() method. 
                        Metrics are stored in a dictionary. 
'''

# LOAD LIBRARIES -----------------------------------------------

import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# DATA --------------------------------------------------------

# Set Seed
seed = 7
np.random.seed(seed)

# Load Data
dir_output  = r'/home/ccirelli2/Desktop/repositories/Keras/output'
dir_data    = r'/home/ccirelli2/Desktop/repositories/Keras/data'
afile       = r'pima-indians-diabetes.csv'
dataset     = np.loadtxt(dir_data + '/' + afile, delimiter=",")
X           = dataset[:, 0:8]
Y           = dataset[:, 8]

# Create Model 
def sequential_model(X, Y):
    model   = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    # Compile Model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Model Fit ***Note you are saving the fit to an object "history"
    history = model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=0)

    # List All Data in History
    print('***** History *****')
    print(history.history.keys())

    # Summarize History for Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()

    '''
    # Evaluate Model 
    print('Training Model')
    score = model.evaluate(X, Y, verbose=0)
    print('Score => {}'.format(score[1]))
    '''
    return model



sequential_model(X, Y)













