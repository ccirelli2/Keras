# DOCUMENTATION ------------------------------------------------
'''
State:      Keras is able to save the model weights in an HDF5 grid format. 
            The model structure can be saved and loaded at a later time. 

JSON        Keras provides the ability to explain any model using JSON format. 
            Call model_from_json()
'''

# LOAD LIBRARIES -----------------------------------------------

import numpy as np
import os

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
def sequential_model():
    model   = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    # Compile Model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Model Fit
    model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=0)

    # Evaluate Model 
    print('Training Model')
    score = model.evaluate(X, Y, verbose=0)
    print('Score => {}'.format(score[1]))

    return model

model = sequential_model()



# SERIALIZE MODEL TO JSON -----------------------------------
os.chdir(dir_output)
model_json  = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')
print('Model saved to disk')



# NOW LOAD THE MODEL BACK INTO MEMORY ----------------------
'Note:  Model from json was a keras module that we loaded at the beg of the script'

# Load Model
json_file           = open('model.json', 'r')
loaded_model_json   = json_file.read()
json_file.close()
loaded_model        = model_from_json(loaded_model_json)

# Load Weights 
loaded_model.load_weights('model.h5')

# Evaluate Loaded Model on Test Data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print('\nLoaded model score => {}'.format(score[1]))













