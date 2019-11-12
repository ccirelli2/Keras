# DOCUMENTATION ---------------------------------------------------------
'''
Description:        Tutorial from book "Develop Deep Learning Models w/ Keras" Jason Browlee
Dataset:            Pima Indians, onset of diabetes, binary classification problem. 
Dense:              Class used to define fully connected layers
                    First argument is the number of layers (Dense(12,


'''

# LOAD PYTHON PACKAGES --------------------------------------------------

# Packages
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold 
import numpy

# Set Seed
seed = 7
numpy.random.seed(seed)


# LOAD DATASET ---------------------------------------------------------

# Dataset
dir_data    = r'/home/ccirelli2/Desktop/repositories/Keras/data'
afile       = r'pima-indians-diabetes.csv' 
dataset     = numpy.loadtxt(dir_data + '/' + afile, delimiter=",")
print(dataset.shape)


def perceptron(dataset):
    # X / Y Split
    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    # CREATE MODEL ---------------------------------------------------------

    # Model
    model = Sequential()
    model.add(Dense(12, input_dim = X.shape[1], init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu')) # why an activation at each layer?
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit / Train Model
    ''' Auto Verification:  We add validation_split=0.33 in order to automatically 
                            validate our model
        Manually            We can use Scikit train/test/split and during the fit specify
                            validation_split=(X_test, y_test)
        Verbose             0= basically silent
                            1= verbose
    '''

    #model.fit(X, Y, nb_epoch=150, batch_size=10, validation_split=0.33, verbose=0)

    # Evaluate Model 
    scores = model.evaluate(X, Y)
    print(model.metrics_names[1], scores)


# Example Using Kfold Validation
def kfold_cv(dataset):
    # X / Y Split
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_scores = []
    # Compile

    # Loop Over Folds
    for train, test in kfold.split(X, Y): 
        model = Sequential()
        model.add(Dense(12, input_dim = X.shape[1], init='uniform', activation='relu'))
        model.add(Dense(8, init='uniform', activation='relu')) # why an activation at each layer?
        model.add(Dense(1, init='uniform', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=0)
        scores = model.evaluate(X[test], Y[test], verbose=0)
        cv_scores.append(scores[1]*100)

    print(cv_scores)


kfold_cv(dataset)



