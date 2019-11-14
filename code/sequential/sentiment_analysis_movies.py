# DOCUMENTATION -------------------------------------------------
'''
Objective:      Determine whether the movie has a negative or positive sentiment
Dataset:        IMDB Large Movie Reviews
                X:  It looks like they already converted the text of each review into a list
                    of tokens that were converted to numbers. Not clear what pre-processing
                    they did.
                Y:  Binary value [0,1]

'''


# LIBRARIES -----------------------------------------------------

# Python
import numpy as np
import matplotlib.pyplot as plt

# Keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers embeddings import Embedding
from keras.preprocessing import sequence


# SET SEED -------------------------------------------------------
seed = 7
np.random.seed(seed)

# LOAD DATASET ---------------------------------------------------
'''
Limit vocab:    Only load the top 5,000 words imdb.load_data(nb_words=5000)
concatenate:    just combining the two arrays.  see X.shape

'''

# Dataset Load Restrictions
top_words   = 5000
test_split  = 0.33
max_words   = 500

# Train Test Split
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

# Truncate/Pad Longer/Shorter Reviews 
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test  = sentiment.pad_sequences(X_test, maxlen=max_words)

# Create a Single Vector of Tokens
X   = np.concatenate((X_train, X_test), axis=0)
Y   = np.concatenate((y_train, y_test), axis=0)


# MODEL ---------------------------------------------------------
'''
Architecture    First layer will be an Embedding Layer. 
                Dims = 5,000 words, 32 dimensions and input_lenth 500

'''















