# DOCUMENTATION ----------------------------------------------------
'''
Docs		https://scikit-learn.org/stable/modules/generated/
		sklearn.datasets.fetch_20newsgroups.html#
User Guide	https://scikit-learn.org/stable/datasets/index.html#newsgroups-dataset
Categories:	newsgroups_train.target_names


Bunch Object:
data:		list, length [n_samples]
target: 	array, shape [n_samples]
filenames: 	list, length [n_samples]
DESCR: 		a description of the dataset.
target_names: 	a list of categories of the returned data, length [n_classes]. This depends on the 
		categories parameter.


'''





# LOAD LIBRARIES --------------------------------------------------

# Python
import pandas as pd
import numpy as np
import os
from pprint import pprint


# Scikit Learn
from sklearn.datasets import fetch_20newsgroups


# DEFINE DIRECTORIES ----------------------------------------------
dir_data	= r'/home/ccirelli2/Desktop/repositories/Keras/practice_projects/sklearn_newsgroups/data'


# IMPORT DATA ------------------------------------------------------
'''
'''
atheism		= fetch_20newsgroups(data_home=dir_data, subset='all', shuffle=True, random_state=1, 					categories=['alt.atheism'])
motorcycles	= fetch_20newsgroups(data_home=dir_data, subset='all', shuffle=True, random_state=1,					categories=['rec.motorcycles'])


print(atheism['data'][0])
print('******************')
print(motorcycles['data'][0])




