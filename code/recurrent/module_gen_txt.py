from __future__ import print_function
import numpy as np
import pandas as pd
import random
import os
import io
import sys
import enchant
dict_en = enchant.Dict('en_US')


# Keras
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop




def load_txt(afile, dir_data):

    with io.open(dir_data + '/' + afile, encoding='utf-8') as f:
        text = f.read().lower()
        return text


def cut_txt_sequences(text):
    maxlen      = 40
    step        = 3
    sentences   = []
    next_char   = []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_char.append(text[1 + maxlen])
    return sentences, next_char




def get_only_english_words(txt_tokenized):

    txt_en_words = []

    try:
        for word in txt_tokenized:
            if dict_en.check(word.strip()) is True:
                print(word)
                txt_en_words.append(word.strip())
    except ValueError:
        pass

    
    return ' '.join(txt_en_words)




