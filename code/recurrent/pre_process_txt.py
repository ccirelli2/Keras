


# IMPORT LIBRARIES -----------------------------------------------------

# Python
import os
import numpy as np
import pandas as pd
import string


# NLTK
from nltk.tokenize import RegexpTokenizer as rtk

# Enchant
import enchant 
dict_en = enchant.Dict('en_US')


# LOAD FILE ------------------------------------------------------------
afile       = r'alice_wonderland.txt'
dir_data    = r'/home/ccirelli2/Desktop/repositories/Keras/data'
raw_text    = open(dir_data + '/' + afile).read().lower()


# PRE PROCESS TEXT -----------------------------------------------------
punct           = string.punctuation + '0123456789'
txt_nopunct     = raw_text.translate({ord(i) : None for i in punct})
txt_tokenized   = txt_nopunct.split(' ')
txt_en_words    = []

def get_only_english_words(txt_tokenized):

    txt_en_words = []
    
    try:
        for word in txt_tokenized:
            if dict_en.check(word.strip()) is True:
                txt_en_words.append(word.strip())
    except ValueError:
        pass

    return txt_en_words 



test = get_only_english_words(txt_tokenized)

print(test)



