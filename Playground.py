"""
Created on Sat Sep 30 13:24:17 2017

@author: KarimM
"""
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re

import json

#with open("Google Drive/PhD/Courses/Deep Learning/Project/train.json") as json_data:
#    d = json.load(json_data)

dataDir = '/Users/KarimM/GoogleDrive/PhD/Courses/Deep_Learning/Project/data/squad/'
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100

with open(dataDir + 'train.ids.context') as f:
    contexts = f.readlines()

with open(dataDir + 'train.ids.question') as f:
    queries = f.readlines()
  
answerSpan = np.zeros(len(queries,2))
with open(dataDir + 'train.span') as f:
    for line in f.readlines():
        answerSpan = f.readlines()
    
Max_Context_Length = max([len(i) for i in contexts])
Max_Query_Length = max([len(i) for i in queries])