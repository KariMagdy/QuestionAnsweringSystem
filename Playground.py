"""
Created on Sat Sep 30 13:24:17 2017

@author: KarimM
"""
from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Dropout, concatenate
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.platform import gfile
from os.path import join as pjoin
from keras.layers import Embedding,Bidirectional,LSTM



dataDir = '/Users/KarimM/GoogleDrive/PhD/Courses/Deep_Learning/Project/data/squad/'
GLOVE_DIR = '/Users/KarimM/data/glove/'
EMBEDDING_DIM = 100
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.2


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)
        
vocab, rev_vocab = initialize_vocabulary(pjoin(dataDir, "vocab.dat"))

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# Reading the tekenized contexts
contexts = []
with open(dataDir + 'train.ids.context') as f:
    for line in f.readlines():
        contexts.append(map(int,line.split()))

# Reading the tokenized queries
queries = []
with open(dataDir + 'train.ids.question') as f:
    for line in f.readlines():
        queries.append(map(int,line.split()))
 
# Reading the answer spans 
answerSpan = np.zeros([len(queries),2],dtype = int)
with open(dataDir + 'train.span') as f:
    for idx,line in enumerate(f.readlines()):
        answerSpan[idx,:] = map(int , line.split())
    
Max_Context_Length = max([len(i) for i in contexts])
Max_Query_Length = max([len(i) for i in queries])

print('Found %s contexts.' % len(contexts))


contexts = pad_sequences(contexts, maxlen=Max_Context_Length)
queries = pad_sequences(queries, maxlen=Max_Query_Length)

print('Shape of context tensor:', contexts.shape)
print('Shape of query tensor:', queries.shape)
print('Shape of answers tensor:', answerSpan.shape)


# prepare embedding matrix [[[Use all vocab????]]]
num_words = len(vocab)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in vocab.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
# load pre-trained word embeddings into an Embedding layer
embedding_layer_Context = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=Max_Context_Length,
                            trainable=False)

embedding_layer_Query = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=Max_Query_Length,
                            trainable=False)

# Label for start as hotvector [[[Edit for start and end properly]]]
labels = np.zeros([len(contexts),Max_Context_Length])
for idx, start in enumerate(answerSpan[:,0]):
    labels[idx,start] = 1

# split the data into a training set and a validation set
indices = np.arange(contexts.shape[0])
np.random.shuffle(indices)
contexts = contexts[indices]
queries = queries[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * contexts.shape[0])

contexts_train = contexts[:-num_validation_samples,:]
queries_train = queries[:-num_validation_samples,:]
labels_train = labels[:-num_validation_samples]
contexts_val = contexts[-num_validation_samples:,:]
queries_val = queries[-num_validation_samples:,:]
labels_val = labels[-num_validation_samples:]

print('Training model.')

# train a 1D convnet with global maxpooling
Context_input = Input(shape=(Max_Context_Length,), dtype='int32')
Query_input = Input(shape=(Max_Query_Length,), dtype='int32')
embedded_Context = embedding_layer_Context(Context_input)
embedded_Query = embedding_layer_Query(Query_input)

# LSTM layer for both question and query
contextLstm = Bidirectional(LSTM(64))(embedded_Context)
queryLstm = Bidirectional(LSTM(64))(embedded_Query)


# Concatenating output and input and passing to a dense layer
aggregated = concatenate([contextLstm,queryLstm],axis = -1)
aggregated = Dense(1000, activation='relu')(aggregated)
answer = Dense(Max_Context_Length)(aggregated)
answer = Activation('softmax')(answer)
model = Model([Context_input, Query_input], answer)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
history = model.fit([contexts_train, queries_train], labels_train,
          batch_size=128,
          epochs=2,validation_data=([contexts_val, queries_val], labels_val))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


"""
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
validation_data=(x_val, y_val))
"""