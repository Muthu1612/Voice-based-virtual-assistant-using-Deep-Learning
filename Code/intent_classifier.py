import pandas as pd
print(f"Pandas: {pd.__version__}")
import numpy as np
print(f"Numpy: {np.__version__}")
import tensorflow as tf
print(f"Tensorflow: {tf.__version__}")
from tensorflow import keras
print(f"Keras: {keras.__version__}")
import sklearn
print(f"Sklearn: {sklearn.__version__}")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import collections
import yaml
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
train = pd.read_pickle('objects/train.pkl')
print(f'Training data: {train.head()}')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
X_train, X_val, y_train, y_val = train_test_split(train['Utterance'], train['Intent'], test_size = 0.3, 
                                                   shuffle = True, stratify = train['Intent'], random_state = 7)

print(f'\nShape checks:\nX_train: {X_train.shape} X_val: {X_val.shape}\ny_train: {y_train.shape} y_val: {y_val.shape}')

temp=pd.get_dummies(y_train)

y_train=temp

temp=pd.get_dummies(y_val)

y_val=temp

t = Tokenizer()

t.fit_on_texts(X_train)

print("Document Count: \n{}\n".format(t.document_count))


def convert_to_padded(tokenizer, docs):
    ''' Taking in Keras API Tokenizer and documents and returns their padded version '''

    embedded = t.texts_to_sequences(docs)

    padded = pad_sequences(embedded, maxlen = max_length, padding = 'post')
    return padded


vocab_size = len(t.word_counts) + 1
print(f'Vocab size:\n{vocab_size}')


max_length = len(max(X_train, key = len))

print(f'Max length:\n{max_length}')

padded_X_train = convert_to_padded(tokenizer = t, docs = X_train)
padded_X_val = convert_to_padded(tokenizer = t, docs = X_val)

print(f'padded_X_train\n{padded_X_train}')
print(f'padded_X_val\n{padded_X_val}')


padded_X_train.shape, padded_X_val.shape, y_train.shape, y_val.shape


padded_X_train[1]


embeddings_index = {}
f = open('models/glove.twitter.27B/glove.twitter.27B.50d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


word_index = t.word_index
EMBEDDING_DIM = 50 


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        
        embedding_matrix[i] = embedding_vector


embedding_matrix, embedding_matrix.shape


y_train.shape


padded_X_train.shape


import tensorflow as tf


class ryz(tf.keras.losses.Loss):
    def __init__(self, batch_size: int = 32):
        super(ryz, self).__init__()
        self.batch_size = batch_size

    def call(self, y_true, y_pred):
        tmp = []
        for item in range(self.batch_size):
            tf.print(f'Working on batch: {item}\n')
            s = y_true[item, :]
            t = y_pred[item, :]
            n, m = len(s), len(t)
            dtw_matrix = []
            for i in range(n + 1):
                line = []
                for j in range(m + 1):
                    if i == 0 and j == 0:
                        line.append(0)
                    else:
                        line.append(np.inf)
                dtw_matrix.append(line)

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = tf.abs(s[i - 1] - t[j - 1])
                    last_min = tf.reduce_min([dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1]])
                    dtw_matrix[i][j] = tf.cast(cost, dtype=tf.float32) + tf.cast(last_min, dtype=tf.float32)

            temp = []
            for i in range(len(dtw_matrix)):
                temp.append(tf.stack(dtw_matrix[i]))

            tmp.append(tf.stack(temp)[n, m])
        
        cce = tf.keras.losses.CategoricalCrossentropy()
        error=cce(y_true, y_pred)
        check=tf.math.greater(error,tmp)
        if check==True:
          return tf.reduce_mean(tmp)
        else:
          return tf.reduce_mean(error)


import keras.backend as K
def get_f1(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


y_val = y_val.astype('float32')
y_train = y_train.astype('float32')


def make_model(vocab_size, max_token_length):
    ''' In this function I define all the layers of my neural network'''
    
    model = Sequential()
    #model.add(Input(shape = (32,), dtype = 'int32'))

    
    model.add(Embedding(vocab_size, embedding_matrix.shape[1], input_length = 32, 
                        trainable = False, weights = [embedding_matrix]))
    
    model.add(Bidirectional(LSTM(128,return_sequences= True)))
    model.add(LSTM(128)) 
    
    model.add(Dense(600, activation = "tanh",kernel_regularizer ='l2')) 
    
    
    # model.add(Dense(600, activation = "relu",kernel_regularizer ='l2'))
    model.add(Dense(300, activation = "tanh",kernel_regularizer ='l2'))
    model.add(Dense(150, activation = "tanh",kernel_regularizer ='l2'))
    # model.add(Dense(50, activation = "tanh",kernel_regularizer ='l2'))
    # model.add(Dense(64, activation = "relu",kernel_regularizer ='l2'))
    
    model.add(Dropout(0.5))
    

    model.add(Dense(11, activation = "sigmoid"))
    
    return model


model = make_model(vocab_size, 32)
model.compile(loss = ryz(batch_size=1), 
              optimizer = "rmsprop", metrics = ["accuracy"])
model.summary()


filename = 'models/intent_classification.h5'


def scheduler(epoch, lr):
    if epoch < 12:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_sched_checkpoint = tf.keras.callbacks.LearningRateScheduler(scheduler)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)



checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')


hist = model.fit(padded_X_train, y_train, epochs = 20, batch_size = 32, 
                 validation_data = (padded_X_val, y_val), 
                 callbacks = [checkpoint, lr_sched_checkpoint, early_stopping])


# plt.figure(figsize=(10,7))
# plt.plot(hist.history['val_loss'], label = 'Validation Loss', color = 'cyan')
# plt.plot(hist.history['loss'], label = 'Training Loss', color = 'purple')
# plt.title('Training Loss vs Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# plt.figure(figsize=(10,7))
# plt.plot(hist.history['val_accuracy'], label = 'Validation Accuracy', color = 'cyan')
# plt.plot(hist.history['accuracy'], label = 'Training Accuracy', color = 'purple')
# plt.title('Training Accuracy vs Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
