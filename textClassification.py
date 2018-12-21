#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:50:34 2018

@author: KushDani
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

imdb = keras.datasets.imdb

#load the data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#number of training reviews and trainnig labels
print('Training entries: {}, labels: {}'.format(len(train_data),len(train_labels)))

#a dictionary mapping words to an integer index
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()} #maps keys to values using dictionary
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 #unknown
word_index["<UNUSED"] = 3

#the first indices are reversed using a dicitonary
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_review(text):
    #turn word indexes into array and join with space to make sentences
    #'?' is default value in case key does not exist
    return ' '.join([reverse_word_index.get(i,'?') for i in text])

#print(decode_review(train_data[0]))

strang=''
for i in range(0,50):
    strang += decode_review(train_data[0])[i]
print(strang)

train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256, padding='post',
                                                        value=word_index["<PAD>"])

test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256, padding='post',
                                                     value=word_index["<PAD>"])

#print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

#print(model.summary())

#configure model for training
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#check accuracy of model on data it hasn't seen before; set aside reviews from training for this
x_validation_set = train_data[:10000]
partial_x_train = train_data[10000:]

y_validation_set = train_labels[:10000]
partial_y_train = train_labels[10000:]

print("Train on 15000 samples, validate on 10000 samples")
#train model for given number fo epochs
history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=40 , verbose=1,
                    validation_data=(x_validation_set, y_validation_set))

#see how model performs
results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history
#4 different keys were measured
print(history_dict.keys())

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

#Training and Validation Loss graph
plt.plot(epochs, loss, 'bo', label='Training loss') #'bo' is for 'blue dot'
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#clear figure
plt.clf() 

#Training and Validation Accuracy graph
plt.plot(epochs, acc, 'cs', label ='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
