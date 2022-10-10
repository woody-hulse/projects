import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
import random as r

train = True

# get text
poems = (open("poems2.txt").read())
poems = poems.lower()

'''
# remove some characters
poems = poems.replace("(", "")
poems = poems.replace("-", " ")
poems = poems.replace(")", "")
'''

# character integer dicts
characters = sorted(list(set(poems)))
n_to_char = {n: char for n, char in enumerate(characters)}
char_to_n = {char: n for n, char in enumerate(characters)}

char_to_n = {'\n': 0, ' ': 1, '!': 2, "'": 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, ':': 9, ';': 10, '?': 11,
             'a': 12, 'b': 13, 'c': 14, 'd': 15, 'e': 16, 'f': 17, 'g': 18, 'h': 19, 'i': 20, 'j': 21, 'k': 22,
             'l': 23, 'm': 24, 'n': 25, 'o': 26, 'p': 27, 'q': 28, 'r': 29, 's': 30, 't': 31, 'u': 32, 'v': 33,
             'w': 34, 'x': 35, 'y': 36, 'z': 37, '—': 38, '‘': 39, '’': 40}

n_to_char = {0: '\n', 1: ' ', 2: '!', 3: "'", 4: '(', 5: ')', 6: ',', 7: '-', 8: '.', 9: ':', 10: ';', 11: '?',
             12: 'a', 13: 'b', 14: 'c', 15: 'd', 16: 'e', 17: 'f', 18: 'g', 19: 'h', 20: 'i', 21: 'j', 22: 'k',
             23: 'l', 24: 'm', 25: 'n', 26: 'o', 27: 'p', 28: 'q', 29: 'r', 30: 's', 31: 't', 32: 'u', 33: 'v',
             34: 'w', 35: 'x', 36: 'y', 37: 'z', 38: '—', 39: '‘', 40: '’'}

print(len(char_to_n))
print(n_to_char)

for char in poems:
    try:
        char_to_n[char]
    except:
        poems = poems.replace(char, " ")

# fix data
X = []
Y = []
length = len(poems)
seq_length = 100

for i in range(length - seq_length):
    sequence = poems[i:i + seq_length]
    label = poems[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

# create model
model = Sequential()
model.add(LSTM(700, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

if train:
    # model.load_weights('bigfrost.h5')

    model.fit(X_modified, Y_modified, epochs=4, batch_size=50)
    model.save_weights('bigfrost2.h5')

# load model
# model.load_weights("text_generator_400_0.2_400_0.2_400_0.2_100.h5")
model.load_weights('bigfrost2.h5')

# generate text
# string_mapped = X[99]

# create random seed
string_mapped = []
for i in range(100):
    string_mapped.append(r.randint(0, len(n_to_char) - 1))

full_string = [n_to_char[value] for value in string_mapped]
# generating characters
for i in range(200):
    x = np.reshape(string_mapped, (1, len(string_mapped), 1))
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

# combining text
text = ""
for char in full_string:
    text += char

print(text)
