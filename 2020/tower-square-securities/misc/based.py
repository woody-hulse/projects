from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np
import re
import random as rand

train = False

# get poems
data = open('marx.txt', encoding="utf8").read()
# data = re.sub(r'[^\w\s]', '', data)
text = data.lower().split("\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
word_count = len(tokenizer.word_index) + 1

input_sequences = []
for line in text:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        seq = token_list[:i + 1]
        input_sequences.append(seq)

# pad sequences
longest_seq = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=longest_seq, padding='pre'))
# create train and target set
x, y = input_sequences[:, :-1], input_sequences[:, -1]
y = ku.to_categorical(y, num_classes=word_count)

# build model
model = Sequential()
model.add(Embedding(word_count, 128, input_length=longest_seq-1))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(word_count/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(word_count, activation='softmax'))

# embed for text data
# model.add(Embedding(1000, 64, input_length=10))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('based.h5')

if train:
    # fit model
    history = model.fit(x, y, epochs=25, verbose=1)

    model.save_weights('based.h5')

# generate new poems
for i in range(1):
    seed_text = text[rand.randint(0, len(text) - 1)]
    next_words = rand.randint(25, 100000)
    based = "   "

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=longest_seq - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        based += " " + output_word

        if _ % 1000 == 0:
            based += ".\n\n" + str(int(i / 1000 + 1)) + "\n\n   "

    print(based)
    with open("book.txt", "w") as out:
        out.write(based)
