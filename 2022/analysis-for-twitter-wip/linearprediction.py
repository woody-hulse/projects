import math
import numpy as np
import plotly.io as pio
import plotly.graph_objects as graph
import matplotlib.pyplot as plt
import random
import os

import h5py
import tensorflow as tf
import pandas as pd
import pandas_datareader
from pandas.tseries.offsets import DateOffset
import keras
from keras import Input, Model
from keras.preprocessing.sequence import TimeseriesGenerator
from pygments.lexers import go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense, Dropout, Activation, LSTM, Convolution1D, MaxPooling1D, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.models import load_model

import os
path = os.path.dirname(__file__)

def predict(data, newModel=False, trainModel=False):

    xdata = range(0, len(data))

    data = [xdata, data]

    #   scale data
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    #   create timeseries to input to neural network
    n_input = int(len(data[0]))
    n_features = 1
    generator = TimeseriesGenerator(data, data, length=n_input, end_index=n_input, batch_size=1)

    #   create and train model
    if newModel:

        model = Sequential()
        model.add(Convolution1D(64, 3, input_shape=(n_input, n_features), border_mode='same'))
        model.add(MaxPooling1D(pool_length=2))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Convolution1D(32, 3, border_mode='same'))
        model.add(MaxPooling1D(pool_length=2))
        model.add(Flatten())
        model.add(Dense(output_dim=1))
        model.add(Activation('linear'))

        #   save model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")

    else:

        # load json and create model
        json_file = open(path + "/model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(path + "/model.h5")

        model = loaded_model

    if trainModel:
        # optimizer = keras.optimizers.Adam(learning_rate=0.0022)
        model.compile(optimizer='rmsprop', loss='mse')
        history = model.fit_generator(generator, epochs=1, verbose=1)

        #   save model to disk
        model.save_weights(path + "/model.h5")

    #   make future predictions
    pred_list = []
    print(data)

    batch = data[0][-n_input:].reshape((1, n_input, n_features))
    for i in range(n_input):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)

    return pred_list