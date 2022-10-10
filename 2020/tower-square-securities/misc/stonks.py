import matplotlib.pyplot as plt
import math
import numpy as np

import h5py
import tensorflow as tf
import pandas as pd
import pandas_datareader
from keras import Input, Model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense, LSTM, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.models import load_model


# keep false
new_model = False

company = 'DG'
df = pandas_datareader.DataReader(company, data_source='yahoo', start='2015-01-01', end='2020-6-10')


def run_lstm(input_data, train_model):
    sim_days = 200

    #   create data frame with only 'Close' data
    data = input_data.filter(['Close'])

    #   calculate simple moving averages
    sma_60 = input_data.filter(['Close']).rolling(window=60).mean()
    sma_200 = input_data.filter(['Close']).rolling(window=200).mean()

    input_data['SMA 60 Difference'] = data - sma_60
    input_data['SMA 200 Difference'] = data - sma_200

    #   convert data frame to numpy array
    dataset = input_data.filter(['Close']).values

    # print("length of trading data used: " + str(training_data_len))

    #    scale data
    training_data_len = math.ceil(len(dataset) - sim_days)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # print(scaled_data)

    #   create training data set
    train_data = scaled_data[0:training_data_len, :]

    #   split data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(sim_days, len(train_data)):
        x_train.append(train_data[i - sim_days:i])
        y_train.append(train_data[i, 0])

    #   convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #   reshape data (LSTM network expects input to be 3-dimensional (number of samples, timesteps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    if new_model:
        #   create model
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))

        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.1))

        model.add(Dense(64))
        model.add(Dense(64))
        model.add(Dense(32))
        model.add(Dense(1))

        #   save model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("saved model to disk")

    else:
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("loaded model from disk")

        model = loaded_model

    train_model = True

    if train_model:
        #   compile/optimize model
        model.compile(optimizer='adam', loss='mean_squared_error')

        #   train model
        model.fit(x_train, y_train, batch_size=8, epochs=1)

        #   save model to disk
        model.save_weights('model.h5')

    #   create testing data set (new array of scaled values from i=x_train.shape[0] to 2003)
    test_data = scaled_data[training_data_len - sim_days:, :]
    #   create data set x_test, y_test
    x_test = []
    #   all values that model should predict
    y_test = dataset[training_data_len:, :]

    for i in range(sim_days, len(test_data)):
        x_test.append(test_data[i - sim_days:i, 0])

    #   convert data to numpy array
    x_test = np.array(x_test)
    #   reshape data into 3d array for lstm model
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #   retrieve model's predicted price values (x_test data set)
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    #   get error (root mean squared error again, lower values are better fit)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)

    print("root mean squared error: " + str(rmse))

    print("tomorrow: " + str(predictions[len(predictions) - 1]))
    print("tomorrow diff: " + str(predictions[len(predictions) - 1] - predictions[len(predictions) - 2]))

    #   theoretical revenue with model used over sim_days
    money = 30000
    stock_owned = 0
    for i in range(0, len(predictions) - 2):
        if predictions[i + 1] > predictions[i]:
            purchase = math.floor(money / dataset[training_data_len + i])
            money -= purchase * dataset[training_data_len + i]
            stock_owned += purchase
        else:
            money += stock_owned * dataset[training_data_len + i]
            stock_owned = 0

    print("final value of all assets: " + str(money + stock_owned * dataset[len(dataset) - 1]))

    #   plot data
    # forecasts = lstm.predict_sequences_multiple(model, x_test, 50, 50)
    # lstm.plot_results_multiple(forecasts, y_test, 50)

    fig, ax1 = plt.subplots()

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    #   draw plot
    ax1.plot(train['Close'], color='red')
    ax1.plot(valid[['Close', 'Predictions']])

    ax2 = ax1.twinx()
    ax2.plot(sma_60, label='60-day sma', color='green')
    ax3 = ax1.twinx()
    ax3.plot(sma_200, label='200-day sma', color='yellow')
    # plt.legend(['training data set', 'correct path', 'predicted path'], loc='lower right')
    plt.figure(figsize=(16, 8))
    plt.title('lstm model (' + company + ')')
    plt.xlabel('date', fontsize=13)
    plt.ylabel('close price usd', fontsize=13)
    plt.show()

    # print(valid)


def main():
    run_lstm(df, False)


main()
