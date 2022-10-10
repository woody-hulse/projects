import matplotlib.pyplot as plt
import math
import numpy as np

import h5py
import tensorflow as tf
import pandas as pd
import pandas_datareader
from keras import Input, Model, models
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense, LSTM, Flatten, concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.models import load_model


#   keep false
new_model = True

sim_days = 200

company = 'MSFT'
start = '2016-01-01'
end = '2020-5-1'

df = pandas_datareader.DataReader(company, data_source='yahoo', start=start, end=end)


def create_multivariable_model(data_length):

    #   getting input formats to each dataset
    close = Input(shape=(data_length, 64), name='close')
    sma_60 = Input(shape=(data_length, 64), name='sma_60')
    ema_60 = Input(shape=(data_length, 64), name='ema_60')

    #   create separate lstm layers for each dataset
    close_layers = LSTM(64, return_sequences=True)(close)
    sma_60_layers = LSTM(64, return_sequences=True)(sma_60)
    ema_60_layers = LSTM(64, return_sequences=True)(ema_60)

    #   combine lstm layers into one layer
    output = concatenate([close_layers, sma_60_layers, ema_60_layers])
    output = Dense(64, activation='softmax', name='final')(output)

    #   create model
    input_model = Model(
        inputs=[close, sma_60, ema_60],
        outputs=[output]
    )
    input_model.compile(optimizer='adam', loss='mse')

    '''
    print(input_model.summary())

    #   convert model to Sequential to add additional layers
    model = models.Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(input_model.input_shape[1], input_model.input_shape[2])))
    #   model.add(input_model.get_layer(name='final'))
    model.add(Dense(64))
    model.add(Dense(1))

    #   compile model
    model.compile(optimizer='adam', loss='mse')
    '''

    return input_model


def get_training_data_len(input_data):

    return math.ceil(len(input_data.values) - sim_days)


def create_scaled_dataset(input_data):
    #   create numpy array
    dataset = input_data.values

    #   scale dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    return scaled_data


def create_unscaled_dataset(input_data, close_data):

    #   fit MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_data)

    #   unscale dataset
    unscaled_data = scaler.inverse_transform(input_data)

    return unscaled_data


def create_train_dataset(input_data):

    #   scale dataset between 0 and 1
    scaled_data = create_scaled_dataset(input_data)

    #   get length of training dataset
    training_data_len = get_training_data_len(input_data)

    #   create training datasets
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []
    for i in range(sim_days, len(train_data)):
        x_train.append(train_data[i - sim_days:i])
        y_train.append(train_data[i, 0])

    #   convert training datasets to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # remove nan from data
    x_train = clean_dataset(x_train)

    #   reshape data into 3d array for lstm
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


#       i guess clean input_data[0] and [1] but idk
def clean_dataset(input_data):

    for i in range(0, len(input_data)):
        for j in range(0, len(input_data[i])):
            if np.isnan(input_data[i][j][0]):
                input_data[i][j][0] = 0

    return input_data


def create_test_dataset(input_data):

    #   scale dataset between 0 and 1
    scaled_data = create_scaled_dataset(input_data)

    #   get length of input data
    training_data_len = get_training_data_len(input_data)

    #   get test data
    test_data = scaled_data[training_data_len - sim_days:, :]

    #   create test dataset
    x_test = []

    #   all values that model should predict
    y_test = input_data.values[training_data_len:, :]

    for i in range(sim_days, len(test_data)):
        x_test.append(test_data[i - sim_days:i, 0])

    #   convert data to numpy array
    x_test = np.array(x_test)

    #   reshape data into 3d array for lstm model
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # remove nan from data
    x_test = clean_dataset(x_test)

    return x_test, y_test


def get_model(y_train, train_data):

    if new_model:

        model = create_multivariable_model(sim_days)

        #   save model to JSON
        model_json = model.to_json()
        with open("model2.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model2.h5")
        print("saved model to disk")

    else:
        # load json and create model
        json_file = open('model2.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model2.h5")
        print("loaded model from disk")

        model = loaded_model

    #   convert train data to numpy array
    train_data = np.array(train_data)

    #   train model
    model.fit(
        train_data[0], train_data[1], train_data[2],
        y_train, epochs=1, batch_size=1
    )

    #   save model to disk
    model.save_weights('model2.h5')

    return model


def simulate(predicted_data, actual_data):

    training_data_len = get_training_data_len(actual_data)

    #   theoretical revenue with model used over sim_days
    money = 30000
    stock_owned = 0
    for i in range(0, len(predicted_data) - 2):
        if predicted_data[i + 1] > predicted_data[i]:
            purchase = math.floor(money / actual_data[training_data_len + i])
            money -= purchase * actual_data[training_data_len + i]
            stock_owned += purchase
        else:
            money += stock_owned * actual_data[training_data_len + i]
            stock_owned = 0

    return money + stock_owned * actual_data[len(actual_data) - 1]


def run_lstm(input_data, train_model):

    #   create data frame with only 'Close' data
    close = input_data.filter(['Close'])

    #   calculate simple moving averages
    sma_60 = close.rolling(window=60).mean()
    sma_200 = close.rolling(window=200).mean()

    #   calculate exponential moving averages
    ema_60 = close.ewm(span=60, adjust=False).mean()

    #   using the difference between stock price and moving average
    sma_60_diff = close - sma_60
    sma_200_diff = close - sma_200

    #   diff between stock price and ema
    ema_60_diff = close - ema_60

    #   get training datasets
    close_train, y_train = create_train_dataset(close)
    sma_60_train, y_train = create_train_dataset(sma_60_diff)
    ema_60_train, y_train = create_train_dataset(ema_60_diff)

    #   create model or load it from file (depending on 'new_file' boolean)
    model = get_model([y_train], [close_train, sma_60_train, ema_60_train])

    #   create testing datasets
    close_test, y_test = create_test_dataset(close)
    sma_60_test, y_train = create_test_dataset(sma_60_diff)
    ema_60_test, y_train = create_test_dataset(ema_60_diff)

    #   retrieve model's predicted price values (x_test data set)
    predictions = model.predict([close_test, sma_60_test, ema_60_test])
    predictions = create_unscaled_dataset(predictions, close)

    #   get error (root mean squared error again, lower values are better fit)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)

    print("root mean squared error: " + str(rmse))

    print("tomorrow: " + str(predictions[len(predictions) - 1]))
    print("tomorrow diff: " + str(predictions[len(predictions) - 1] - predictions[len(predictions) - 2]))

    #   plot data
    # forecasts = lstm.predict_sequences_multiple(model, x_test, 50, 50)
    # lstm.plot_results_multiple(forecasts, y_test, 50)

    training_data_len = get_training_data_len(close)

    fig, ax1 = plt.subplots()

    train = close[:training_data_len]
    valid = close[training_data_len:]
    valid['Predictions'] = predictions
    #   draw plot
    ax1.patch.set_facecolor('#ababab')
    ax1.patch.set_alpha(0.2)

    ax1.plot(train['Close'], color='red')
    ax1.plot(valid[['Close', 'Predictions']])

    #   plot additional indexes
    ax2 = ax1.twinx()
    ax2.plot(sma_60, label='60-day sma', color='yellow')
    ax3 = ax1.twinx()
    ax3.plot(sma_200, label='200-day sma', color='yellow')
    ax4 = ax1.twinx()
    ax4.plot(ema_60, label='60-day ema', color='yellow')

    plt.legend(['training data set', 'correct path', 'predicted path'], loc='lower right')
    plt.figure(figsize=(16, 8))
    plt.title('lstm model (' + company + ')')
    plt.xlabel('date', fontsize=13)
    plt.ylabel('close price usd', fontsize=13)
    plt.show()

    # print(valid)


def main():
    run_lstm(df, True)


main()
