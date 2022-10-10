import matplotlib.pyplot as plt
import math
import numpy as np
import datetime
import h5py
import tensorflow
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM
import os
from keras.models import load_model
# plt.style.use('dark_background')

df = web.DataReader('AAPL', data_source='yahoo', start='2018-01-01', end='2020-04-25')


def simulate(input_data):
    money = 300000
    stock_owned = 0

    #   create dataframe with only 'Close' data
    print(input_data)
    data = input_data.filter(['Close'])
    #   convert dataframe to numpy array
    dataset = data.values
    # print("length of trading data used: " + str(training_data_len))

    #    scale data
    sim_days = int(len(dataset) * 0.1)
    training_data_len = math.ceil(len(dataset) - sim_days)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # print(scaled_data)

    #   create training dataset
    train_data = scaled_data[0:training_data_len, :]
    #   split data into x_train and y_train datasets
    x_train = []
    y_train = []

    for i in range(sim_days, len(train_data)):
        x_train.append(train_data[i - sim_days:i, 0])
        y_train.append(train_data[i, 0])

    #   convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #   reshape data (LSTM network expects input to be 3-dimensional (number of samples, timesteps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #   create model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    #   compile/optimize model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #   train model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    #   save model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    #   create testing dataset (new array of scaled values from i=x_train.shape[0] to 2003)
    test_data = scaled_data[training_data_len - sim_days:, :]
    #   create data set x_test, y_test
    x_test = []
    #   all values that model should predict
    y_test = dataset[training_data_len:, :]

    for i in range(sim_days, len(test_data)):
        x_test.append(test_data[i-sim_days:i, 0])

    #   convert data to numpy array
    x_test = np.array(x_test)
    #   reshape data into 3d array for lstm model
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #   retrieve model's predicted price values (x_test dataset)
    predictions = model.predict(x_test)

    predictions = scaler.inverse_transform(predictions)

    #   get error (root mean squared error again, lower values are better fit)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)

    print("root mean squared error: " + str(rmse))

    #   theoretical revenue with model used over sim_days
    for i in range(1, len(predictions) - 1):
        print(str(predictions[i]) + ", " + str(dataset[training_data_len + i - 1]))
        if predictions[i] > predictions[i-1]:
            purchase = math.floor(money / dataset[training_data_len + i - 1])
            money -= purchase * dataset[training_data_len + i - 1]
            stock_owned += purchase
        else:
            money += stock_owned * dataset[training_data_len + i - 1]
            stock_owned = 0

    print(str(money + stock_owned * dataset[len(dataset) - 1]))

    #   plot data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    #   draw plot
    plt.figure(figsize=(16, 8))
    plt.title('lstm model (AAPL)')
    plt.xlabel('date', fontsize=13)
    plt.ylabel('close price usd', fontsize=13)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['training dataset', 'correct path', 'predicted path'], loc='lower right')
    plt.show()


def main():
    simulate(df)


main()
