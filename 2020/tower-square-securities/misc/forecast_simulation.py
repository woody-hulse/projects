import numpy as np
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
import datetime
import random

#   day length of each forecast iteration
forecast_length = 86

today = '2020-08-13'

#   load json and neural network model used in "forecast.py" program
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#   load weights into new model
loaded_model.load_weights("model.h5")
print("loaded neural network model from disk")

model = loaded_model


#   transform text file into list of strings
def to_list(file):
    file = open(file, 'r')
    list_of_words = []

    for line in file:
        stripped_line = line.strip()
        list_of_words.append(stripped_line)

    file.close()
    return list_of_words


ticker_list = to_list("all_tickers.txt")


def buy_stock(ticker, date):

    df = pandas_datareader.DataReader(
        ticker, data_source='yahoo', start='2013-01-01', end=date
    )
    real_price = pandas_datareader.DataReader(
        ticker, data_source='yahoo', start=date, end=date
    )
    train = df.filter(['Close'])

    #   scale data
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)

    #   create timeseries
    n_input = 253
    n_features = 1

    #   make future predictions
    pred_list = []
    batch = train[-n_input:].reshape((1, n_input, n_features))
    for i in range(n_input):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)

    #   create dataframe of future months
    add_dates = [df.index[-1] + DateOffset(days=x) for x in range(0, 254)]
    future_dates = pd.DataFrame(index=add_dates[1:], columns=df.columns)

    #   create prediction list
    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                              index=future_dates[-n_input:].index, columns=['Prediction'])

    #   comparison
    start = df.iat[len(df) - 1, 0]

    if df_predict.iat[forecast_length, 0] > start:
        return real_price.iat[0, 0]
    else:
        return 0


def simulate():

    date = datetime.datetime(year=2014, month=1, day=10)

    money = 200.0

    owned = []

    while float(date.strftime('%Y')) < 2020:

        print('\n' + date.strftime('%m-%d-%Y') + ':\n')
        #   sell
        while len(owned) > 0:
            price = 0
            while price == 0:
                date = date + datetime.timedelta(days=1)
                try:
                    price = pandas_datareader.DataReader(
                        owned[0], data_source='yahoo', start=date.strftime('%Y-%m-%d'), end=date.strftime('%Y-%m-%d')
                    ).filter(['Close']).iat[0, 0]
                except KeyError:
                    continue

            print('sold ' + owned[0] + ' for ' + str(price))

            money += price
            owned.remove(owned[0])

        #   print day's money
        print('\nStart of ' + date.strftime('%m-%d-%Y') + ': $' + str(money))

        #   buy new stocks
        while money > 0:

            choice = int(random.randrange(0, len(ticker_list)))

            ticker = ticker_list[choice]

            try:
                price = buy_stock(ticker, date.strftime('%Y-%m-%d'))

                if price > 10:

                    if money - price < 0:
                        if money > 25:
                            continue
                        else:
                            break
                    else:
                        print('bought ' + ticker + ' for ' + str(price))
                        money -= price
                        owned.append(ticker)
            except:
                continue

        date = date + datetime.timedelta(days=forecast_length - 1)

    print(money)


simulate()

