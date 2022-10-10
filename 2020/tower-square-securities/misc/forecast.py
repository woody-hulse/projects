import math
import numpy as np
import plotly.io as pio
import plotly.graph_objects as graph
import matplotlib.pyplot as plt
import random

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

#   create new model or improve on existing one
new_model = False
#   training model or na
train_model = False
#   if testing forecast on previous data
backtest = False
#   if evaluating forecasts
evaluate = False
#   buying random stocks in evaluation
random_picks = False

today = '2022-01-21'

company = 'GOOG'


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
# ticker_list = ['AAPL']

if not evaluate:
    if backtest:
        df = pandas_datareader.DataReader(
            company, data_source='yahoo', start='2013-01-01', end='2019-01-01'
        )
        train = df.filter(['Close'])

        df_com = pandas_datareader.DataReader(
            company, data_source='yahoo', start='2013-01-01', end='2020-01-01'
        )
    else:
        df = pandas_datareader.DataReader(
            company, data_source='yahoo', start='2013-01-01', end=today
        )
        train = df.filter(['Close'])

        df_com = pandas_datareader.DataReader(
            company, data_source='yahoo', start='2013-01-01', end=today
        )

    print(train)

    #   scale data
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)

    #   create timeseries to input to neural network
    n_input = 253
    n_features = 1
    generator = TimeseriesGenerator(train, train, length=n_input, batch_size=1)

    #   create and train model
    if new_model:

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

    if train_model:
        # optimizer = keras.optimizers.Adam(learning_rate=0.0022)
        print(company)
        model.compile(optimizer='rmsprop', loss='mse')
        history = model.fit_generator(generator, epochs=1, verbose=1)

        #   save model to disk
        model.save_weights('model.h5')
        print('saved weights to disk')

    #   make future predictions
    pred_list = []
    batch = train[-n_input:].reshape((1, n_input, n_features))
    for i in range(n_input):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)

    #   create dataframe of future months
    add_dates = [df.index[-1] + DateOffset(days=x) for x in range(0, 254)]
    future_dates = pd.DataFrame(index=add_dates[1:], columns=df.columns)

    #   merge second dataframe to make cohesive graph
    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                              index=future_dates[-n_input:].index, columns=['Prediction'])
    df_proj = pd.concat([df, df_predict], axis=1)

    #   plot
    plot_data = [
        graph.Scatter(
            x=df_proj.index,
            y=df_proj['Prediction'],
            name='prediction'
        ),
        graph.Scatter(
            x=df_com.index,
            y=df_com['Close'],
            name='actual'
        )
    ]
    plot_layout = graph.Layout(
            title=company + ' stock price'
        )
    fig = graph.Figure(data=plot_data, layout=plot_layout)
    fig.show()

elif backtest and not train_model:

    #   create evaluation stats
    accuracy = [0] * 180

    total = 0

    #   load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #   load weights into new model
    loaded_model.load_weights("model.h5")
    print("loaded model from disk")

    model = loaded_model

    for ticker in ticker_list:

        try:
            #   create dataframes
            df = pandas_datareader.DataReader(
                ticker, data_source='yahoo', start='2013-01-01', end='2019-01-01'
            )
            train = df.filter(['Close'])

            df_com = pandas_datareader.DataReader(
                ticker, data_source='yahoo', start='2018-12-31', end='2020-01-01'
            ).filter(['Close'])

            if df.iat[len(df) - 1, 0] < 5:
                continue

            if not random_picks:
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

                #   merge second dataframe to make cohesive graph
                df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                                          index=future_dates[-n_input:].index, columns=['Prediction'])
                df_proj = pd.concat([df, df_predict], axis=1)

                #   evaluate projection
                start_a = df.iat[len(df) - 1, 0]
                start_b = df_com.iat[0, 0]

                for i in range(0, len(accuracy)):
                    if (df_predict.iat[i, 0] - start_a) * (df_com.iat[round(0.693 * i) + 1, 0] - start_b) > 0:
                        accuracy[i] += 1
            else:
                #   1 = buy, 0 = don't buy
                pick = int(random.random()*2)

                start = df_com.iat[0, 0]

                for i in range(0, len(accuracy)):
                    if pick == 1 and df_com.iat[round(0.693 * i) + 1, 0] - start > 0:
                        accuracy[i] += 1
                    elif pick == 0 and df_com.iat[round(0.693 * i) + 1, 0] - start <= 0:
                        accuracy[i] += 1

            total += 1
            print('|', sep='', end='')
        except:
            continue

    for accurate in accuracy:
        print(accurate / total)

    print('total: ' + str(total))

    indexes = []
    new_accuracy = [0.0] * 180
    for i in range(0, len(accuracy)):
        new_accuracy[i] = accuracy[i] / total
        indexes.append(i)

    plt.scatter(indexes, new_accuracy)
    plt.title('accuracy of neural network forecast')
    plt.xlabel('days after jan 1 2019')
    plt.ylabel('accuracy')
    plt.show()

else:
    print("change settings before evaluating")
