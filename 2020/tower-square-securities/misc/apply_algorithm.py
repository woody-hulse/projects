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

#   ORDER: [trailing_pe, beta, dividend_yield, revenue_growth, diluted_eps, earnings_growth,
#           fair_value, recommendation_rating, share_price_growth, public opinion]


clean = False

preset = 'work'

today = '2020-08-13'


#   predicts growth
if preset == 'growth':
    pe_weight = 0.8
    beta_weight = 0.4
    dividend_weight = 0.2
    eps_weight = 1.1
    earnings_growth_weight = 0.4
    fair_value_weight = 1.1
    analyst_weight = 1.1
    share_price_weight = 0.8
    opinion_weight = 0.6
    revenue_growth_weight = 0.4
    nn_weight = 0.2

#   for job
elif preset == 'work':
    pe_weight = 1
    beta_weight = 1
    dividend_weight = 2.5
    eps_weight = 0.4
    earnings_growth_weight = 0.7
    fair_value_weight = 1
    analyst_weight = 1
    share_price_weight = 1
    opinion_weight = 0
    revenue_growth_weight = 1
    nn_weight = 0

#   initial presets
elif preset == 'original':
    pe_weight = 0.7
    beta_weight = 1.1
    dividend_weight = 0.5
    eps_weight = 0.4
    earnings_growth_weight = 0.4
    fair_value_weight = 0.7
    analyst_weight = 1.2
    share_price_weight = 1
    opinion_weight = 0.4
    revenue_growth_weight = 5
    nn_weight = 0

#   custom
else:
    pe_weight = 0.8
    beta_weight = 0.4
    dividend_weight = 0.2
    eps_weight = 1.1
    earnings_growth_weight = 0.4
    fair_value_weight = 1.1
    analyst_weight = 1.1
    share_price_weight = 0.8
    opinion_weight = 0.6
    revenue_growth_weight = 0.4
    nn_weight = 1


total_weight = pe_weight + beta_weight + dividend_weight + eps_weight + earnings_growth_weight \
               + fair_value_weight + analyst_weight + share_price_weight + opinion_weight\
               + revenue_growth_weight + nn_weight


#   load json and neural network model used in "forecast.py" program
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#   load weights into new model
loaded_model.load_weights("model.h5")
print("loaded neural network model from disk")

model = loaded_model


def neural_network_forecast(ticker):

    df = pandas_datareader.DataReader(
        ticker, data_source='yahoo', start='2013-01-01', end=today
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

    if df_predict.iat[74, 0] > start:
        return 1
    else:
        return 0


#   transform text file into list of strings
def to_list(file):
    file = open(file, 'r')
    list_of_words = []

    for line in file:
        stripped_line = line.strip()
        list_of_words.append(stripped_line)

    file.close()
    return list_of_words


companies = to_list("company_statistics.txt")

# value_array = [[company for company in companies] for i in range(2)]

for i in range(0, len(companies)):

    company_strings = companies[i].split(' ')
    ticker = company_strings[0]

    #   convert to float values
    company_statistics = []
    for i in range(1, len(company_strings)):
        company_statistics.append(float(company_strings[i]))

    #                       bind all values between 0 and 1

    if company_statistics[0] == 1:
        company_statistics[0] = 23.16

    a = company_statistics[0]
    b = company_statistics[1]
    c = company_statistics[2]
    d = company_statistics[3]
    e = company_statistics[4]
    f = company_statistics[5]
    g = company_statistics[6]
    h = company_statistics[7]
    i = company_statistics[8]


    #   trailing p/e ratio, 23.16 average
    company_statistics[0] = pow(0.5, company_statistics[0] / 23.16)
    #   beta
    company_statistics[1] = -1 * 1 / (1 + pow(10, 1 - company_statistics[1])) + 1
    #   dividend rate
    company_statistics[2] = 1 / (1 + pow(1.4, 5 - company_statistics[2]))
    #   revenue growth
    company_statistics[3] = 1 / (1 + pow(1.04, -1 * company_statistics[3]))
    #   diluted eps
    company_statistics[4] = 1 / (1 + pow(1.2, -1 * company_statistics[4]))
    #   earnings_growth
    company_statistics[5] = 1 / (1 + pow(1.04, -1 * company_statistics[5]))
    #   fair value
    company_statistics[6] = 1 / (1 + pow(2000, 1 - company_statistics[6]))
    #   share price growth
    company_statistics[8] = 1 / (1 + pow(100, 1 - company_statistics[8]))

    #   nn forecast
    if nn_weight != 0:
        forecast = neural_network_forecast(ticker)
    else:
        forecast = 0

    #   rename variables
    pe = company_statistics[0]
    beta = company_statistics[1]
    dividend_rate = company_statistics[2]
    revenue_growth = company_statistics[3]
    diluted_eps = company_statistics[4]
    earnings_growth = company_statistics[5]
    fair_value = company_statistics[6]
    recommendation = company_statistics[7]
    share_growth = company_statistics[8]
    public_opinion = company_statistics[9]

    value = (pe_weight * pe + revenue_growth_weight * revenue_growth +
             earnings_growth_weight * earnings_growth + dividend_weight * dividend_rate +
             fair_value_weight * fair_value + share_price_weight * share_growth +
             analyst_weight * recommendation + beta_weight * beta +
             eps_weight * diluted_eps + opinion_weight * public_opinion + forecast * nn_weight) / total_weight

    if clean:
        if value > 0.6:
            print(format(value, '0.3f'))
    else:
        if value > .65:
            print(ticker, a, b, c, d, e, f, format(g, '0.2f'), h, format(i, '0.2f'), format(value, '0.3f'), ticker)
            # print(ticker + ': ' + str(value))
