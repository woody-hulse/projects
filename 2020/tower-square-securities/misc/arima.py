from matplotlib import pyplot as plt
import math
import numpy as np
import csv
import pandas as pd
import pandas_datareader
from pandas import DataFrame
from pandas import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


company = 'MSFT'
df = pandas_datareader.DataReader(company, data_source='yahoo', start='2018-01-01', end='2020-4-28')


def run_arima(input_data):
    #   create data frame with only 'Close' data
    data = input_data.filter(['Close'])

    data.to_csv('arima.csv', header=None)

    data = pd.read_csv('arima.csv', header=None)

    print(data)

    # autocorrelation_plot(dataset)
    # pyplot.show()

    # model_details(dataset)

    train_model(data)


def train_model(dataset):
    X = dataset.values
    size = int(len(X) * 0.5)

    train = X[0:size]
    test = X[size:len(X)]
    history = [x for x in train]

    predictions = list()
    for i in range(len(test)):
        model = ARIMA(dataset, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        initial = output[0]
        predictions.append(initial)
        obs = test[i]
        history.append(obs)
        print('predicted=%f, expected=%f' % (initial, obs))
    error = mean_squared_error(test, predictions)
    print('mean squared error: %.3f' % error)

    #   plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()


def model_details(dataset):
    #   create ARIMA model
    model = ARIMA(dataset, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    #   plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())


def main():
    run_arima(df)

main()
