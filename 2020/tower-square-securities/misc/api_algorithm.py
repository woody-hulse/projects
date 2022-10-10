"""
CRITERIA:

DIVIDEND/SHARE: 0.5
REVENUE: 1.5
P/E RATIO: 0.7
BETA:1.1
ANALYST BUY/SELL/HOLD: 1.2
PRICE HISTORY: 1
QUARTERLY EARNINGS GROWTH: 0.3
FAIR VALUE: 0.7
"""

from yahoo_finance import Share
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import pandas_datareader
import math


#   transform text file into list of strings
def to_list(file):
    file = open(file, 'r')
    list_of_words = []

    for line in file:
        stripped_line = line.strip()
        list_of_words.append(stripped_line)

    file.close()
    return list_of_words


# tickers = to_list('buy_list.txt')

tickers = ['AAPL']


def main():
    for ticker in tickers:

        print(Share('YHOO'))
        # company = Share(ticker)

        # print(company.get_open())


main()
