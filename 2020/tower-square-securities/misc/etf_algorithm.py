import numpy as np
import h5py
import pandas as pd
import datetime
from datetime import date

#   NAV, Yield, Beta, Expense Ratio,
#   1-Year Daily Total Return, 3-Year Daily Total Return,
#   Alpha, Mean Annual Return, Sharpe Ratio, Treynor Ratio


nav_weight = 1.3
yield_weight = 1.6
beta_weight = 0.4
expense_ratio_weight = 0.3
one_year_weight = 0.6
three_year_weight = 0.1
alpha_weight = 1.4
mean_return_weight = 0.6
sharpe_weight = 0.6
treynor_weight = 0.8

'''
nav_weight = 1
yield_weight = 1
beta_weight = 1
expense_ratio_weight = 1
one_year_weight = 1
three_year_weight = 1
alpha_weight = 1
mean_return_weight = 1
sharpe_weight = 1
treynor_weight = 1
'''


total_weight = yield_weight + beta_weight + expense_ratio_weight + one_year_weight + three_year_weight + alpha_weight +\
               mean_return_weight + sharpe_weight + treynor_weight


#   transform text file into list of strings
def to_list(file):
    file = open(file, 'r')
    list_of_words = []

    for line in file:
        stripped_line = line.strip()
        list_of_words.append(stripped_line)

    file.close()
    return list_of_words


etfs = to_list("etf_statistics.txt")

for i in range(0, len(etfs)):

    try:
        etf_strings = etfs[i].split(' ')
        ticker = etf_strings[0]

        #   convert to float values
        etf_stats = []
        for i in range(1, len(etf_strings)):
            etf_stats.append(float(etf_strings[i]))

        #   check for insignificant data collection
        zeros = 0
        for stat in etf_stats:
            if stat == 0:
                zeros += 1

        if zeros > 3:
            continue

        #   reset null statistics to default values
        if etf_stats[2] == 0:
            etf_stats[2] = 1

        if etf_stats[0] == 0:
            etf_stats[0] = 1

        if etf_stats[9] > 100:
            etf_stats[9] = 99

        if etf_stats[9] < -100:
            etf_stats[9] = -99

        a = etf_stats[0]
        b = etf_stats[1]
        c = etf_stats[2]
        d = etf_stats[3]
        e = etf_stats[4]
        f = etf_stats[5]
        g = etf_stats[6]
        h = etf_stats[7]
        i = etf_stats[8]
        j = etf_stats[9]

        #                       bind all values between 0 and 1 and apply weight
        etf_stats[0] = 1 / (1 + pow(10, 4 * (etf_stats[0] - 1))) * nav_weight

        etf_stats[1] = (-1 / (etf_stats[1] + 1) + 1) * yield_weight

        etf_stats[2] = 1 / (1 + pow(20, etf_stats[2] - 1)) * beta_weight

        etf_stats[3] = 1 / (1 + pow(50, etf_stats[3] - 0.74)) * expense_ratio_weight

        etf_stats[4] = 1 / (1 + pow(2, 5 - etf_stats[4])) * one_year_weight

        etf_stats[5] = 1 / (1 + pow(2, 5 - etf_stats[5])) * three_year_weight

        etf_stats[6] = 1 / (1 + pow(3, -etf_stats[6])) * alpha_weight

        etf_stats[7] = 1 / (1 + pow(2, 5 - etf_stats[7])) * mean_return_weight

        etf_stats[8] = 1 / (1 + pow(3, 0.5 - etf_stats[8])) * sharpe_weight

        etf_stats[9] = 1 / (1 + pow(3, -etf_stats[9])) * treynor_weight

        #   calculate final value
        value = 0
        for stat in etf_stats:
            value += stat
        value /= total_weight

        if value > 0.62:
            # print(ticker, format(value, '0.3f'))
            # print(ticker + ": " + str(value))
            # print(ticker, format(a, '0.2f'), b, c, d, e, f, g, h, i, j, format(value, '0.3f'))
            print(ticker, format(value, '0.3f'))

    except:
        print(Exception)
        continue
