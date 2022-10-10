import numpy as np
import h5py
import pandas as pd
import datetime
from datetime import date

#   ORDER: [YTD Return 1.5, Expense Ratio 0.5, Beta 1.2, Yield 1, Holdings Turnover 0.5,
#           Morningstar Rating 1.5,
#           Cash -0.5 (line after), Maturity 1.5 <5 (line after),
#           Duration 1.2 (short term) (line after),
#           Morningstar Return Rating 1.5 (line after), 5-Year Average Return 1.5 (line after)
#           Number of Years Up 0.5 (LA), Number of Years Down <- (LA),
#           Morningstar Risk Rating 1.2 (LA), Alpha 1.5 (LA)

return_weight = 1.5
expense_ratio_weight = 0.5
morningstar_weight = 1.5
beta_weight = 1.2
yield_weight = 1
turnover_weight = 0.5
pct_cash_weight = 0.5
maturity_weight = 1.5
duration_weight = 1.2
return_rating_weight = 1.5
avg_return_weight = 1.5
up_down_weight = 0.5
risk_rating_weight = 1.2
alpha_weight = 1.5

'''
return_weight = 1
expense_ratio_weight = 1
morningstar_weight = 1
beta_weight = 1
yield_weight = 1
turnover_weight = 1
pct_cash_weight = 1
maturity_weight = 1
duration_weight = 1
return_rating_weight = 1
avg_return_weight = 1
up_down_weight = 1
risk_rating_weight = 1
alpha_weight = 1
'''

total_weight = return_weight + expense_ratio_weight + morningstar_weight + beta_weight + turnover_weight + \
               yield_weight + turnover_weight + pct_cash_weight + maturity_weight + duration_weight + \
               return_rating_weight + avg_return_weight + up_down_weight + risk_rating_weight + alpha_weight


#   transform text file into list of strings
def to_list(file):
    file = open(file, 'r')
    list_of_words = []

    for line in file:
        stripped_line = line.strip()
        list_of_words.append(stripped_line)

    file.close()
    return list_of_words


bonds = to_list("bond_statistics.txt")

for i in range(0, len(bonds)):

    try:
        bond_strings = bonds[i].split(' ')
        ticker = bond_strings[0]

        #   convert to float values
        bond_stats = []
        for i in range(1, len(bond_strings)):
            bond_stats.append(float(bond_strings[i]))

        #   reset null statistics to default values
        if bond_stats[1] == 0:
            bond_stats[1] = 0.74

        if bond_stats[2] == 0:
            bond_stats[2] = 1

        if bond_stats[7] == 0:
            bond_stats[7] = 1

        #                       bind all values between 0 and 1 and apply weight
        month = int(date.today().strftime('%m'))
        bond_stats[0] = 1 / (1 + pow(3, 0.7 - bond_stats[0] / month)) * return_weight

        bond_stats[1] = 1 / (1 + pow(50, bond_stats[1] - 0.74)) * expense_ratio_weight

        bond_stats[2] = 1 / (1 + pow(20, bond_stats[2] - 1)) * beta_weight

        bond_stats[3] = 1 / (1 + pow(1.4, 5 - bond_stats[3])) * yield_weight

        bond_stats[4] = 1 / (1 + pow(10, (bond_stats[4] - 50)/100)) * turnover_weight

        bond_stats[5] = bond_stats[5] * morningstar_weight

        bond_stats[6] = 1 / (bond_stats[6] / 25 + 1) * pct_cash_weight

        bond_stats[7] = pow(1.5, -1 * pow(bond_stats[7] - 2.5, 2)) * maturity_weight

        bond_stats[8] = 1 / (1.5 * bond_stats[8] + 1) * duration_weight

        bond_stats[9] = bond_stats[9] * return_rating_weight

        bond_stats[10] = 1 / (1 + pow(2, 5 - bond_stats[10])) * avg_return_weight

        bond_stats[11] = bond_stats[11] * up_down_weight

        bond_stats[12] = bond_stats[12] * risk_rating_weight

        bond_stats[13] = 1 / (1 + pow(3, -bond_stats[13])) * alpha_weight

        #   calculate final value
        value = 0
        for stat in bond_stats:
            value += stat
        value /= total_weight

        #   create exception for less than 3 star rating and print
        if bond_stats[5] < 0.5:
            # print(ticker + ': ' + str(value) + ' ⃠ (<3★)')
            pass
        else:
            if value > 0.5:
                print(ticker, format(value, '0.3f'))
            # print(ticker + ': ' + str(value))
    except:
        continue
