import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

#   PRESETS
# Vanguard Primecap (sd = 0.1550, ar = .1083, cost = 0.0014)
vanguard_primecap = True

num_of_simulations = 100

sim_years = 20
business_days = 12

cash = 40000

# the amount put into the portfolio per month (only applies if business_days = 12)
monthly_input = 200

withdrawal_rate = 0.03

# bond to stock ratio
ratio = 0

# total cash after each simulation
final = []

# standard deviation
sd = []
# average return
ar = []
# stock to bond ratio
ratios = []
# withdrawal rate
wr = []

thousand = []

iterations = sim_years * business_days
for i in range(0, num_of_simulations):

    if vanguard_primecap:
        standard_deviation = 0.1550
        avg_return = 0.1083
        withdrawal_rate = 0.0014
    else:
        standard_deviation = 0.15*ratio*ratio - 0.072*ratio + 0.091
        avg_return = 0.09 + 0.035*ratio

    for j in range(0, iterations):

        cash += cash * np.random.normal(loc=avg_return/business_days,
                                        scale=standard_deviation/math.sqrt(business_days))
        cash -= cash * withdrawal_rate/business_days

        if business_days == 12:
            cash += monthly_input

    final.append(cash)
    sd.append(standard_deviation)
    ar.append(avg_return)
    # thousand.append(cash)
    ratios.append(ratio)
    wr.append(withdrawal_rate)

    ratio += 1/num_of_simulations

# rolling mean of data
sma = pd.DataFrame(final).rolling(window=int(num_of_simulations/100)).mean()

plt.figure(figsize=(16, 8))

fig, ax = plt.subplots(4)

ax[0].set_title('final value after ' + str(sim_years) + ' years')
ax[0].plot(final)
ax[0].plot(thousand)

ax[3].set_title('standard deviation, average return, withdrawal rate')
ax[3].plot(sd)
ax[3].plot(ar)
ax[3].plot(wr)

ax[2].set_title('simple moving average')
ax[2].plot(sma)
ax[2].plot(thousand)

ax[1].set_title('stock to bond ratio')
ax[1].plot(ratios)

plt.xlabel('simulation number')

plt.show()
