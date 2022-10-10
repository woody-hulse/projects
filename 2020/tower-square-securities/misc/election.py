import censusdata
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 2)

import statsmodels.formula.api as sm


#   download basic socioeconomic statistics for each state
statedata = censusdata.download('acs5', 2015, censusdata.censusgeo([('state', '*')]),
                                ['B01001_001E', 'B19013_001E', 'B19083_001E',
                                 'C17002_001E', 'C17002_002E', 'C17002_003E', 'C17002_004E',
                                 'B03002_001E', 'B03002_003E', 'B03002_004E', 'B03002_012E'])

#   percent of each state voting democrat in the 2016 election

democrat_pct = {
    censusdata.censusgeo((('state', '01'),)): 34.6,
    censusdata.censusgeo((('state', '02'),)): 37.7,
    censusdata.censusgeo((('state', '04'),)): 45.4,
    censusdata.censusgeo((('state', '05'),)): 33.8,
    censusdata.censusgeo((('state', '06'),)): 61.6,
    censusdata.censusgeo((('state', '08'),)): 47.2,
    censusdata.censusgeo((('state', '09'),)): 54.5,
    censusdata.censusgeo((('state', '10'),)): 53.4,
    censusdata.censusgeo((('state', '11'),)): 92.8,
    censusdata.censusgeo((('state', '12'),)): 47.8,
    censusdata.censusgeo((('state', '13'),)): 45.6,
    censusdata.censusgeo((('state', '15'),)): 62.3,
    censusdata.censusgeo((('state', '16'),)): 27.6,
    censusdata.censusgeo((('state', '17'),)): 55.4,
    censusdata.censusgeo((('state', '18'),)): 37.9,
    censusdata.censusgeo((('state', '19'),)): 42.2,
    censusdata.censusgeo((('state', '20'),)): 36.2,
    censusdata.censusgeo((('state', '21'),)): 32.7,
    censusdata.censusgeo((('state', '22'),)): 38.4,
    censusdata.censusgeo((('state', '23'),)): 47.9,
    censusdata.censusgeo((('state', '24'),)): 60.5,
    censusdata.censusgeo((('state', '25'),)): 60.8,
    censusdata.censusgeo((('state', '26'),)): 47.3,
    censusdata.censusgeo((('state', '27'),)): 46.9,
    censusdata.censusgeo((('state', '28'),)): 39.7,
    censusdata.censusgeo((('state', '29'),)): 38,
    censusdata.censusgeo((('state', '30'),)): 36,
    censusdata.censusgeo((('state', '31'),)): 34,
    censusdata.censusgeo((('state', '32'),)): 47.9,
    censusdata.censusgeo((('state', '33'),)): 47.6,
    censusdata.censusgeo((('state', '34'),)): 55,
    censusdata.censusgeo((('state', '35'),)): 48.3,
    censusdata.censusgeo((('state', '36'),)): 58.8,
    censusdata.censusgeo((('state', '37'),)): 46.7,
    censusdata.censusgeo((('state', '38'),)): 27.8,
    censusdata.censusgeo((('state', '39'),)): 43.5,
    censusdata.censusgeo((('state', '40'),)): 28.9,
    censusdata.censusgeo((('state', '41'),)): 51.7,
    censusdata.censusgeo((('state', '42'),)): 47.6,
    censusdata.censusgeo((('state', '44'),)): 55.4,
    censusdata.censusgeo((('state', '45'),)): 40.8,
    censusdata.censusgeo((('state', '46'),)): 31.7,
    censusdata.censusgeo((('state', '47'),)): 34.9,
    censusdata.censusgeo((('state', '48'),)): 43.4,
    censusdata.censusgeo((('state', '49'),)): 27.8,
    censusdata.censusgeo((('state', '50'),)): 61.1,
    censusdata.censusgeo((('state', '51'),)): 49.9,
    censusdata.censusgeo((('state', '53'),)): 54.4,
    censusdata.censusgeo((('state', '54'),)): 26.5,
    censusdata.censusgeo((('state', '55'),)): 46.9,
    censusdata.censusgeo((('state', '56'),)): 22.5,
}
voting2016 = pd.DataFrame.from_dict(democrat_pct, orient='index')
statedata['percent_democratic_pres_2016'] = voting2016

#   rename columns to their appropriate name, resize some statistics
statedata = statedata.rename(columns={'B01001_001E': 'population_size'})
statedata.population_size = statedata.population_size / 100000
statedata = statedata.rename(columns={'B19013_001E': 'median_HH_income'})
statedata['median_HH_income'] = statedata['median_HH_income'] / 1000
statedata = statedata.rename(columns={'B19083_001E': 'gini_index'})
statedata.gini_index = statedata.gini_index * 100
statedata['percent_below_125_poverty'] = (statedata['C17002_002E'] + statedata['C17002_003E'] + statedata['C17002_004E']) / statedata['C17002_001E'] * 100
statedata['percent_nonhisp_white'] = statedata['B03002_003E'] / statedata['B03002_001E'] * 100
statedata['percent_nonhisp_black'] = statedata['B03002_004E'] / statedata['B03002_001E'] * 100
statedata['percent_hispanic'] = statedata['B03002_012E'] / statedata['B03002_001E'] * 100

#   check the validity of the data, remove unnecessary variables
assert (statedata['population_size'] == statedata['B03002_001E'] / 100000).all()
for column in ['C17002_001E', 'C17002_002E', 'C17002_003E', 'C17002_004E',
               'B03002_001E', 'B03002_003E', 'B03002_004E', 'B03002_012E',]:
    del statedata[column]

#   remove puerto rico ;(
statedata = statedata.drop([censusdata.censusgeo([('state', '72')])])

#   print
statedata = statedata.reindex(columns=['percent_democratic_pres_2016', 'population_size', 'median_HH_income', 'percent_below_125_poverty', 'gini_index', 'percent_nonhisp_white', 'percent_nonhisp_black', 'percent_hispanic'])
statedata.describe()
