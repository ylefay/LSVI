"""
Second example from Schäfer and Chopin (2013), concrete compressive strength.

To generate bar plots as in the paper, see bar_plots.py

"""

import numpy as np
from particles import datasets

dataset = datasets.Concrete()
names = dataset.predictor_names
raw = dataset.raw_data  # do NOT rescale predictors
n, p = raw.shape
response = raw[:, -1]

cols = {}
for i, k in enumerate(names):
    cols[k] = raw[:, i]
    # add log of certain variables
    if k in ['cement', 'water', 'coarse aggregate', 'age', 'fine aggregate']:
        cols['log(%s)' % k] = np.log(cols[k])

# interactions
colkeys = list(cols.keys())
for i, k in enumerate(colkeys):
    for j in range(i):
        k2 = colkeys[j]
        cols[f'{k} x {k2}'] = cols[k] * cols[k2]

# add intercept last
cols['intercept'] = np.ones(n)

center = True  # Christian centered the columns for some reason
if center:
    for k, c in cols.items():
        if k != 'intercept':
            c -= c.mean(axis=0)

preds = np.stack(list(cols.values()), axis=1)
dict_replace = {'cement': 'C', 'water': 'W', 'coarse aggregate': 'CA', 'age': 'A', 'fine aggregate': 'FA',
                'fly ash': 'FASH', 'superplasticizer': 'PLAST', 'blast': 'BLAST', 'intercept': 'CST'}
dict_replace_2 = {'LOG(': 'LG', ')': ''}
string = ""
for idx, feature in enumerate(list(cols.keys())):
    string += str(idx + 1) + ";" + str(feature) + ";" + "\n"
for key, value in dict_replace.items():
    string = string.replace(key, value)
string = string.replace('log(', 'LG')
string = string.replace(')', '')
print(string)
print(cols.keys())

npreds = len(cols)
data = preds, response

np.savetxt('concrete_from_particles.csv', np.hstack((response[:, np.newaxis], preds)), delimiter=',')
