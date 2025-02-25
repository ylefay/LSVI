"""
Second example from Schäfer and Chopin (2013), concrete compressive strength.

No engineered feature
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

# interactions
colkeys = list(cols.keys())

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
string = ""
for idx, feature in enumerate(list(cols.keys())):
    string += str(idx + 1) + ";" + str(feature) + ";" + "\n"
for key, value in dict_replace.items():
    string = string.replace(key, value)
print(string)
print(cols.keys())

npreds = len(cols)
data = preds, response

np.savetxt('concrete_from_particles_simple.csv', np.hstack((response[:, np.newaxis], preds)), delimiter=',')

print(np.corrcoef(preds, rowvar=False))
