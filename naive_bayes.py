# Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Import dataset
data = pd.read_csv('iris.csv', header=0)
samples = len(data)

# Drop unneeded categorical and numerical columns
categorical = 0
numerical = 0
drop = {'cat': [], 'num': []}
for col in data.iloc[:, :-1]:

    # Check if categorical or numerical
    if data[col].dtype == 'object':
        categorical += 1
    else:
        numerical += 1

    # Check for >10% NaN or identification columns
    if (
        data[col].isna().sum() > 0.1 * len(data[col]) or
        (data[col].dtype == 'object' and len(set(data[col])) > 0.5 * len(data[col]))
        ):

        # Flag column for deletion
        if data[col].dtype == 'object':
            drop['cat'].append(col)
        else:
            drop['num'].append(col)

# Drop columns
data = data.drop(columns=drop['cat'] + drop['num'])

# Drop rows with NaN
data = data.dropna()

# Report dropped features and samples
print('==========DATA CLEANING==========')
print('Categorical features:  %d/%d (%s dropped)'
    % (categorical - len(drop['cat']), categorical,
       (str(drop['cat'])[1:-1]) if drop['cat'] else 'none'))
print('Numerical features:    %d/%d (%s dropped)'
    % (numerical - len(drop['num']), numerical,
       (str(drop['num'])[1:-1]) if drop['num'] else 'none'))
print('Samples:               %d/%d'
    % (len(data), samples))

# Split dataset into training and test datasets
train, test = train_test_split(data, test_size=0.2)

# Numerical features
train_num = train.iloc[:, :-1].select_dtypes(include=['number'])
test_num = test.iloc[:, :-1].select_dtypes(include=['number'])

# Categorical features
train_cat = train.iloc[:, :-1].select_dtypes(exclude=['number'])
test_cat = test.iloc[:, :-1].select_dtypes(exclude=['number'])

# Convert feature columns to numerical type
for col in data.iloc[:, :-1]:
    if data[col].dtype == 'object':
        dicy = {y: x for x, y in enumerate(set(data[col]))}
        data[col] = data[col].map(dicy)

# Map labels column to class index
labels = {y: x for x, y in enumerate(set(data.iloc[:, -1]))}
if data.iloc[:, -1].dtype == 'object':
    data.iloc[:, -1] = data.iloc[:, -1].map(labels)


# Instantiate the classifier
gnb = GaussianNB()

# Train classifier
gnb.fit(train.iloc[:, :-1].values, train.iloc[:, -1])
y_pred = gnb.predict(test.iloc[:, :-1])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          test.shape[0],
          (test.iloc[:, -1] != y_pred).sum(),
          100*(1-(test.iloc[:, -1] != y_pred).sum()/test.shape[0])
))