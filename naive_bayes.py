import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Importing dataset
data = pd.read_csv("iris.csv", header=None)

# Convert feature columns to numerical type
for col in data.iloc[:, :-1]:
    if data[col].dtype == 'object':
        dicy = {y: x for x, y in enumerate(set(data[col]))}
        data[col] = data[col].map(dicy)
        print(data)

# Map labels column to class index
labels = {y: x for x, y in enumerate(set(data.iloc[:, -1]))}
if data.iloc[:, -1].dtype == 'object':
    data.iloc[:, -1] = data.iloc[:, -1].map(labels)

# Split dataset in training and test datasets
train, test = train_test_split(data, test_size=0.2)

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