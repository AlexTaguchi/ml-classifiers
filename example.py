# Fit Tabular Data to Various Machine Learning Classifier Models

# Import modules
from mlc.data import Preprocess
from mlc.model import NaiveBayes, Logistic, SVM, RandomForest, XGBoost, NeuralNet
import pandas as pd

# Import dataset as dataframe where last column is class label
data = pd.read_csv('data/iris.csv', header=None).sample(frac=1)
# data = pd.read_csv('data/titanic.csv', header=0).sample(frac=1)
# data = pd.read_csv('data/skin.csv', header=None, sep='\t').sample(frac=1).iloc[:10000]

# Preprocess dataframe and split into train and test sets
train, test = Preprocess()(data)

# Test out various classifiers
NaiveBayes()(train, test)
Logistic()(train, test)
SVM()(train, test)
RandomForest()(train, test)
XGBoost()(train, test)
NeuralNet()(train, test)
