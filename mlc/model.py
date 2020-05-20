#!/usr/bin/env python3

# Modules
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
import time
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier


class NaiveBayes:
    """Naive Bayes classifier"""
    def __init__(self):
        self.categorical = 'object'
        self.numerical = 'number'

    def cat_num_split(self, feature):

        # Split features into categorical and numberical types
        feature_cat = feature.select_dtypes(include=self.categorical)
        feature_num = feature.select_dtypes(include=self.numerical)

        return feature_cat, feature_num

    def train(self, train_cat, train_num, train_labels):

        # Instantiate the classifiers
        mnb = MultinomialNB()
        gnb = GaussianNB()

        # Train classifier
        if train_cat.shape[1]:
            mnb.fit(train_cat, train_labels)
        if train_num.shape[1]:
            gnb.fit(train_num, train_labels)

        return mnb, gnb

    def predict(self, feature_cat, feature_num, mnb, gnb):

        # Predict labels
        cat_pred = mnb.predict_proba(feature_cat) if feature_cat.shape[1] else 0
        cat_pred *= feature_cat.shape[1]
        num_pred = gnb.predict_proba(feature_num) if feature_num.shape[1] else 0
        num_pred *= feature_num.shape[1]

        return (cat_pred + num_pred).argmax(axis=1)

    def __call__(self, train, test):

        # Record start time
        print('==============NAIVE BAYES==============')
        start = time.time()

        # Split train and test into categorical and numerical features
        train_cat, train_num = self.cat_num_split(train.iloc[:, :-1])
        test_cat, test_num = self.cat_num_split(test.iloc[:, :-1])

        # Assign labels
        train_labels = train.iloc[:, -1]
        test_labels = test.iloc[:, -1]

        # Train Gaussian and Multinomial classifiers
        mnb, gnb = self.train(train_cat, train_num, train_labels)

        # Predict train and test labels
        train_pred = self.predict(train_cat, train_num, mnb, gnb)
        test_pred = self.predict(test_cat, test_num, mnb, gnb)

        # Print results
        train_hit = (train_labels == train_pred).sum()
        test_hit = (test_labels == test_pred).sum()
        print('Train accuracy:  %.2f%%'
              % (100 * train_hit / len(train)))
        print('Test accuracy:   %.2f%%'
              % (100 * test_hit / len(test)))
        print('Time: %.1f seconds' % (time.time() - start))
        print('')


class Logistic:
    """Logistic regression classifier"""
    def __init__(self):
        pass

    def train(self, train):

        # Instantiate the classifiers
        mlr = LogisticRegression(solver='sag', max_iter=10000, n_jobs=-1,
                                 multi_class='multinomial')
        ovr = LogisticRegression(solver='sag', max_iter=10000, n_jobs=-1,
                                 multi_class='ovr')

        # Train classifier
        mlr.fit(train.iloc[:, :-1], train.iloc[:, -1])
        ovr.fit(train.iloc[:, :-1], train.iloc[:, -1])

        return mlr, ovr

    def predict(self, feature, mlr, ovr):
        return mlr.predict(feature), ovr.predict(feature)

    def __call__(self, train, test):

        # Record start time
        print('==========LOGISTIC REGRESSION==========')
        start = time.time()

        # Train Logistic regression classifiers
        mlr, ovr = self.train(train)

        # Predict train and test labels
        train_pred_mlr, train_pred_ovr = self.predict(train.iloc[:, :-1], mlr, ovr)
        test_pred_mlr, test_pred_ovr = self.predict(test.iloc[:, :-1], mlr, ovr)

        # Print results
        train_hit_mlr = (train.iloc[:, -1] == train_pred_mlr).sum()
        train_hit_ovr = (train.iloc[:, -1] == train_pred_ovr).sum()
        test_hit_mlr = (test.iloc[:, -1] == test_pred_mlr).sum()
        test_hit_ovr = (test.iloc[:, -1] == test_pred_ovr).sum()
        print('Train accuracy:  %5.2f%% (Multinomial), %6.2f%% (One-vs-Rest)'
              % (100 * train_hit_mlr / len(train), 100 * train_hit_ovr / len(train)))
        print('Test accuracy:   %5.2f%% (Multinomial), %6.2f%% (One-vs-Rest)'
              % (100 * test_hit_mlr / len(test), 100 * test_hit_ovr / len(test)))
        print('Time: %.1f seconds' % (time.time() - start))
        print('')


class SVM:
    """Support Vector Machine classifier"""
    def __init__(self):
        pass

    def train(self, train):
        svc = SVC()
        svc.fit(train.iloc[:, :-1], train.iloc[:, -1])
        return svc

    def predict(self, feature, svc):
        return svc.predict(feature)

    def __call__(self, train, test):

        # Record start time
        print('========SUPPORT VECTOR MACHINE=========')
        start = time.time()

        # Train support vector machine
        svc = self.train(train)

        # Predict train and test labels
        train_pred = self.predict(train.iloc[:, :-1], svc)
        test_pred = self.predict(test.iloc[:, :-1], svc)

        # Print results
        train_hit = (train.iloc[:, -1] == train_pred).sum()
        test_hit = (test.iloc[:, -1] == test_pred).sum()
        print('Train accuracy:  %5.2f%% (Radial Basis Function)'
              % (100 * train_hit / len(train)))
        print('Test accuracy:   %5.2f%% (Radial Basis Function)'
              % (100 * test_hit / len(test)))
        print('Time: %.1f seconds' % (time.time() - start))
        print('')


class RandomForest:
    """Random Forest classifier"""
    def __init__(self):
        pass

    def train(self, train, trees):

        # Instantiate the classifiers
        if trees == 1:
            rfc = RandomForestClassifier(n_estimators=trees, n_jobs=-1, bootstrap=False)
        else:
            rfc = RandomForestClassifier(n_estimators=trees, n_jobs=-1, bootstrap=True)

        # Train classifier
        rfc.fit(train.iloc[:, :-1], train.iloc[:, -1])

        return rfc

    def predict(self, feature, rfc):
        return rfc.predict(feature)

    def __call__(self, train, test):

        # Record start time
        print('=============RANDOM FOREST=============')
        start = time.time()

        # Train Random Forest classifiers
        rfc1 = self.train(train, 1)
        rfc10 = self.train(train, 10)
        rfc100 = self.train(train, 100)

        # Predict train and test labels
        train_pred_1 = self.predict(train.iloc[:, :-1], rfc1)
        train_pred_10 = self.predict(train.iloc[:, :-1], rfc10)
        train_pred_100 = self.predict(train.iloc[:, :-1], rfc100)
        test_pred_1 = self.predict(test.iloc[:, :-1], rfc1)
        test_pred_10 = self.predict(test.iloc[:, :-1], rfc10)
        test_pred_100 = self.predict(test.iloc[:, :-1], rfc100)

        # Print results
        train_hit_1 = (train.iloc[:, -1] == train_pred_1).sum()
        train_hit_10 = (train.iloc[:, -1] == train_pred_10).sum()
        train_hit_100 = (train.iloc[:, -1] == train_pred_100).sum()
        test_hit_1 = (test.iloc[:, -1] == test_pred_1).sum()
        test_hit_10 = (test.iloc[:, -1] == test_pred_10).sum()
        test_hit_100 = (test.iloc[:, -1] == test_pred_100).sum()
        print('Train accuracy:  %5.2f%% (1 Trees), %6.2f%% (10 Trees), %6.2f%% (100 Trees)'
              % (100 * train_hit_1 / len(train),
                 100 * train_hit_10 / len(train),
                 100 * train_hit_100 / len(train)))
        print('Test accuracy:   %5.2f%% (1 Trees), %6.2f%% (10 Trees), %6.2f%% (100 Trees)'
              % (100 * test_hit_1 / len(test),
                 100 * test_hit_10 / len(test),
                 100 * test_hit_100 / len(test)))
        print('Time: %.1f seconds' % (time.time() - start))
        print('')


class XGBoost:
    """XGBoost"""
    def __init__(self):
        pass

    def train(self, train):
        xgb = XGBClassifier(max_depth=3, n_estimators=100)
        xgb.fit(train.iloc[:, :-1], train.iloc[:, -1])
        return xgb

    def predict(self, feature, xgb):
        return xgb.predict(feature)

    def __call__(self, train, test):

        # Record start time
        print('================XGBOOST================')
        start = time.time()

        # Train support vector machine
        xgb = self.train(train)

        # Predict train and test labels
        train_pred = self.predict(train.iloc[:, :-1], xgb)
        test_pred = self.predict(test.iloc[:, :-1], xgb)

        # Print results
        train_hit = (train.iloc[:, -1] == train_pred).sum()
        test_hit = (test.iloc[:, -1] == test_pred).sum()
        print('Train accuracy:  %5.2f%% (Max Depth of 3, 100 Trees)'
              % (100 * train_hit / len(train)))
        print('Test accuracy:   %5.2f%% (Max Depth of 3, 100 Trees)'
              % (100 * test_hit / len(test)))
        print('Time: %.1f seconds' % (time.time() - start))
        print('')


class NeuralNet:
    """Neural network with a single hidden layer"""
    def __init__(self):
        self.hidden = 100
        self.batch = 50

    def train(self, train):

        # Convert train dataset to tensor
        feature = torch.from_numpy(train.iloc[:, :-1].values).float()
        label = torch.from_numpy(train.iloc[:, -1].values).long()

        # Initialize the neural network
        net = nn.Sequential(nn.Linear(feature.shape[1], self.hidden),
                            nn.ReLU(),
                            nn.Linear(self.hidden, len(set(train.iloc[:, -1]))))

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Train classifier
        for i in range(10000):

            permutation = np.random.choice(feature.shape[0], self.batch)
            optimizer.zero_grad()
            output = net(feature[permutation])
            loss = criterion(output, label[permutation])
            loss.backward()
            optimizer.step()

        return net

    def predict(self, feature, net):
        feature = torch.from_numpy(feature.values).float()
        return net(feature)

    def __call__(self, train, test):

        # Record start time
        print('============NEURAL NETWORK=============')
        start = time.time()

        # Train support vector machine
        net = self.train(train)

        # Predict train and test labels
        train_pred = self.predict(train.iloc[:, :-1], net)
        train_pred = train_pred.detach().numpy().argmax(axis=1)
        test_pred = self.predict(test.iloc[:, :-1], net)
        test_pred = test_pred.detach().numpy().argmax(axis=1)

        # Print results
        train_hit = (train.iloc[:, -1].values == train_pred).sum()
        test_hit = (test.iloc[:, -1].values == test_pred).sum()
        print('Train accuracy:  %5.2f%% (Single Hidden Layer)'
              % (100 * train_hit / len(train)))
        print('Test accuracy:   %5.2f%% (Single Hidden Layer)'
              % (100 * test_hit / len(test)))
        print('Time: %.1f seconds' % (time.time() - start))
        print('')
