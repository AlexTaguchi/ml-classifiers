#!/usr/bin/env python3

# Modules
from sklearn.naive_bayes import GaussianNB, MultinomialNB


class Bayes:
    """
    Naive Bayes classifier
    """

    def __init__(self):
        self.categorical = 'object'
        self.numerical = 'number'

    def cat_num_split(self, feature):

        # Split features into categorical and numberical types
        feature_cat = feature.select_dtypes(include=self.categorical)
        feature_num = feature.select_dtypes(include=self.numerical)

        return feature_cat, feature_num

    def train(self, train_cat, train_num, train_labels):

        # Instantiate the classifier
        mnb = MultinomialNB()
        gnb = GaussianNB()

        # Train classifier
        if train_cat.shape[1]:
            mnb.fit(train_cat, train_labels)
        if train_num.shape[1]:
            gnb.fit(train_num, train_labels)

        return mnb, gnb

    def predict(self, feature_cat, feature_num, mnb, gnb):

        # Predict train labels
        cat_pred = mnb.predict_proba(feature_cat) if feature_cat.shape[1] else 0
        cat_pred *= feature_cat.shape[1]
        num_pred = gnb.predict_proba(feature_num) if feature_num.shape[1] else 0
        num_pred *= feature_num.shape[1]

        return (cat_pred + num_pred).argmax(axis=1)

    def __call__(self, train, test):

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
        print('==========MODEL PERFORMANCE==========')
        train_hit = (train_labels == train_pred).sum()
        test_hit = (test_labels == test_pred).sum()
        print('Train accuracy:  %d/%d (%.2f%%)'
              % (train_hit, len(train), 100 * train_hit / len(train)))
        print('Test accuracy:   %d/%d (%.2f%%)'
              % (test_hit, len(test), 100 * test_hit / len(test)))
