#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split



def clean_csv(filename):

    # Import dataset
    try:
        data = pd.read_csv(filename, header=0)
        if data.shape[1] < 2:
            data = pd.read_csv(filename, sep='\s+', header=0)
    except pd.errors.ParserError:
        data = pd.read_csv(filename, sep='\s+', header=0)
    samples = len(data)

    # Categorical and numerical feature totals
    cat_total = data.iloc[:, :-1].select_dtypes(include='object').shape[1]
    num_total = data.iloc[:, :-1].select_dtypes(include='number').shape[1]

    # Preallocate lists of columns to drop
    cat_drop = []
    num_drop = []

    # Flag categorical and numerical features to drop
    for col in data.iloc[:, :-1]:

        # Conditions to check for >10% NaN or identification columns
        is_object = data[col].dtype == 'object'
        is_nan = data[col].isna().sum() > 0.1 * len(data[col])
        is_id = len(set(data[col])) > 0.5 * len(data[col])

        # Flag column for deletion
        if is_nan or (is_object and is_id):
            if is_object:
                cat_drop.append(col)
            else:
                num_drop.append(col)

    # Drop columns
    data = data.drop(columns=cat_drop + num_drop)

    # Drop rows with NaN
    data = data.dropna()

    # Report dropped features and samples
    print('=============DATA CLEANING=============')
    print('Categorical features:  %d/%d (%s dropped)'
          % (cat_total - len(cat_drop), cat_total,
             (str(cat_drop)[1:-1]) if cat_drop else 'none'))
    print('Numerical features:    %d/%d (%s dropped)'
          % (num_total - len(num_drop), num_total,
             (str(num_drop)[1:-1]) if num_drop else 'none'))
    print('Samples for training:  %d/%d (%s dropped)'
          % (len(data), samples,
             str(samples - len(data)) if len(data) < samples else 'none'))
    print('')

    return data


def cast_numeric(data):

    # Categorical features
    data_cat = data.iloc[:, :-1].select_dtypes(include='object')

    # Map categorical features to integer representation
    for col in data_cat:
        map_feature = {y: x for x, y in enumerate(set(data_cat[col]))}
        data_cat[col] = data_cat[col].map(map_feature)

    # Numerical features
    data_num = data.iloc[:, :-1].select_dtypes(include='number').astype(float)

    # Map labels to integer representation
    if len(set(data.iloc[:, -1])) < 0.1 * len(data.iloc[:, -1]):
        map_label = {y: x for x, y in enumerate(set(data.iloc[:, -1]))}
        data_labels = data.iloc[:, -1].map(map_label)
    else:
        data_labels = data.iloc[:, -1]

    # Merge feature and label columns
    data = pd.concat([data_cat, data_num, data_labels], axis=1)

    return data


def one_hot(data):

    # Float features already as one-hot
    data_onehot = data.iloc[:, :-1].select_dtypes(include='float')

    # Integer features to convert to one-hot
    data_int = data.iloc[:, :-1].select_dtypes(include='int')

    # Map integer categorical features to one-hot representation
    for col in data_int:
        col_onehot = pd.get_dummies(data_int[col], prefix=col).astype(float)
        data_onehot = pd.concat([data_onehot, col_onehot], axis=1)

    return data_onehot


def train_test(data):

    # Split into train and test sets
    train, test = train_test_split(data, test_size=0.2)

    # Report Train test split
    print('===========TRAIN TEST SPLIT============')
    print('Train: %d, Test: %d' % (len(train), len(test)))
    print('')

    return train, test
