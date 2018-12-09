#!/usr/bin/env python3

import pandas as pd


def clean_csv(filename):

    # Import dataset
    data = pd.read_csv(filename, header=0)
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
    print('==========DATA CLEANING==========')
    print('Categorical features:  %d/%d (%s dropped)'
          % (cat_total - len(cat_drop), cat_total,
             (str(cat_drop)[1:-1]) if cat_drop else 'none'))
    print('Numerical features:    %d/%d (%s dropped)'
          % (num_total - len(num_drop), num_total,
             (str(num_drop)[1:-1]) if num_drop else 'none'))
    print('Samples:               %d/%d (%s dropped)'
          % (len(data), samples,
             str(samples - len(data)) if len(data) < samples else 'none'))

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
    if data.iloc[:, -1].dtype == 'object':
        map_label = {y: x for x, y in enumerate(set(data.iloc[:, -1]))}
        data_labels = data.iloc[:, -1].map(map_label)
    else:
        data_labels = data.iloc[:, -1]

    # Merge feature and label columns
    data = pd.concat([data_cat, data_num, data_labels], axis=1)

    return data
