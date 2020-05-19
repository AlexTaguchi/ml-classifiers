#!/usr/bin/env python3

# Modules
import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocess:
    """Preprocess dataframe for machine learning"""
    def __call__(self, data):
        data = self.clean_csv(data)
        data = self.cast_numeric(data)
        data = self.one_hot(data)
        data = self.train_test(data)

        return data

    def clean_csv(self, data):

        # Count total number of samples
        samples = len(data)

        # Count total number of categorical and numerical features
        cat_total = data.iloc[:, :-1].select_dtypes(include='object').shape[1]
        num_total = data.iloc[:, :-1].select_dtypes(include='number').shape[1]

        # Preallocate lists of columns to drop
        cat_drop = []
        num_drop = []

        # Flag categorical and numerical features to drop
        for col in data.iloc[:, :-1]:

            # Conditions to check for excessive NaN or excessive categorical feature diversity
            nan = data[col].isna().sum() > 0.1 * len(data[col])
            diverse = data[col].dtype == 'object' and len(set(data[col])) > 0.5 * len(data[col])

            # Flag columns for deletion
            if nan or diverse:
                if data[col].dtype == 'object':
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

    def cast_numeric(self, data):

        # Identify categorical and numerical columns
        data_cat = data.iloc[:, :-1].select_dtypes(include='object')
        data_num = data.iloc[:, :-1].select_dtypes(include='number')

        # Map categorical features to integer representation
        for col in data_cat:
            map_feature = {y: x for x, y in enumerate(data_cat[col].unique())}
            data_cat[col] = data_cat[col].map(map_feature)

        # Map class labels to integer representation
        map_label = {y: x for x, y in enumerate(data.iloc[:, -1].unique())}
        data_labels = data.iloc[:, -1].map(map_label)

        # Merge feature and label columns
        data = pd.concat([data_cat, data_num, data_labels], axis=1)

        return data
    
    def one_hot(self, data):

        # Float and high diversity integer features already fine as is
        data_float = data.iloc[:, :-1].select_dtypes(include='float')
        data_int = data.iloc[:, :-1].select_dtypes(include='int')
        numeric_int = [len(data_int[x].unique()) > (0.1 * len(data)) for x in data_int]
        data_int_num = data_int[data_int.columns[numeric_int]]

        # Convert low diversity integer features to one-hot
        data_int_cat = []
        for col in data_int.columns[[not x for x in numeric_int]]:
            data_int_cat.append(pd.get_dummies(data_int[col], prefix=col).astype(int))
        data_int_cat = pd.concat(data_int_cat, axis=1) if data_int_cat else pd.DataFrame()

        # Concatenate features and labels
        data = pd.concat([data_float, data_int_num, data_int_cat, data.iloc[:, -1]], axis=1)

        return data
    
    def train_test(self, data):

        # Split into train and test sets
        train, test = train_test_split(data, test_size=0.2)

        # Report Train test split
        print('===========TRAIN TEST SPLIT============')
        print('Train: %d, Test: %d' % (len(train), len(test)))
        print('')

        return train, test
