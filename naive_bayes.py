# Modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Import dataset
data = pd.read_csv('iris.csv', header=0)
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

# Split dataset into training and test datasets
train, test = train_test_split(data, test_size=0.2)

# Categorical features
train_cat = train.iloc[:, :-1].select_dtypes(include='object')
test_cat = test.iloc[:, :-1].select_dtypes(include='object')

# Numerical features
train_num = train.iloc[:, :-1].select_dtypes(include='number')
test_num = test.iloc[:, :-1].select_dtypes(include='number')

# Map categorical features to integer representation
for col in train_cat:
    map_feature = {y: x for x, y in enumerate(set(train_cat[col]))}
    train_cat[col] = train_cat[col].map(map_feature)
for col in test_cat:
    map_feature = {y: x for x, y in enumerate(set(test_cat[col]))}
    test_cat[col] = test_cat[col].map(map_feature)

# Map labels to integer representation
map_label = {y: x for x, y in enumerate(set(data.iloc[:, -1]))}
train_labels = train.iloc[:, -1].map(map_label)
test_labels = test.iloc[:, -1].map(map_label)

# Instantiate the classifier
mnb = MultinomialNB()
gnb = GaussianNB()

# Train classifier
if train_cat.shape[1]:
    mnb.fit(train_cat, train_labels)
if train_num.shape[1]:
    gnb.fit(train_num, train_labels)

# Predict train labels
train_cat_pred = mnb.predict_proba(train_cat) if cat_total - len(cat_drop) else 0
train_cat_pred *= cat_total - len(cat_drop)
train_num_pred = gnb.predict_proba(train_num) if num_total - len(num_drop) else 0
train_num_pred *= num_total - len(num_drop)
train_pred = (train_cat_pred + train_num_pred).argmax(axis=1)

# Predict test labels
test_cat_pred = mnb.predict_proba(test_cat) if cat_total - len(cat_drop) else 0
test_cat_pred *= cat_total - len(cat_drop)
test_num_pred = gnb.predict_proba(test_num) if num_total - len(num_drop) else 0
test_num_pred *= num_total - len(num_drop)
test_pred = (test_cat_pred + test_num_pred).argmax(axis=1)

# Print results
print('==========MODEL PERFORMANCE==========')
train_hit = (train_labels == train_pred).sum()
test_hit = (test_labels == test_pred).sum()
print('Train accuracy:  %d/%d (%.2f%%)'
      % (train_hit, len(train), 100 * train_hit / len(train)))
print('Test accuracy:   %d/%d (%.2f%%)'
      % (test_hit, len(test), 100 * test_hit / len(test)))
