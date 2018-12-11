# Import modules
from mlm.data import clean_csv, cast_numeric, train_test
from mlm.model import NaiveBayes, Logistic, RandomForest

# Clean dataset
data = clean_csv('data/titanic.csv')

# Cast data to numeric representation
data = cast_numeric(data)

# Split into train and test sets
train, test = train_test(data)

# Naive Bayes classifier
bayes = NaiveBayes()
bayes(train, test)

# Logistic Regression classifier
logistic = Logistic()
logistic(train, test)

# Random Forest classifier
forest = RandomForest()
forest(train, test)
