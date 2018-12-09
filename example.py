# Import modules
from mlm.data import clean_csv, cast_numeric
from mlm.model import Bayes
from sklearn.model_selection import train_test_split

# Clean dataset
data = clean_csv('data/titanic.csv')

# Cast data to numeric representation
data = cast_numeric(data)

# Split into train and test sets
train, test = train_test_split(data, test_size=0.2)

# Run Naive Bayes classifier
bayes = Bayes()
bayes(train, test)
