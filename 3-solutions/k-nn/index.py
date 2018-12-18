from sklearn import neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# TODO: Optimize for many 0 y-values
# https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

# Map strings to integers
def map_strings_to_integers(df, column):
    labels = df[column].unique().tolist()
    mapping = dict(zip(labels, range(len(labels))))
    df.replace({column: mapping}, inplace=True)

    return df


# Read data
data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/athlete_events.csv')

# Remove ID because it is used as index
del data['ID']

# Remove name because irrelevant
del data['Name']

# Remove name because already present in Year and Season
del data['Games']

# Remove name because already present in NOC
del data['Team']

print('Preparing data...')
data = map_strings_to_integers(data, 'Sex')
data = map_strings_to_integers(data, 'NOC')
data = map_strings_to_integers(data, 'Season')
data = map_strings_to_integers(data, 'City')
data = map_strings_to_integers(data, 'Sport')
data = map_strings_to_integers(data, 'Event')
data.replace({'Medal': {
    'Bronze': 1,
    'Silver': 2,
    'Gold': 3,
}}, inplace=True)

# Fill all np.NaN with 0
data = data.fillna(0)

# Split training and test data
train_data, test_data = train_test_split(data, test_size=0.2)
train_target = train_data['Medal']
del train_data['Medal']

test_target = test_data['Medal']
del test_data['Medal']

# Number of neighbors used
n_neighbors = 15

# We create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

print('Fitting model...')
clf.fit(train_data, train_target)

# Predict first test data
first_prediction = clf.predict(
    test_data.head(1).values
)
print("Prediction (first item): {}".format(first_prediction[0]))
print("Medal (first item; 0 = None; 1 = Bronze; 2 = Silver; 3 = Gold): {}".format(test_target.iloc[0]))

# Calculate general accuracy
print("General accuracy: {:.2f}".format(clf.score(test_data, test_target)))
