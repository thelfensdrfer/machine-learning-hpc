import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import os


# Map strings to integers
def map_strings_to_integers(df, column):
    labels = df[column].unique().tolist()
    mapping = dict(zip(labels, range(len(labels))))
    df.replace({column: mapping}, inplace=True)

    return df


# Load sample dataset (https://www.kaggle.com/spscientist/students-performance-in-exams)
data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/StudentsPerformance.csv', index_col=False)

# Rename columns
data.rename(columns={
    'race/ethnicity': 'race',
    'parental level of education': 'parental_education',
    'test preparation course': 'test_course'
}, inplace=True)

data = map_strings_to_integers(data, 'gender')
data = map_strings_to_integers(data, 'race')
data = map_strings_to_integers(data, 'parental_education')
data = map_strings_to_integers(data, 'lunch')
data = map_strings_to_integers(data, 'test_course')

# Split train and test data
train_data, test_data = train_test_split(data, test_size=0.2)
train_target = train_data['math score']
del train_data['math score']

test_target = test_data['math score']
del test_data['math score']

# Create regression tree
clf = tree.DecisionTreeRegressor()

# Build tree
clf = clf.fit(train_data, train_target)

# Predict first test data
first_prediction = clf.predict(
    test_data.head(1).values
)
print("Prediction (first item): {}".format(first_prediction[0]))
print("Correct math score (first item): {}".format(test_target.iloc[0]))

# Calculate general accuracy
print("General accuracy: {:.2f}".format(clf.score(test_data, test_target)))

# Convert tree into graphviz object
dot_data = tree.export_graphviz(clf, out_file=None)

# Create graph from graphviz object
graph = graphviz.Source(dot_data)

# Render graph
graph.render("tree")
