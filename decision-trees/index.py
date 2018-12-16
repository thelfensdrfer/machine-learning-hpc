import graphviz
from sklearn import tree
from sklearn.datasets import load_iris

# Load sample dataset (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
iris = load_iris()

# Create decision tree classifier
# Alternative: tree.DecisionTreeRegressor()
clf = tree.DecisionTreeClassifier()

# Build the tree
# iris.data is a multidimensional array with features
# [
#  [1, 2, 3, 4]
# ]
clf = clf.fit(iris.data, iris.target)

# Predict class with the following 4 features
predictions = clf.predict([
    [6.6, 2.4, 5.7, 1.7]
])
print("Prediction: {}".format(predictions[0]))

# Get the propabilities of each prediction
predictions = clf.predict_proba([
    [6.6, 2.4, 5.7, 1.7]
])
print("Probabilities (3 classes): {}".format(predictions[0]))

# Convert tree into graphviz object
dot_data = tree.export_graphviz(clf, out_file=None)

# Create graph from graphviz object
graph = graphviz.Source(dot_data)

# Render graph
graph.render("tree")
