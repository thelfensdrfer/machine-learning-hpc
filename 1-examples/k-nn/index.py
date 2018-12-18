import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# Set k
n_neighbors = 15

# Import iris sample data: https://en.wikipedia.org/wiki/Iris_flower_data_set
iris = datasets.load_iris()

# We only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# Step size in the mesh
h = .02

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # We create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # np.meshgrid: Return coordinate matrices from coordinate vectors.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict all values
    # np.c_: Translates slice objects to concatenation along the second axis.
    # p.c_[np.array([1,2,3]), np.array([4,5,6])]
    # array([[1, 4],
    #        [2, 5],
    #        [3, 6]])
    #
    # ndarray.ravel: Return a contiguous flattened array.
    # x = np.array([[1, 2, 3], [4, 5, 6]])
    # x.ravel()
    # [1 2 3 4 5 6]
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    # ndarray.reshape: Gives a new shape to an array without changing its data.
    Z = Z.reshape(xx.shape)
    plt.figure()
    # Draw prediction
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.show()
