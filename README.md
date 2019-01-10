## Installation

### Python packages
* Install packages from requirements.txt
    * `pip install -r requirements.txt`
    
### macOS

* Install graphviz: `brew install graphviz` (via [homebrew](https://brew.sh/))

### Ubuntu

* Install graphviz: `sudo apt-get install graphviz`
* Install tkinter: `sudo apt-get install python3-tk`

## Examples

### Decision Trees (sklearn)

https://scikit-learn.org/stable/modules/tree.html

Run: `python 1-examples/decision-trees/index.py`.

Two files will be created:
* 1 `tree` which contains the visualization in text form
* 2 `tree.pdf` which contains the actual visualization of the tree

### K-Nearest-Neighbour (sklearn)

https://scikit-learn.org/stable/modules/neighbors.html

Run `python 1-examples/k-nn/index.py`.

Two plots will open which only differ in the `weights` argument.

> The basic nearest neighbors classification uses uniform weights: that is, the value assigned to a query point is computed from a simple majority vote of the nearest neighbors. Under some circumstances, it is better to weight the neighbors such that nearer neighbors contribute more to the fit. This can be accomplished through the weights keyword. The default value, weights = 'uniform', assigns uniform weights to each neighbor. weights = 'distance' assigns weights proportional to the inverse of the distance from the query point. Alternatively, a user-defined function of the distance can be supplied to compute the weights.

### Neuronal Network (Tensorflow/Keras)

https://www.tensorflow.org/tutorials/keras/basic_classification

Tensorflow only works with python2 or python3.[4-6]. 

Run `python 1-examples/nn/index.py`

### Natural Language Processing

https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

Test data from: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

Run `python 1-examples/nlp/index.py`

## Exercises

### Decision Trees

Test data: https://www.kaggle.com/spscientist/students-performance-in-exams

Task: Create a tree which shows the probable math score of a student when all but the math score columns are given.

### K-Nearest-Neighbour

Test data: https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results

Task: Create a k-nn algorithm that can predict whether a athelete won a medal or not.
Bonus Points: Optimize accuracy (See `3-solutions/k-nn/index.py` "TODO")

### Neuronal Network

Test data: https://www.kaggle.com/moltean/fruits

Task: Create a neuronal network that can identify a fruit in an image.
