import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# Import fashion mnist dataset (https://github.com/zalandoresearch/fashion-mnist)
# 10 categories of grayscaled clothes
fashion_mnist = keras.datasets.fashion_mnist

# Class names [0-9]
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load and split into train and test data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the pixels (it is already a grayscale value between 0 and 255)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create network layers
model = keras.Sequential([
    # Flatten the input
    # Here the input shape is 28x28 (in this case pixel)
    # The result will be a vecotor? with 28*28 = 784 rows
    keras.layers.Flatten(input_shape=(28, 28)),
    # Densen the 784 row vector into 128 rows with an relu activation function (https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Densen the 128 rows into 10 rows (10 categories)
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model: https://stackoverflow.com/a/47996024/1100089
model.compile(
    # Optimizer: http://ruder.io/optimizing-gradient-descent/index.html#adam | https://arxiv.org/abs/1412.6980
    optimizer=tf.train.AdamOptimizer(),
    # Loss function: https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
# Epohcs: https://stackoverflow.com/a/44907684/1100089
model.fit(train_images, train_labels, epochs=5)

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)

# Predict test images
predictions = model.predict(test_images)


# Helper function to show an image with a label
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# Helper function to show an bar graph
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Show prediction of first image (left) and accuracies (right)

# Image
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)

# Values
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# Show detailed predictions of the first image

img = test_images[0]
# Add image to a list because keras is optimized for batch predictions
img = (np.expand_dims(img, 0))
# Predict category
predictions_single = model.predict(img)
# Create plot
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()