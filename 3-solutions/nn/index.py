import tensorflow as tf
from tensorflow import keras
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

import os


class_names = []


# Import images
def load_images(path, set_class_names=True):
    print("Reading images from {}".format(path))
    images = []
    labels = []

    i = 0

    for root, directories, filenames in os.walk(path):
        if set_class_names:
            for directory in directories:
                class_names.append(directory)

        for filename in filenames:
            full_path = os.path.join(root, filename)
            # Load image from file and convert to grayscale
            images.append(io.imread((full_path)))
            labels.append(
                class_names.index(os.path.basename(os.path.normpath(root)))
            )

            i += 1

    return np.array(images), np.array(labels)


# Load and split into train and test data
train_images, train_labels = load_images(os.path.dirname(os.path.realpath(__file__)) + '/Training')
test_images, test_labels = load_images(os.path.dirname(os.path.realpath(__file__)) + '/Test', set_class_names=False)

train_images = train_images / 255.0
test_images = test_images / 255.0

# Create network layers
model = keras.Sequential([
    # https://keras.io/layers/convolutional/
    # input_shape = 100x100 pixels with 3 values (r, g, b)
    keras.layers.Conv2D(16, 3, input_shape=(100, 100, 3), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    # Optimizer: http://ruder.io/optimizing-gradient-descent/index.html#adam | https://arxiv.org/abs/1412.6980
    optimizer=tf.train.AdamOptimizer(),
    # Loss function: https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_images, train_labels, epochs=10, batch_size=128)

# Evaluate model
test_loss, test_acc = model.evaluate([test_images], [test_labels])
print('Test accuracy: ', test_acc)

# Predict test images
predictions = model.predict([test_images])


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
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
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
_ = plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.show()
