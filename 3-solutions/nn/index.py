import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import imageio
import os
import sys


# Import images
def load_images(path):
    files = []

    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            files.append({
                'image': imageio.imread(full_path),
                'label': os.path.basename(os.path.normpath(root))
            })

    print(files[0])
    data = pd.DataFrame(files)

    return data


# Load and split into train and test data
train_images = load_images(os.path.dirname(os.path.realpath(__file__)) + '/Training')
# print(train_images)
sys.exit(0)
