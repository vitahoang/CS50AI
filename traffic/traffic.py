import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split

EPOCHS = 20
IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)

labels = {0: 'Speed limit (20km/h)',
          1: 'Speed limit (30km/h)',
          2: 'Speed limit (50km/h)',
          3: 'Speed limit (60km/h)',
          4: 'Speed limit (70km/h)',
          5: 'Speed limit (80km/h)',
          6: 'End of speed limit (80km/h)',
          7: 'Speed limit (100km/h)',
          8: 'Speed limit (120km/h)',
          9: 'No passing',
          10: 'No passing veh over 3.5 tons',
          11: 'Right-of-way at intersection',
          12: 'Priority road',
          13: 'Yield',
          14: 'Stop',
          15: 'No vehicles',
          16: 'Veh > 3.5 tons prohibited',
          17: 'No entry',
          18: 'General caution',
          19: 'Dangerous curve left',
          20: 'Dangerous curve right',
          21: 'Double curve',
          22: 'Bumpy road',
          23: 'Slippery road',
          24: 'Road narrows on the right',
          25: 'Road work',
          26: 'Traffic signals',
          27: 'Pedestrians',
          28: 'Children crossing',
          29: 'Bicycles crossing',
          30: 'Beware of ice/snow',
          31: 'Wild animals crossing',
          32: 'End speed + passing limits',
          33: 'Turn right ahead',
          34: 'Turn left ahead',
          35: 'Ahead only',
          36: 'Go straight or right',
          37: 'Go straight or left',
          38: 'Keep right',
          39: 'Keep left',
          40: 'Roundabout mandatory',
          41: 'End of no passing',
          42: 'End no passing veh > 3.5 tons'}


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose="2")

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        if not os.path.exists(category_dir):
            continue

        for filename in os.listdir(category_dir):
            img_path = os.path.join(category_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(category)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = keras.Sequential([
        keras.layers.Conv2D(32, KERNEL_SIZE, activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        keras.layers.MaxPooling2D(pool_size=POOL_SIZE),
        keras.layers.Conv2D(64, KERNEL_SIZE, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=POOL_SIZE),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == "__main__":
    main()
