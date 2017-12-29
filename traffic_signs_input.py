import sys
import numpy as np
import random
import pickle
import keras.preprocessing.image as io
from sklearn.utils import shuffle
import cv2
import utils

BATCH_SIZE = 16

def convert_to_grayscale(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def convert_to_grayscale_3d(x):
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return np.expand_dims(x, 2)


def one_hot_encode(x, n_classes=43):
    encoded = np.zeros((len(x), n_classes), dtype=np.int)
    for i, xi in enumerate(x):
        encoded[ i ][ xi ] = 1

    return encoded

def minmax_normalization(x):
    return (x - x.min()) / (x.max() - x.min())

def global_contrast_normalization(x, s=1, eps=1e-8, _lambda=1):
    return s * (x - x.mean()) / (max(eps, (((x - x.mean())** 2).mean() + _lambda) ** .5))


def random_transform(x, seed=42, rotation_range=40,
                     width_shift_range=.2, height_shift_range=.2,
                     shear_range=.2, zoom_range=(.8, 1.2),
                     row_axis=0, col_axis=1, channel_axis=2):
    '''
    https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
    :param x: 3D input image 
    :param seed: 
    :param rotation_range: Rotation range, in degrees.
    :param width_shift_range: Width shift range, as a float fraction of the width.
    :param height_shift_range: Height shift range, as a float fraction of the height.
    :param shear_range: Transformation intensity
    :param zoom_range: Tuple of floats; zoom range for width and height.
    :return: transformed image that has the same shape
    '''
    random.seed(seed)

    x = io.random_rotation(x, rotation_range,
                           row_axis=row_axis, col_axis=col_axis, channel_axis=channel_axis)

    x = io.random_shift(x, width_shift_range, height_shift_range,
                        row_axis=row_axis, col_axis=col_axis, channel_axis=channel_axis)

    x = io.random_shear(x, shear_range,
                        row_axis=row_axis, col_axis=col_axis, channel_axis=channel_axis)

    x = io.random_zoom(x, zoom_range,
                       row_axis=row_axis, col_axis=col_axis, channel_axis=channel_axis)

    return x



def preprocess_and_save(features, labels, dest):
    preprocessed = [ ]
    for i, feature in enumerate(features):
        feature = global_contrast_normalization(feature)
        preprocessed.append(minmax_normalization(feature))

    preprocessed = np.array(preprocessed).reshape(
        (len(preprocessed), 32, 32, 3))
    encoded_label = one_hot_encode(labels)
    pickle.dump((preprocessed, encoded_label), open(dest, 'wb'))

    return preprocessed, encoded_label


def augment_examples(X, y, s=1, is_color=True,
                     dataset_name="", random_state=42):
    '''
    
    :param X: list of 3D image dataset, shape = (r, c, channel)
    :param y: labels
    :param s: integer scale factor 
    :param is_color: 
    :param dataset_name: name of dataset being boosted  
    :param random_state
    :return: len(X) * s number of boosted examples
    '''
    n = len(y)
    n_channel = X[0].shape[-1]
    X_augmented = [ ]
    y_augmented = [ ]


    for i, image in enumerate(X):
        sys.stdout.write('\r>> Augmenting image %s (%.1f%%)' % (
            str(i), float(i + 1) / float(n) * 100.0))
        sys.stdout.flush()

        for j in range(s):
            if n_channel == 3: # color examples
                image = global_contrast_normalization(image)
                image = random_transform(image)
                image = minmax_normalization(image)

            if n_channel == 1: # grayscale examples
                image = random_transform(image)
                image = minmax_normalization(image)

            X_augmented.append(image)
            y_augmented.append(y[ i ])

    X_augmented = np.array(X_augmented).reshape(len(X_augmented), 32, 32, n_channel )
    X_augmented, y_augmented = shuffle(X_augmented, y_augmented, random_state=random_state)

    #     print("\nNumber of examples after augmentation : {}".format(len(X_augmented)))
    utils.get_stats(X_augmented, y_augmented, dataset_name)

    return X_augmented, y_augmented
