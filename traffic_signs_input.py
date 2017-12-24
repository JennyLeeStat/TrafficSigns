import numpy as np
import random
import pickle
import keras.preprocessing.image as io


BATCH_SIZE = 16


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


def augment_examples(X, y, dist_table, s=1, is_train=True):
    if is_train:
        n_train = len(y)
        count = dist_table[ 'train' ] * n_train
        count = np.array(count, dtype=np.int)
        weights = [ s * np.ceil(max(1, 50000 / c)) for c in count ]

    else:
        n_valid = len(y)
        count = dist_table[ 'valid' ] * n_valid
        count = np.array(count, dtype=np.int)
        weights = [ s * np.ceil(max(1, 20000 / c)) for c in count ]


    X_augmented = [ ]
    y_augmented = [ ]

    for i, feature in enumerate(X):

        n_transform = int(weights[ y[ i ] ])
        for n in range(n_transform):
            image = global_contrast_normalization(feature)
            image = minmax_normalization(image)
            image = random_transform(image)
            X_augmented.append(image)
            y_augmented.append(y[ i ])

    X_augmented = np.array(X_augmented).reshape(len(X_augmented), 32, 32, 3)
    y_augmented = one_hot_encode(y_augmented)
    print("Number of augmented training dataset: {}".format(len(X_augmented)))

    return X_augmented, y_augmented


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