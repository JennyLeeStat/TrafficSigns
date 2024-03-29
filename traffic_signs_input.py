import os
import sys
import numpy as np
import random
import pickle
import keras.preprocessing.image as io
from sklearn.utils import shuffle
import cv2
import utils

import plots

BATCH_SIZE = 16

def convert_to_grayscale_3d(x):
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return np.expand_dims(x, 2)

def convert_to_grayscale(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def one_hot_encode(x, n_classes=43):
    encoded = np.zeros((len(x), n_classes), dtype=np.int)
    for i, xi in enumerate(x):
        encoded[ i ][ xi ] = 1

    return encoded

def minmax_normalization(x, eps=1e-8):
    '''
    Squash image intensities into range [0, 1]  
    :param x: 3D image data
    :param eps: small number to prevent illegal division 
    :return: normalized in range [0, 1]
    '''
    return (x - x.min()) / ((x.max() - x.min()) + eps)

def global_contrast_normalization(x, s=1, eps=1e-8, lambda_=1):
    return s * (x - x.mean()) / (max(eps, (((x - x.mean())** 2).mean() + lambda_) ** .5))


def random_transform(x, seed=42, rotation_range=30,
                     width_shift_range=.1, height_shift_range=.1,
                     shear_range=0.2, zoom_range=(.9, 1.1),
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
                           row_axis=row_axis,
                           col_axis=col_axis,
                           channel_axis=channel_axis)

    x = io.random_shift(x, width_shift_range,
                        height_shift_range,
                        row_axis=row_axis,
                        col_axis=col_axis,
                        channel_axis=channel_axis)

    x = io.random_shear(x, shear_range,
                        row_axis=row_axis,
                        col_axis=col_axis,
                        channel_axis=channel_axis)

    x = io.random_zoom(x, zoom_range,
                       row_axis=row_axis,
                       col_axis=col_axis,
                       channel_axis=channel_axis)

    return x

def preprocess_and_save(features, labels, is_color,
                        dest, random_perturb=False,
                        save_output=False):
    if is_color:
        n_channel = 3

    else:
        n_channel = 1

    preprocessed = [ ]
    for i, image in enumerate(features):
        if random_perturb:
            if random.uniform(0, 1) > .9:
                image = random_transform(image)

        if not is_color:
            image = convert_to_grayscale(image)

        image = global_contrast_normalization(image)
        image = minmax_normalization(image)
        preprocessed.append(image)

    preprocessed = np.array(preprocessed).reshape(
        (len(preprocessed), 32, 32, n_channel))

    if save_output:
        pickle.dump((preprocessed, labels), open(dest, 'wb'))

    return preprocessed, labels


def augment_examples(X, y, s=1,
                     weights_on_scarce=.75, dataset_name="",
                     random_state=42, show_stats=True):
    '''
    generate randomly transformed example images
    :param X: list of 3D image dataset, shape = (r, c, channel)
    :param y: labels
    :param s: integer scale factor 
    :parma weights_on_scarse: correction factor on scarce label classes
    :param dataset_name: name of dataset being boosted  
    :param random_state
    :return: len(X) * s number of boosted examples
    '''

    n_channel = X[ 0 ].shape[ -1 ]
    n_train = len(y)

    train_freq = plots.get_label_dist(y)
    train_freq_normalized = minmax_normalization(train_freq, eps=1e-8)
    n_transform_list = np.floor((1 - weights_on_scarce * train_freq_normalized) * s)

    X_augmented = [ ]
    y_augmented = [ ]

    for i, image in enumerate(X):
        sys.stdout.write('\r>> Augmenting image %s (%.1f%%)' % (
            str(i), float(i + 1) / float(n_train) * 100.0))
        sys.stdout.flush()
        n_transform = int(n_transform_list[ 0 ][ y[ i ] ])

        for j in range(n_transform):
            image = random_transform(image)
            image = minmax_normalization(image)

            X_augmented.append(image)
            y_augmented.append(y[ i ])

    X_augmented = np.array(X_augmented).reshape(len(X_augmented), 32, 32, n_channel)
    X_augmented, y_augmented = shuffle(X_augmented, y_augmented, random_state=random_state)

    if show_stats:
        utils.get_stats(X_augmented, y_augmented, dataset_name)

    return X_augmented, y_augmented


def preprocess_test_examples():
    preprocessed = [ ]

    for i in range(5):
        img_path = "images/test_" + str(i + 1) + ".png"
        image = plt.imread(img_path)

        # upsample or downsample to 32 * 32 * 3
        image = cv2.resize(image, (32, 32))

        # grayscaling
        image = convert_to_grayscale_3d(image)
        image = global_contrast_normalization(image)
        image = minmax_normalization(image)
        preprocessed.append(np.expand_dims(image, 0))

    return np.concatenate(preprocessed, 0)


def prepare_arugment_dataset(is_color=True, s=4):

    # Download and unzip the file
    data_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
    dest_dir = "traffic-signs-data"
    training_file = os.path.join(dest_dir, "train.p")
    validation_file = os.path.join(dest_dir, "valid.p")
    testing_file = os.path.join(dest_dir, "test.p")
    utils.download_and_unzip(data_url, dest_dir, training_file, validation_file, testing_file)

    # Load pickled data
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train[ 'features' ], train[ 'labels' ]
    X_valid, y_valid = valid[ 'features' ], valid[ 'labels' ]
    X_test, y_test = test[ 'features' ], test[ 'labels' ]



    # Shuffle the datasets
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_valid, y_valid = shuffle(X_valid, y_valid, random_state=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=0)


    # GCN/ grayscale/ MinMax preprocessing
    X_train_color, y_train_color = preprocess_and_save(
        X_train, y_train, True, "traffic-signs-data/train_preprocessed_color.p")

    X_train_gray, y_train_gray = preprocess_and_save(
        X_train, y_train, False, "traffic-signs-data/train_preprocessed_gray.p")

    X_valid_color, y_valid_color = preprocess_and_save(
        X_valid, y_valid, True, "traffic-signs-data/valid_preprocessed_color.p", True, False)

    X_valid_gray, y_valid_gray = preprocess_and_save(
        X_valid, y_valid, False, "traffic-signs-data/valid_preprocessed_gray.p", True, False)

    X_test_color, y_test_color = preprocess_and_save(
        X_test, y_test, True, "traffic-signs-data/test_preprocessed_color.p")

    X_test_gray, y_test_gray = preprocess_and_save(
        X_test, y_test, False, "traffic-signs-data/test_preprocessed_gray.p")


    # Augment the examples
    if is_color:
        X_train_augmented_color, y_train_augmented_color = augment_examples(
            X_train_color, y_train_color, s, dataset_name="Train_augmented (color)")
        X_train_large_color = np.concatenate((X_train_color, X_train_augmented_color), axis=0)
        y_train_large_color = np.concatenate((y_train_color, y_train_augmented_color), axis=0)
        X_train_large_color, y_train_large_color = shuffle(X_train_large_color, y_train_large_color, random_state=42)
        utils.get_stats(X_train_large_color, y_train_large_color, "Original + Train_augmented (color)")
        return X_train_large_color, y_train_large_color, X_valid_color, y_valid_color, \
                X_test_color, y_test_color

    if not is_color:
        X_train_augmented_gray, y_train_augmented_gray = augment_examples(
            X_train_gray, y_train_gray, s, dataset_name="Train_augmented (gray)")
        X_train_large_gray = np.concatenate((X_train_gray, X_train_augmented_gray), axis=0)
        y_train_large_gray = np.concatenate((y_train_gray, y_train_augmented_gray), axis=0)
        X_train_large_gray, y_train_large_gray = shuffle(X_train_large_gray, y_train_large_gray, random_state=42)
        utils.get_stats(X_train_large_gray, y_train_large_gray, "Original + Train_augmented (gray)")
        return X_train_large_gray, y_train_large_gray, X_valid_gray, y_valid_gray, \
                X_test_gray, y_test_gray
