import numpy as np
import random
import keras.preprocessing.image as io

BATCH_SIZE = 16

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


