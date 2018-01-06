from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras.preprocessing.image as io
import traffic_signs_input as inputs


def display_square_grid(features, normalize=False, greys=False):
    plt.figure(figsize=(8, 4.5))
    images = features[ : 16 * 9 ]

    for i, _ in enumerate(images):
        plt.subplot(9, 16, i + 1)
        plt.xticks(())
        plt.yticks(())
        image = images[ i ]

        if greys and not normalize:
            image = inputs.convert_to_grayscale(image)
            plt.imshow(image, cmap='Greys')

        if not greys and normalize:
            image = inputs.global_contrast_normalization(image)
            image = inputs.minmax_normalization(image)
            plt.imshow(image)

        if greys and normalize:
            image = inputs.convert_to_grayscale_3d(image)
            image = inputs.global_contrast_normalization(image)
            image = inputs.minmax_normalization(image)
            plt.imshow(np.squeeze(image), cmap='Greys')

        if not greys and not normalize:
            plt.imshow(image)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def display_w_labels(X, label, sign_names):
    first_43 = defaultdict(dict)
    for i, classID in enumerate(label):
        if classID not in first_43:
            first_43[ classID ][ 'image_id' ] = i
            first_43[ classID ][ 'name' ] = sign_names[ 'SignName' ][ classID ]
        if len(first_43) == 43:
            break

    plt.figure(figsize=(9, 16))
    for n in range(43):
        plt.subplot(15, 3, n + 1)
        plt.xticks(())
        plt.yticks(())
        image = X[ first_43[ n ][ 'image_id' ] ]
        gcn = inputs.global_contrast_normalization(image)
        mm = inputs.minmax_normalization(gcn)
        plt.imshow(X[ first_43[ n ][ 'image_id' ] ])
        plt.title("{}: {}".format(n, first_43[ n ][ 'name' ]), fontsize=8)
    plt.subplots_adjust(wspace=1, hspace=0.5)
    return first_43


def compare_preprocessing_methods(img):
    plt.figure(figsize=(8, 4.5))
    plt.subplot(1, 4, 1)
    plt.xticks(())
    plt.yticks(())
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1, 4, 2)
    plt.xticks(())
    plt.yticks(())
    plt.title("Grayscale")
    plt.imshow(inputs.convert_to_grayscale(img), cmap='Greys')

    plt.subplot(1, 4, 3)
    plt.xticks(())
    plt.yticks(())
    plt.title("Grayscale \nand normalized")
    image = inputs.convert_to_grayscale_3d(img)
    image = inputs.global_contrast_normalization(image)
    image = inputs.minmax_normalization(image)
    plt.imshow(np.squeeze(image), cmap='Greys')

    plt.subplot(1, 4, 4)
    plt.xticks(())
    plt.yticks(())
    plt.title("Normalized")
    gcn = inputs.global_contrast_normalization(img)
    mm = inputs.minmax_normalization(gcn)
    plt.imshow(mm);

def show_preprocessed_example(X_train):
    x = X_train[5]
    row_axis, col_axis, channel_axis = 0, 1, 2

    # random rotation
    x_r = io.random_rotation(x, rg=45,
                             row_axis=row_axis,
                             col_axis=col_axis,
                             channel_axis=channel_axis)

    # random horizontal shift
    x_ws = io.random_shift(x, wrg=.2, hrg=0,
                           row_axis=row_axis,
                           col_axis=col_axis,
                           channel_axis=channel_axis)

    # random vertical shift
    x_hs = io.random_shift(x, wrg=0, hrg=.2,
                           row_axis=row_axis,
                           col_axis=col_axis,
                           channel_axis=channel_axis)

    # random shear
    x_s = io.random_shear(x, intensity=.2,
                          row_axis=row_axis,
                          col_axis=col_axis,
                          channel_axis=channel_axis)

    # random zoom
    x_z = io.random_zoom(x, zoom_range=(.8, 1.2),
                         row_axis=row_axis,
                         col_axis=col_axis,
                         channel_axis=channel_axis)

    images = [x, x_r, x_ws, x_hs, x_s, x_z]
    titles = ["Original", "Rotate", "Horizontal shift",
             "Vertical shift", "Shear", "Zoom"]

    plt.figure(figsize=(8, 4.5))
    for i, image in enumerate(images):
        plt.subplot(2, 3, i + 1)
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
        image = inputs.global_contrast_normalization(image)
        image = inputs.minmax_normalization(image)
        plt.imshow(image)

    plt.show()


def display_sample_images(path1, path2, path3, path4, path5):

    blank = np.zeros((32, 32 * 5))
    for i in range(5):
        img = plt.imshow(path1)
        resized = cv2.resize(img, (32, 32))
        blank[:, 32*i: 32 * (i + 1)] = resized

    return blank

path_1 = 'images/test_1.png'
path_2 = 'images/test_2.png'
path_3 = 'images/test_3.png'
path_4 = 'images/test_4.png'
path_5 = 'images/test_5.png'