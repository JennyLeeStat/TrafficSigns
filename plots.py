from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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


def get_label_dist(labels):
    ratio = {}
    tmp_count = Counter(labels)
    for k, v in tmp_count.items():
        ratio[ k ] = 100 * tmp_count[ k ] / len(labels)

    ratio = pd.DataFrame.from_dict(ratio, 'index').sort_index()
    return ratio


def show_label_dist(y_train, y_valid, y_test):
    train_ratio = get_label_dist(y_train)
    valid_ratio = get_label_dist(y_valid)
    test_ratio = get_label_dist(y_test)

    dist_table = pd.concat([ train_ratio, valid_ratio, test_ratio ], axis=1)
    dist_table.columns = [ 'train', 'valid', 'test' ]
    ax = dist_table.plot(kind='bar', figsize=(8, 4.5))
    ax.set_ylabel('Ratio (%)', fontsize=12)
    ax.set_xlabel('Class labels', fontsize=12)
    ax.set_title('Class label distributions for three datasets', fontsize=17);



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
    plt.title("Grayscale \nand GCN")
    image = inputs.convert_to_grayscale_3d(img)
    image = inputs.global_contrast_normalization(image)
    image = inputs.minmax_normalization(image)
    plt.imshow(np.squeeze(image), cmap='Greys')

    plt.subplot(1, 4, 4)
    plt.xticks(())
    plt.yticks(())
    plt.title("GCN")
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



def display_conf_matrix(final_test_stats):
    plt.figure(figsize=(10, 10))
    conf_mat = confusion_matrix(final_test_stats[ 'true_labels' ], final_test_stats[ 'preds' ])
    norm_conf_mat = conf_mat / conf_mat.sum(axis=1)
    np.fill_diagonal(norm_conf_mat, 0)
    plt.matshow(norm_conf_mat, fignum=1)
    plt.xticks(list(range(43)))
    plt.yticks(list(range(43)))
    plt.xlabel("Predicted labels", fontweight='bold')
    plt.ylabel("True labels", fontweight='bold')
    plt.title("Normalized confusion matrix (TP set zero)")
    plt.show()