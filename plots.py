from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import traffic_signs_input as input_

sign_names = pd.read_csv("signnames.csv")


def display_square_grid(features, n=5, normalize=False):
    plt.figure(figsize=(n, n))
    images = features[: n * n]

    for i, _ in enumerate(images):
        plt.subplot(n, n, i + 1)
        plt.xticks(())
        plt.yticks(())
        image = images[ i ]
        if normalize:
            image = input_.global_contrast_normalization(image)
            image = input_.minmax_normalization(image)
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
        gcn = input_.global_contrast_normalization(image)
        mm = input_.minmax_normalization(gcn)
        plt.imshow(X[ first_43[ n ][ 'image_id' ] ])
        plt.title("{}: {}".format(n, first_43[ n ][ 'name' ]), fontsize=8)
    plt.subplots_adjust(wspace=1, hspace=0.5)
    return first_43
