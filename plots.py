import numpy as np
import matplotlib.pyplot as plt
import random
import cv2


def display_square_grid(features, n=5, seed=42):
    random.seed(seed)
    plt.figure(figsize=(n, n))
    random_index = random.sample(list(range(len(features))), n * n)
    images = np.array([ features[ index ] for index in random_index ]).astype(np.float32)
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    for i, _ in enumerate(images):
        plt.subplot(n, n, i + 1)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(images[ i ])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

