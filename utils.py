import os
import sys
import urllib.request
import zipfile
import logging
from collections import Counter
import pandas as pd


logging.basicConfig(
    format='%(levelname)s %(message)s',
    stream=sys.stdout, level=logging.INFO)

def download_and_unzip(url, dest_dir, training_file, validation_file, testing_file):
    zipped_file = dest_dir + ".zip"

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            zipped_file, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    if not os.path.isfile(zipped_file):
        zipped_file, _ = urllib.request.urlretrieve(url, zipped_file, reporthook=_progress)
        statinfo = os.stat(zipped_file)
        logging.info('Successfully downloaded {}'.format(zipped_file))
        logging.info('{} bytes.'.format(statinfo.st_size))

    if not os.path.isfile(training_file):
        logging.info("Unzipping {}".format(zipped_file))
        zipfile.ZipFile(zipped_file, 'r').extractall(dest_dir)
        logging.info("{} is successfully unzipped".format(zipped_file))

    logging.info("Data set \n{}, \n{}, \n{}".format(training_file, validation_file, testing_file))
    logging.info("from url: {}".format(url))
    logging.info("successfully downloaded and unzipped")

    #os.remove(zipped_file)


def get_stats(X, y, dataset_name, is_one_hot=False):
    n = X.shape[ 0 ]
    image_shape = X.shape[ 1: ]

    if is_one_hot:
        n_classes = y.shape[ 1 ]
    else:
        n_classes = len(set(y))

    print("\n====================================")
    print("Dataset: {} ".format(dataset_name))
    print("Number of examples: {}".format(n))
    print("Image data shape: {}".format(image_shape))
    print("Min: {:.4f}".format(X.min()))
    print("Max: {:.4f}".format(X.max()))
    print("Mean: {:.4f}".format(X.mean()))
    print("Std Dev: {:.4f}".format(X.var() ** 0.5))
    print("Number of classes: {}".format(n_classes))


def get_label_dist(labels):
    ratio = {}
    tmp_count = Counter(labels)
    for k, v in tmp_count.items():
        ratio[ k ] = 100 * tmp_count[ k ] / len(labels)

    ratio = pd.DataFrame.from_dict(ratio, 'index').sort_index()
    return ratio
