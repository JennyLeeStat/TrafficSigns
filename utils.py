import os
import sys
import urllib.request
import zipfile
import logging

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

    logging.info("Data set {}, {}, {}".format(training_file, validation_file, testing_file))
    logging.info("from url: {}".format(url))
    logging.info("successfully downloaded and uncompressed")




