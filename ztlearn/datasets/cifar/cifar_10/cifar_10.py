import os
import gzip
import numpy as np
from six.moves import cPickle

from ztlearn.utils import extract_files
from ztlearn.utils import maybe_download
from ztlearn.utils import train_test_split
from ztlearn.datasets.data_set import DataSet

URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

CIFAR_10_BASE_PATH      = '../../ztlearn/datasets/cifar/cifar_10'
CIFAR_10_BATCHES_FOLDER = 'cifar-10-batches-py'

train_files = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]
test_files = ['test_batch']

def fetch_cifar_10(data_target = True):
    extract_files(CIFAR_10_BASE_PATH, maybe_download(CIFAR_10_BASE_PATH, URL))

    for train_file in train_files:
        if not os.path.exists(os.path.join(CIFAR_10_BASE_PATH, CIFAR_10_BATCHES_FOLDER, train_file)):
            print('{} File Not Found'.format(train_file)) # dont continue

    train_data  = np.zeros((50000, 3, 32, 32), dtype = 'uint8')
    train_label = np.zeros((50000,), dtype = 'uint8')
    for idx, train_file in enumerate(train_files):

        with open(os.path.join(CIFAR_10_BASE_PATH, CIFAR_10_BATCHES_FOLDER, train_file),'rb') as file:
            data        = cPickle.load(file, encoding = 'latin1')
            batch_data  = data['data'].reshape((-1, 3, 32, 32)).astype('float32')
            batch_label = np.reshape(data['labels'], len(data['labels'],))

        train_data[idx * 10000: (idx + 1) * 10000, ...] = batch_data
        train_label[idx * 10000: (idx + 1) * 10000]     = batch_label

    with open(os.path.join(CIFAR_10_BASE_PATH, CIFAR_10_BATCHES_FOLDER, test_files[0]),'rb') as file:
        data       = cPickle.load(file, encoding = 'latin1')
        test_data  = data['data'].reshape((-1, 3, 32, 32)).astype('float32')
        test_label = np.reshape(data['labels'], len(data['labels'],))

        if data_target:
            return DataSet(np.concatenate((train_data,  test_data),  axis = 0),
                           np.concatenate((train_label, test_label), axis = 0))
        else:
            return train_data, test_data, train_label, test_label
