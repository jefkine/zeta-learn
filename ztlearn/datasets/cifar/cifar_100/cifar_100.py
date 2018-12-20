import os
import gzip
import numpy as np
from six.moves import cPickle

from ztlearn.utils import extract_files
from ztlearn.utils import maybe_download
from ztlearn.utils import train_test_split
from ztlearn.datasets.data_set import DataSet

URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

CIFAR_100_BASE_PATH      = '../../ztlearn/datasets/cifar/cifar_100'
CIFAR_100_BATCHES_FOLDER = 'cifar-100-python'

train_files = ['train']
test_files  = ['test']

def fetch_cifar_100(data_target = True):
    extract_files(CIFAR_100_BASE_PATH, maybe_download(CIFAR_100_BASE_PATH, URL))

    if not os.path.exists(os.path.join(CIFAR_100_BASE_PATH, CIFAR_100_BATCHES_FOLDER, train_files[0])):
        raise FileNotFoundError('{} File Not Found'.format(train_files[0])) # dont continue

    if not os.path.exists(os.path.join(CIFAR_100_BASE_PATH, CIFAR_100_BATCHES_FOLDER, test_files[0])):
        raise FileNotFoundError('{} File Not Found'.format(test_files[0])) # dont continue

    with open(os.path.join(CIFAR_100_BASE_PATH, CIFAR_100_BATCHES_FOLDER, train_files[0]),'rb') as file:
        data        = cPickle.load(file, encoding = 'latin1')
        train_data  = np.reshape(data['data'], (data['data'].shape[0], 3, 32, 32))
        train_label = np.reshape(data['fine_labels'], len(data['fine_labels'],))

    with open(os.path.join(CIFAR_100_BASE_PATH, CIFAR_100_BATCHES_FOLDER, test_files[0]),'rb') as file:
        data       = cPickle.load(file, encoding = 'latin1')
        test_data  = np.reshape(data['data'], (data['data'].shape[0], 3, 32, 32))
        test_label = np.reshape(data['fine_labels'], len(data['fine_labels'],))

    if data_target:
        return DataSet(np.concatenate((train_data,  test_data),  axis = 0),
                       np.concatenate((train_label, test_label), axis = 0))
    else:
        return train_data, test_data, train_label, test_label
