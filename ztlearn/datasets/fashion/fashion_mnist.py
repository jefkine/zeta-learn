import gzip
import numpy as np

from ..data_set import DataSet
from ztlearn.utils import maybe_download

URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

train_files = {
    'train_labels' : 'train-labels-idx1-ubyte.gz',
    'train_data'   : 'train-images-idx3-ubyte.gz'
}

test_files = {
    'test_labels' : 't10k-labels-idx1-ubyte.gz',
    'test_data'   : 't10k-images-idx3-ubyte.gz'
}

def fetch_fashion_mnist(data_target = True):
    train_dict = {}
    for file_key, file_value in train_files.items():
        train_dict.update({file_key : maybe_download('../../ztlearn/datasets/fashion/', URL + file_value)})

    with gzip.open(list(train_dict.values())[0], 'rb') as label_path:
        train_label = np.frombuffer(label_path.read(), dtype = np.uint8, offset = 8)

    with gzip.open(list(train_dict.values())[1], 'rb') as data_path:
        train_data = np.frombuffer(data_path.read(), dtype = np.uint8, offset = 16).reshape(len(train_label), 784)

    test_dict = {}
    for file_key, file_value in test_files.items():
        test_dict.update({file_key : maybe_download('../../ztlearn/datasets/fashion/', URL + file_value)})

    with gzip.open(list(test_dict.values())[0], 'rb') as label_path:
        test_label = np.frombuffer(label_path.read(), dtype = np.uint8, offset = 8)

    with gzip.open(list(test_dict.values())[1], 'rb') as data_path:
        test_data = np.frombuffer(data_path.read(), dtype = np.uint8, offset = 16).reshape(len(test_label), 784)

    if data_target:
        return DataSet(np.concatenate((train_data,  test_data),  axis=0),
                       np.concatenate((train_label, test_label), axis=0))
    else:
        return train_data, test_data, train_label, test_label
