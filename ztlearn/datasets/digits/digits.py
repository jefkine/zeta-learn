import os
import gzip
import numpy as np

from ztlearn.utils import maybe_download
from ztlearn.utils import train_test_split
from ztlearn.datasets.data_set import DataSet

URL = 'https://github.com/scikit-learn/scikit-learn/raw/master/sklearn/datasets/data/digits.csv.gz'

def fetch_digits(data_target = True, custom_path = os.getcwd()):
    file_path = maybe_download(custom_path+'/../../ztlearn/datasets/digits/', URL)

    with gzip.open(file_path, 'rb') as digits_path:
        digits_data = np.loadtxt(digits_path, delimiter=',')

    data, target = digits_data[:, :-1], digits_data[:, -1].astype(np.int)

    if data_target:
        return DataSet(data, target)
    else:
        return train_test_split(data, target, test_size = 0.33, random_seed = 5)
