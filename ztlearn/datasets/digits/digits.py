import gzip
import numpy as np

from ..data_set import DataSet
from ztlearn.utils import maybe_download
from ztlearn.utils import train_test_split

URL = 'https://github.com/scikit-learn/scikit-learn/raw/master/sklearn/datasets/data/digits.csv.gz'

def fetch_digits(in_class = True):
    file_path   = maybe_download('../../ztlearn/datasets/digits/', URL)

    with gzip.open(file_path, 'rb') as digits_path:
        digits_data = np.loadtxt(digits_path, delimiter=',')

    data, targets = digits_data[:, :-1], digits_data[:, -1].astype(np.int)

    if in_class:
        return DataSet(data, targets)
    else:
        return train_test_split(data, targets, test_size = 0.33, random_seed = 5)
