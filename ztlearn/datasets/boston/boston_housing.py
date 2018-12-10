import pandas as pd

from ..data_set import DataSet
from ztlearn.utils import maybe_download
from ztlearn.utils import train_test_split

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'

def fetch_boston(in_class = True):
    file_path = maybe_download('../../ztlearn/datasets/boston/', URL)
    describe  = [
        'CRIM',
        'ZN',
        'INDUS',
        'CHAS',
        'NOX',
        'RM',
        'AGE',
        'DIS',
        'RAD',
        'TAX',
        'PTRATIO',
        'B',
        'LSTAT',
        'MEDV'
    ]

    dataframe     = pd.read_csv(file_path, delim_whitespace = True, names = describe)
    data, targets = dataframe.values[:,0:13], dataframe.values[:,13]

    if in_class:
        return DataSet(data, targets, describe)
    else:
        return train_test_split(data, targets, test_size = 0.2, random_seed = 2)
