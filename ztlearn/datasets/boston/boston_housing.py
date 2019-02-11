import os
import pandas as pd

from ztlearn.utils import maybe_download
from ztlearn.utils import train_test_split
from ztlearn.datasets.data_set import DataSet

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'

def fetch_boston(data_target = True, custom_path = os.getcwd()):
    file_path = maybe_download(custom_path + '/../../ztlearn/datasets/boston/', URL)
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

    dataframe    = pd.read_csv(file_path, delim_whitespace = True, names = describe)
    data, target = dataframe.values[:,0:13], dataframe.values[:,13]

    if data_target:
        return DataSet(data, target, describe)
    else:
        return train_test_split(data, target, test_size = 0.2, random_seed = 2)
