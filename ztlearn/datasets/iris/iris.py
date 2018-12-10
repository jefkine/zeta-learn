import pandas as pd

from ..data_set import DataSet
from ztlearn.utils import maybe_download
from ztlearn.utils import train_test_split

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

def fetch_iris(data_target = True):
    file_path = maybe_download('../../ztlearn/datasets/iris/', URL)
    describe  = [
        'sepal-length (cm)',
        'sepal-width (cm)',
        'petal-length (cm)',
        'petal-width (cm)',
        'petal_type'
    ]

    dataframe = pd.read_csv(file_path, names = describe)

    # convert petal type column to categorical data i.e {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}
    dataframe.petal_type    = pd.Categorical(dataframe.petal_type)
    dataframe['petal_type'] = dataframe.petal_type.cat.codes

    data, targets = dataframe.values[:,0:4], dataframe.values[:,4].astype('int')

    if data_target:
        return DataSet(data, targets, describe)
    else:
        return train_test_split(data, targets, test_size = 0.2, random_seed = 2)
