import pandas as pd

from ztlearn.utils import maybe_download

class Data_set:

    def __init__(self, data, target, describe):
        self.data     = data
        self.target   = target
        self.describe = describe

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

def fetch_iris():
    file_path   = maybe_download('../../datasets/iris/', URL)
    describe    = ['sepal-length (cm)', 'sepal-width (cm)', 'petal-length (cm)', 'petal-width (cm)', 'petal_type']
    dataframe   = pd.read_csv(file_path, names = describe)

    # convert petal type column to categorical data i.e {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}
    dataframe.petal_type    = pd.Categorical(dataframe.petal_type)
    dataframe['petal_type'] = dataframe.petal_type.cat.codes

    iris_dataset = Data_set(dataframe.values[:,0:4], dataframe.values[:,4].astype('int'), describe)

    return iris_dataset