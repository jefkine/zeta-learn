import pandas as pd

from ..data_set import DataSet
from ztlearn.utils import maybe_download
from ztlearn.utils import train_test_split

URL = 'http://ftp.ics.uci.edu/pub/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'

def fetch_pima_indians(data_target = True):
    file_path = maybe_download('../../ztlearn/datasets/pima/', URL)
    describe  = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'DiabetesPedigreeFunction',
        'Age',
        'Insulin',
        'BMI',
        'Outcome (0 or 1)'
    ]

    dataframe     = pd.read_csv(file_path, names = describe)
    data, target = dataframe.values[:,0:8], dataframe.values[:,8]

    if data_target:
        return DataSet(data, target, describe)
    else:
        return train_test_split(data, targets, test_size = 0.2, random_seed = 2)
