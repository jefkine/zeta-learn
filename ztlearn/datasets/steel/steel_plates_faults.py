import os
import pandas as pd

from ztlearn.utils import maybe_download
from ztlearn.utils import train_test_split
from ztlearn.datasets.data_set import DataSet

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA'
URL_2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults27x7_var'

def fetch_steel_plates_faults(data_target = True, custom_path = os.getcwd()):
    file_path = maybe_download(custom_path + '/../../ztlearn/datasets/steel/', URL)
    file_path_2 = maybe_download(custom_path + '/../../ztlearn/datasets/steel/', URL_2)
    describe  = [
        'Pastry',
        'Z_Scratch',
        'K_Scatch',
        'Stains',
        'Dirtiness',
        'Bumps',
        'Other_Faults'
    ]
        
    InputDataHeader = pd.read_csv(file_path_2, header=None)  
    InputData = pd.read_csv(file_path, header=None, sep="\t")
    InputData.set_axis(InputDataHeader.values.flatten(), axis=1, inplace=True)
    
    dataframe = InputData.copy()
    dataframe.drop(describe, axis=1,inplace=True)
    targetframe = InputData[describe].copy()
    
    data, target = dataframe.values, targetframe.values
    
    if data_target:
        return DataSet(data, target, describe)
    else:
        return train_test_split(data, target, test_size = 0.2, random_seed = 2)
