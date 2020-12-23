import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def to_ndarray(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        if data.shape[1] == 1:
            return to_array(data)
        return data.values
    if isinstance(data, list):
        return np.array(data)
    return data


def to_array(x, one_dimensional=True) -> np.array:
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) > 1:
        assert x.shape[1] == 1, "Array must be 1-dimensional"
    if one_dimensional:
        x = x.reshape(x.shape[0])
    else:
        x = x.reshape(-1, 1)
    return x

import scipy.sparse as sp
import pandas as pd 
import numpy as np
import inspect
import argus


PANDAS = 'PD'
NUMPY = 'NP'
SPARSE_CSR = 'SP'
DATA_TYPES = [PANDAS, NUMPY, SPARSE_CSR]
DATA_MAP = {
    PANDAS: pd.DataFrame,
    NUMPY: np.array,
    SPARSE_CSR: sp.csr_matrix
}

CONCAT_FUNCTIONS = {
    k
}

class ArgusDataset(tuple):
    """
    To add a new type of dataset:
    1- Decalre class static variable, update DATA_TYPES and DATA_MAP
    2- Update .__getitem__
    3- Update utils.py with concatenation method, must have same prefix as data_type declared in this module.
    4- Update .get_data to retrieve merged dataset
    """
    

    def __init__(self, dataset_dict=None,
        df: pd.DataFrame=None,
        X: np.array=None,
        X_sp=None, 
        use_as_index=PANDAS, 
        verbose=False):
        # TODO : format verification
        if dataset_dict is None:
            dataset_dict = {}
        if not df is None:
            dataset_dict[PANDAS] = df
        if not X is None:
            dataset_dict[NUMPY] = X
        if not X_sp is None:
            dataset_dict[SPARSE_CSR] = X_sp        
        assert isinstance(dataset_dict, dict), "dataset_dict must be a dictionarys"
        
        self.index_type = use_as_index
        self.dataset_dict = dataset_dict
        self.verbose = verbose

    def _init_shape(self):
        pass

    def is_empty(self):
        if self.dataset_dict[self.index_type]:
            return self.dataset_dict[self.index_type].shape[0] == 0
        return True

    def __getitem__(self, dataset_id):
        return self.dataset_dict.get(dataset_id)

    def get_dataset(self, dataset_id):
        return self.dataset_dict[dataset_id]

    def get_data(self, axis_dict=None):
        # return a merged version of the data
        # TO OVERRIDE
        pass


   
def get_datasets_by_type(dataset_list: list, data_type):
    dataset_list = [dataset[data_type] for dataset in dataset_list if not dataset[data_type] is None]
    if len(dataset_list) == 0:
        return None
    return dataset_list
    
def concat(dataset_list: list) -> ArgusDataset:
    concatenated_dataset = {}
    for data_type in DATA_TYPES:
        assert np.unique([dataset[data_type].shape[1] for dataset in dataset_list]), "Concat mismatch: All {data_type}-type data must have the same number of columns."
        concat_dataset = get_datasets_by_type(dataset_list, data_type)
        if concat_dataset:
            concatenated_dataset[data_type] = concat_dataset
    return ArgusDataset(dataset_dict=concatenated_dataset)
    # Concatenate list checking if diff from None