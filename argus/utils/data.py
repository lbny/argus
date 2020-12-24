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
import argus.utils.utils as utils


PANDAS = 'PD'
NUMPY = 'NP'
SPARSE_CSR = 'SP'
DATA_TYPES = [PANDAS, NUMPY, SPARSE_CSR]
DATA_MAP = {
    PANDAS: pd.DataFrame,
    NUMPY: np.array,
    SPARSE_CSR: sp.csr_matrix
}
NULL_VALUE_MAP = {
    PANDAS: pd.DataFrame(),
    NUMPY: np.empty(shape=(0,0)),
    SPARSE_CSR: sp.csr_matrix(np.empty(shape=(0,0)))
}

CONCAT_FUNCTIONS = {data_type: dict(inspect.getmembers(utils))[f"{data_type.lower()}_concat"] for data_type in DATA_TYPES}

class ArgusDataset(tuple):
    """
    To add a new type of dataset:
    1- Decalre class static variable, update DATA_TYPES and DATA_MAP
    2- Update .__getitem__
    3- Update utils.py with concatenation method, must have same prefix as data_type declared in this module.
    4- Update .get_data to retrieve merged dataset
    """
    

    def __init__(self, 
        dataset_dict=None,
        df: pd.DataFrame=None,
        X: np.array=None,
        X_sp=None, 
        use_as_index=PANDAS,
        verbose=False):
        """[summary]

        Args:
            dataset_dict ([type], optional): [description]. Defaults to None.
            df (pd.DataFrame, optional): [description]. Defaults to None.
            X (np.array, optional): [description]. Defaults to None.
            X_sp ([type], optional): [description]. Defaults to None.
            use_as_index ([type], optional): [description]. Defaults to PANDAS.
            verbose (bool, optional): [description]. Defaults to False.
        """
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
        
        for data_type in DATA_TYPES:
            if not data_type in dataset_dict:
                dataset_dict[data_type] = NULL_VALUE_MAP[data_type] 

        self.index_type = use_as_index
        self.dataset_dict = dataset_dict
        self.verbose = verbose

        self._init_shape()

    def _init_shape(self):
        self.shape = (self.dataset_dict[self.index_type].shape[0], self.dataset_dict[self.index_type].shape[1]) 

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

def to_dataset_list(datasets) -> list:
    """[Returns a list of dataset]

    Args:
        datasets ([list or dict]): [description]

    Returns:
        list: [description]
    """
    if isinstance(datasets, dict):
        datasets = list(datasets.values())
    assert isinstance(datasets, list), "Input must be a list or dictionary of ArgusDataset"
    if len(dataset_list) == 0:
        return None
    return datasets

def get_datasets_by_type(datasets, data_type):
    datasets = to_dataset_list(datasets)
    if datasets:
        datasets = [dataset[data_type] for dataset in datasets if not dataset[data_type] is None]

    return datasets
    
def concat(datasets, axis=0) -> ArgusDataset:
    datasets = to_dataset_list(datasets)
    if datasets:
        concatenated_dataset = {}
        for data_type in DATA_TYPES:
            assert np.unique([dataset[data_type].shape[1] for dataset in datasets]) == 1, "Concat mismatch: All {data_type}-type data must have the same number of columns."
            concat_dataset = get_datasets_by_type(datasets, data_type)
            if concat_dataset:
                concatenated_dataset[data_type] = CONCAT_FUNCTIONS[data_type](concat_dataset, axis=axis)
        return ArgusDataset(dataset_dict=concatenated_dataset)
    return ArgusDataset()
    # Concatenate list checking if diff from None