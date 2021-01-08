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
# List is ordered by prefered data type to be used
# for reference index
DATA_TYPES = [PANDAS, NUMPY, SPARSE_CSR]
DATA_MAP = {
    PANDAS: pd.DataFrame,
    NUMPY: np.array,
    SPARSE_CSR: sp.csr_matrix
}

def pd_get_null(shape=(0,0)):
    return pd.DataFrame(index=np.arange(shape[0]))

def np_get_null(shape=(0,0)):
    return np.empty(shape=shape)

def sp_get_null(shape=(0,0)):
    return sp.csr_matrix(np.empty(shape=shape))

NULL_VALUE_MAP = {
    PANDAS: pd_get_null,
    NUMPY: np_get_null,
    SPARSE_CSR: sp_get_null}



CONCAT_FUNCTIONS = {data_type: dict(inspect.getmembers(utils))[f"{data_type.lower()}_concat"] for data_type in DATA_TYPES}

class ArgusDataset(tuple):
    """
    To add a new type of dataset:
    1- Decalre module static variable, update DATA_TYPES and DATA_MAP
    2- Decakre <type_prefix>_get_null function
    3- DEcalre <type_prefix>_concat function
    4- Update DATA_TYPES list (ordered)
    5- Update DATA_MAP
    """
    

    def __init__(self, 
        dataset_dict=None,
        df: pd.DataFrame=None,
        X: np.array=None,
        X_sp=None, 
        use_as_index=None,
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
                dataset_dict[data_type] = None#NULL_VALUE_MAP(shape=(self.shape[0],))[data_type] 

        if use_as_index:
            self.index_type = use_as_index
        else:
            self.index_type = DATA_TYPES[sorted([DATA_TYPES.index(x) for x in dataset_dict.keys() if not dataset_dict.get(x) is None])[0]]
        self.dataset_dict = dataset_dict
        self.verbose = verbose

        self._init_shapes()

        

    def _init_shapes(self):
        self.shape = (self.dataset_dict[self.index_type].shape[0], self.dataset_dict[self.index_type].shape[1]) 
        self.shape_dict = {k: data.shape for k, data in self.dataset_dict.items() if not data is None}
        self.shape_dict.update({k: None for k, data in self.dataset_dict.items() if data is None })

    def is_null(self):
        if not self.dataset_dict[self.index_type] is None:
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
    if len(datasets) == 0:
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
            concat_dataset = get_datasets_by_type(datasets, data_type)
            if len(concat_dataset) > 1:
                n_axis_per_dataset = np.unique([len(data.shape) for data in concat_dataset])
                assert n_axis_per_dataset.shape[0] == 1, f"Concat mismatch: Different number of axis for {data_type}-type data"
                n_axis = n_axis_per_dataset[0]

                for _axis in np.arange(n_axis):
                    if _axis != axis:
                        assert np.unique([data.shape[_axis] for data in concat_dataset]).shape[0] == 1, f"Concat mismatch: All {data_type}-type data must have the dimension on axis {_axis}"
                                
                if concat_dataset:
                    concatenated_dataset[data_type] = CONCAT_FUNCTIONS[data_type](concat_dataset, axis=axis)
                    
            elif len(concat_dataset) == 1:
                concatenated_dataset[data_type] = concat_dataset[0]
            else:
                concatenated_dataset[data_type] = None
                    
        return ArgusDataset(dataset_dict=concatenated_dataset)
    return ArgusDataset()
    # Concatenate list checking if diff from None