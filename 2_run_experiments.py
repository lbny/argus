#%%
import pandas as pd 
import numpy as np
import yaml
from yaml import Loader

import argus
from argus import *

from pandarallel import pandarallel

"""
Author: Lucas Bony

This recipe is used to run experiments.
Two feature pipelines are run separately:
- shared features : run once
- experiment features : run in each experiment
Basically, features which are parametrized by variables
which you intend to search by hyperoptimization should be incroporated
in the experiment features.

Data structure for the datasets follows : 
- dictionary of pd.DataFrame (meaning each dataset must be named)
- prior to shared feature pipeline, turn into dictionary of tuples :
    pd.DataFrame, np.array, scipy.sparse.csr_matrix
"""
#############################################
#############################################
# 1. Parameters

PARAMETERS = '''
source_filepaths: 
    # datasets should be named
    train_dataset_filepath: ./datasets/jane_street/preprocessed/train.csv
    valid_dataset_filepath: ./datasets/jane_street/preprocessed/valid.csv
    test_dataset_filepath: ./datasets/jane_street/preprocessed/test.csv
    features_filepath: ./datasets/jane_street/features.csv

seed: 1234

shared_features:
    tf_idf:
        text_colname: text
        sublinear_tf: true
        min_df: 10

experiment_features:
    tf_idf:
        text_colname: 
            type: fixed
            val: text
        min_df:
            type: randint
            val: [5, 25]
        


'''

NB_WORKERS = 1

# Initialization
pandarallel.initialize(nb_workers=NB_WORKERS)
config = yaml.load(PARAMETERS, Loader=Loader)

#############################################
#############################################
# 2. Parameters Consistency

# %%
#############################################
#############################################
# 3. Loading datasets
df_dict = {}

for dataset_name, dataset_filepath in config['source_filpaths'].items():
    df_dict[dataset_name] = pd.read_csv(dataset_filepath)

#############################################
#############################################
# 3. Dataset splitting

def split_datasets():
    pass

#############################################
#############################################
# 4. Shared features pipeline
# to stack horizontally sparse matrices : sp.hstack([a,b])
# Initialize array and sparse features
dataset_dict = {}

for dataset_name in dataset_dict.keys():
    dataset_tuple = (
        dataset_dict[dataset_name], 
        np.empty(shape=(df_dict[dataset_name].shape[0])),
        csr_matrix(shape=(df_dict[dataset_name].shape[0]))
    )
    dataset_dict[dataset_name] = dataset_tuple

# %%
import scipy.sprase as sp

class ArgusDataset(tuple):
    """
    To add a new type of dataset:
    1- Decalre class static variable, update DATA_TYPES and DATA_MAP
    2- Update .__getitem__
    3- Update .concat with concatenation method
    4- Update .get_data to retrieve merged dataset
    """
    PANDAS = 'PD'
    NUMPY = 'NP'
    SPARSE_CSR = 'SP'
    DATA_TYPES = [PANDAS, NUMPY, SPARSE_CSR]
    DATA_MAP = {
        PANDAS: pd.DataFrame,
        NUMPY: np.array,
        SPARSE_CSR: sp.csr_matrix
    }

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
        self.dataset_dict = dataset_dict
        self.verbose = verbose

    def is_empty(self):
        if self.dataset_dict[PANDAS]:
            return self.dataset_dict[PANDAS].shape[0] == 0
        return True

    def __getitem__(self, dataset_id):
        return self.dataset_dict.get(dataset_id)

    def get_dataset(self, dataset_id)
        return self.dataset_dict[dataset_id]

    def get_data(self):
        # return a merged version of the data

    
def get_datasets_by_type(dataset_list: list, data_type):
    return [dataset[data_type] for dataset in dataset_list if not dataset[PANDAS] is None]

    
def concat(dataset_list: list) -> ArgusDataset:
    for data_type in DATA_TYPES:
        assert np.unique([dataset[data_type].shape[1] for dataset in dataset_list]), "Concat mismatch: All {data_type}-type data must have the same number of columns."

    return ArgusDataset(dataset_dict={
        ArgusDataset.PANDAS: pd.concat(get_datasets_by_type(dataset_list, PANDAS)),
        ArgusDataset.NUMPY: np.concatenate(get_datasets_by_type(dataset_list, NUMPY), axis=0),
        ArgusDataset.SPARSE_CSR: sp.vstack(get_datasets_by_type(dataset_list, SPARSE_CSR))
    })
    # Concatenate list checking if diff from None


class ArgusFeaturesPipeline:
    def __init__(self, functions_list: list, functions_by_name: dict, nb_workers=1, verbose=False):
        self.functions_list = functions_list
        self.functions_by_name = functions_by_name 
        self.nb_workers = nb_workers
        self.verbose = verbose

    def apply(self, dataset_dict: dict) -> dict:
        """
        Dataset-level transformations are applied on-the-fly
        Row-level transformations are applied in lazy mode: i.e. 
        functions are "pipelined" (and not applied) until a non-row-level
        function pops up.
        """
        null_dataset_dict = {}
        for key in dataset_dict:
            if dataset_dict[key].is_null()
                null_dfs_dict[key] = dataset_dict.pop(key)


        row_level_functions_buffer = list()

        for function, params in self.functions_list:
            
            # Detect if fitting function is required
            fitting_functions_list = [param for param in params.keys() if param.find(argus.FITTING_PREXIF) >= 0]
            assert len(fitting_functions_list) in [0, 1], "At most 1 fitting function must be provided"
                          
            # Fit object on concatenated data
            if len(fitting_functions_list) > 0:
                if self.verbose:
                    print(f"Fitting function detected: {fitting_functions_list[0]}")
                dataset = concat([dataset_dict.values()])
                params = self.functions_by_name[fitting_functions_list[0]](dataset, params)
                del dataset
                gc.collect()

            # If function is row-level, append to buffer
            # and do not apply
            if get_preprocessing_function_level(function.__name__) == argus.ROW_LEVEL:
                row_level_functions_buffer.append((function, params))
            else:
                for 
                df_dict, row_level_functions_buffer = self._apply_row_level_functions(dataset_dict, row_level_functions_buffer)

                # Transform each dataset
                for key, _df in df_dict.items():
                    if self.verbose:
                        print(f'Applying {function} with parameters {params}...')
                    df_dict[key] = function(df_dict[key], params)
        if len(row_level_functions_buffer) > 0:
            df_dict, row_level_functions_buffer = self._apply_row_level_functions(df_dict, row_level_functions_buffer)
        
        df_dict.update(null_dfs_dict)
        return df_dict

    def _apply_row_level_functions(self, df_dict: dict, row_level_functions_buffer: list):
        # Check if buffer is not empty
        if len(row_level_functions_buffer) > 0:
            if self.verbose:
                print(f"Current buffer: {row_level_functions_buffer}")
            for key, _df in df_dict.items():
                df_dict[key] = self._apply_function(df_dict[key], row_level_functions_buffer)
            # Then flush
            row_level_functions_buffer = []
        return df_dict, row_level_functions_buffer

    def _apply_function(self, df: pd.DataFrame, row_level_functions_list: list) -> pd.DataFrame:
        # Check level of function : dataset or row
        def _f(row: pd.Series) -> pd.Series:
            for function, params in row_level_functions_list:
                row = function(row, params)
            return row

        if self.verbose:
            print('Applying preprocessing...')
            print(f'With parameters: {row_level_functions_list}')
        if self.nb_workers > 1:
            df = df.pandarallel_apply(_f, axis=1)
        else:
            df = df.progress_apply(_f, axis=1)
        return df


#############################################
#############################################
# 5. Configure experiment setup

# Convert search space to ray.tune configuration

#############################################
#############################################
# 6. Run experiments
