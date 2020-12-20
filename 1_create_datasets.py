#%%
import pandas as pd 
import numpy as np
import yaml
from yaml import Loader

import argus
from argus import *


PREPROCESSINGS_FUNCTIONS = {}

MANDATORY_KEYS = ['seed', 'source_filepath']
"""
Author: Lucas Bony

This recipe is used to create datasets for experiments.
"""
#############################################
#############################################
# 1. Parameters

PARAMETERS = '''
source_filepath: ./datasets/jane_street/train_100_000.csv

train_dataset_filepath: xx
valid_dataset_filepath: xx
test_dataset_filepath: xx

seed: 1234

preprocessings:
    -
        sample:
            subsample_ratio: 0.3
    -
        sample:
            subsample_ratio: 0.5
    -
        eliminate_stopwords:
            source_colname: text
            destination_colname: text

    -
        encode_date:
            
    
data_split:
    type: date_split
    date_colname: date
    train:
        start: 0
        end: 10
    valid:
        start: 11
        end: 15
    test:
        start: 16
        end: 19

'''

NB_WORKERS = 1

data_config = yaml.load(PARAMETERS, Loader=Loader)

#############################################
#############################################
# 2. Parameters Consistency
assert 'data_split' in data_config, "data_split key must be provided"
assert 'type' in data_config['data_split'], "type of data_split must be provided"
for key in MANDATORY_KEYS:
    assert key in data_config, f"{key} must be provided in configuration."

#############################################
#############################################
# 3. Loading data

df = pd.read_csv(data_config['source_filepath'])
df['text'] = 'Bonjour je suis je le moi je'

#%%
#############################################
#############################################
# 4. Apply preprocessing and filtering


def get_functions(config: dict, functions_by_name: dict, file_folder=None, seed=1234) -> list:
    """
    :config: A dictionary, keys are functions names, values are either true boolean
    either a dictionary of parameters. Parameters must be consistent with the corresponding
    function declared in function_by_name dictinary. In order to fit an object beforehand,
    declare a function with the prefix "fit_" (or any prefix declared in argus.FITTING_PREXIF)
    that will be called on the whole dataset
    :function_by_name: A dictionary, keys are function names, values the actual function objects
    :file_folder: A string, path to the folder containing the files required to apply functions
    Return:
        A list of functions with corresponding parameters
    """
    functions_list = []

    for func_dict in config:
        function_name = list(func_dict.keys())[0]
        params = func_dict[function_name]
        params['seed'] = seed
        assert function_name in functions_by_name, f"{function_name} function doesnt exist."
        functions_list.append((functions_by_name[function_name], format_parameters(params, file_folder)))
    return functions_list

def format_parameters(params: dict, file_folder: str) -> dict:
    for key, param in params.items():
        if isinstance(key, str):
            if '.' in key:
                # it has a file extension
                data = None
                try:
                    with open(osp.join(file_folder, key), 'r') as f:
                        data = f.read()
                        f.close()
                except:
                    with open(osp.join(file_folder, key), 'rb') as f:
                        data = pickle.load(f)
                        f.close()
                params[key] = data
    return params
                

preprocessing_functions_list = get_functions(data_config['preprocessings'], argus.PREPROCESSING_FUNCTIONS, seed=data_config['seed'])



#%%
#############################################
#############################################
# 5. Splitting data

def split_data(df: pd.DataFrame, data_config: dict) -> list:
    """
    Returns splitted data
    """
    if data_config['data_split']['type'] == 'date_split':
        train_df = df.loc[
            (df[data_config['data_split']['date_colname']] >= data_config['data_split']['train']['start']) & (df[data_config['data_split']['date_colname']] <= data_config['data_split']['train']['end'])
            ]
        valid_df = df.loc[
            (df[data_config['data_split']['date_colname']] >= data_config['data_split']['valid']['start']) & (df[data_config['data_split']['date_colname']] <= data_config['data_split']['valid']['end'])
            ]
        test_df = df.loc[
            (df[data_config['data_split']['date_colname']] >= data_config['data_split']['test']['start']) & (df[data_config['data_split']['date_colname']] <= data_config['data_split']['test']['end'])
            ]

    return train_df, valid_df, test_df

train_df, valid_df, test_df = split_data(df, data_config)

#%%
#############################################
#############################################
# 6. Applying preprocessing and filtering

import argus
from argus.design import get_preprocessing_function_level
import pandas as pd 
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import os
import gc

from pandarallel import pandarallel

pandarallel.initialize(nb_workers=NB_WORKERS)

class ArgusPipeline:
    def __init__(self, functions_list: list, functions_by_name: dict, nb_workers=1, verbose=False):
        self.functions_list = functions_list
        self.functions_by_name = functions_by_name 
        self.nb_workers = nb_workers
        self.verbose = verbose

    def apply(self, df_dict: dict) -> dict:
        """
        Dataset-level transformations are applied on-the-fly
        Row-level transformations are applied in lazy mode: i.e. 
        functions are "pipelined" (and not applied) until a non-row-level
        function pops
        """
        null_dfs_dict = {}
        for key in df_dict:
            if df_dict[key] is None:
                null_dfs_dict[key] = df_dict.pop(key)
            elif isinstance(df_dict[key], pd.DataFrame):
                if df_dict[key].shape[0] == 0:
                    null_dfs_dict[key] = df_dict.pop(key)

        row_level_functions_buffer = list()

        for function, params in self.functions_list:
            
            # Detect if fitting function is required
            fitting_functions_list = [param for param in params.keys() if param.find(argus.FITTING_PREXIF) > 0]
            assert len(fitting_functions_list) in [0, 1], "At most 1 fitting function must be provided"
                          
            # Fit object on concatenated data
            if len(fitting_functions_list) > 0:
                if self.verbose:
                    print(f"Fitting function detected: {fitting_functions_list[0]}")
                df = pd.concat([list(df_dict.values())])
                params = fitting_functions_list[0](df, params)
                del df
                gc.collect()

            # If function is row-level, append to buffer
            # and do not apply
            if get_preprocessing_function_level(function.__name__) == argus.ROW_LEVEL:
                row_level_functions_buffer.append((function, params))
            else:
                df_dict, row_level_functions_buffer = self._apply_row_level_functions(df_dict, row_level_functions_buffer)

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
    





# %%
##############################################################
##############################################################
# 7. Apply preprocessing and filters

pipeline = ArgusPipeline(functions_list=preprocessing_functions_list, functions_by_name=PREPROCESSING_FUNCTIONS, verbose=True)
dfs_dict = pipeline.apply({
    'train': train_df,
    'valid': valid_df,
    'test': test_df
})


# %%
