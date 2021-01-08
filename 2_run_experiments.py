#%%
import pandas as pd 
import numpy as np
import yaml
from yaml import Loader

import argus
from argus import *

import gc
import os
import os.path as osp

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
    #features_filepath: ./datasets/jane_street/features.csv

seed: 1234

file_folder: null

shared_features:
    -
        tf_idf:
            fit_tf_idf: null
            text_colname: text
            sublinear_tf: true
            min_df: 10

    -
        add_sum_col:
            colnames: ['feature_1', 'feature_2']
            destination_colname: sum_col

experiment_features:
    -
        tf_idf:
            text_colname: 
                type: fixed
                val: text
            min_df:
                type: randint
                val: [5, 25]
            
training_functions:
    -
        train_lr:
            C:
                type: fixed
                val: 6.
    -
        train_sgd:
            D:
                type: fixed
                val: 8.

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

for dataset_name, dataset_filepath in config['source_filepaths'].items():
    df_dict[dataset_name] = pd.read_csv(dataset_filepath, nrows=10_000)
    df_dict[dataset_name]['text'] = 'BOnjour à tous !'

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

for dataset_name in df_dict.keys():
    dataset_dict[dataset_name] = ArgusDataset(df=df_dict[dataset_name])


# %%

import argus
from argus.utils.data import concat
from argus.design import get_feature_function_level
from argus.design.utils import get_functions

class ArgusFeaturesPipeline:
    def __init__(self, functions_list: list, functions_by_name: dict, nb_workers=1, verbose=False):
        
        self.functions_list = functions_list
        self.functions_by_name = functions_by_name 
        self.nb_workers = nb_workers
        self.verbose = verbose

    def apply(self, dataset_dict: dict) -> dict:
        """[Dataset-level transformations are applied on-the-fly
        Row-level transformations are applied in lazy mode: i.e. 
        functions are "pipelined" (and not applied) until a non-row-level
        function pops up.]

        Args:
            dataset_dict (dict): [Dictionary of ArgusDataset objects]

        Returns:
            dict: [description]
        """
        null_datasets_dict = {}
        for key in dataset_dict:
            if dataset_dict[key].is_null():
                null_datasets_dict[key] = dataset_dict.pop(key)


        row_level_functions_buffer = list()

        for function, params in self.functions_list:
            
            # Detect if fitting function is required
            fitting_functions_list = [param for param in params.keys() if param.find(argus.FITTING_PREXIF) >= 0]
            assert len(fitting_functions_list) in [0, 1], "At most 1 fitting function must be provided"
                          
            # Fit object on concatenated data
            if len(fitting_functions_list) > 0:
                if self.verbose:
                    print(f"Fitting function detected: {fitting_functions_list[0]}")
                dataset = concat(dataset_dict)
                params = self.functions_by_name[fitting_functions_list[0]](dataset, params)
                del dataset
                gc.collect()

            # If function is row-level, append to buffer
            # and do not apply
            if get_feature_function_level(function.__name__) == argus.ROW_LEVEL:
                row_level_functions_buffer.append((function, params))
            else:
                # Apply all row-level transformations from buffer (then flush)
                dataset_dict, row_level_functions_buffer = self._apply_row_level_functions(dataset_dict, row_level_functions_buffer)

                # Transform each dataset
                for key, dataset in dataset_dict.items():
                    if self.verbose:
                        print(f'Applying {function} with parameters {params}...')
                    dataset_dict[key] = function(dataset, params)
        
        # After end of iteration, applu row-level transformations from buffer (if any)
        if len(row_level_functions_buffer) > 0:
            dataset_dict, row_level_functions_buffer = self._apply_row_level_functions(dataset_dict, row_level_functions_buffer)
        
        dataset_dict.update(null_datasets_dict)
        return dataset_dict

    def _apply_row_level_functions(self, dataset_dict: dict, row_level_functions_buffer: list):
        # Check if buffer is not empty
        if len(row_level_functions_buffer) > 0:
            if self.verbose:
                print(f"Current buffer: {row_level_functions_buffer}")
            for key, _dataset in dataset_dict.items():
                dataset_dict[key][argus.PANDAS] = self._apply_function(dataset_dict[key][argus.PANDAS], row_level_functions_buffer)
            # Then flush
            row_level_functions_buffer = []
        return dataset_dict, row_level_functions_buffer

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
# 6. Run experiments

# %%
features_function_list = get_functions(config['shared_features'], argus.FEATURES_FUNCTIONS, file_folder=config['file_folder'])

# %%



shared_features_pipeline = ArgusFeaturesPipeline(features_function_list, functions_by_name=argus.FEATURES_FUNCTIONS, verbose=True)
dataset_dict = shared_features_pipeline.apply(dataset_dict)
# %%
from ray import tune




# %%
#############################################
#############################################
# 5. Configure experiment setup

# Convert search space to ray.tune configuration

tune_config = argus.run.to_tune_format(config, ['experiment_features', 'training_functions'])

    # %%
# Run ray experiments
# %%

def trial(config: dict):
    pass


