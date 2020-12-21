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

def split_datasets()

#############################################
#############################################
# 4. Shared features pipeline

# Initialize array and sparse features
dataset_dict = {}

for dataset_name in df_dict.keys():
    dataset_tuple = (d
        f_dict[dataset_name], 
        np.empty(shape=(df_dict[dataset_name].shape[0])),
        csr_matrix()
    )
    dataset_dict[dataset_name] 

#############################################
#############################################
# 5. Configure experiment setup

# Convert search space to ray.tune configuration

#############################################
#############################################
# 6. Run experiments
