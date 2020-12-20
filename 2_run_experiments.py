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
seed: 


'''

NB_WORKERS = 1

# Initialization
pandarallel.initialize(nb_workers=NB_WORKERS)
data_config = yaml.load(PARAMETERS, Loader=Loader)

#############################################
#############################################
# 2. Parameters Consistency

#############################################
#############################################
# 3. Loading datasets

for dataset_name, dataset_filepath in config['source_filpaths'].items():


#############################################
#############################################
# 3. Dataset splitting

def split_datasets()

#############################################
#############################################
# 4. Shared features pipeline

# Initialize array and sparse features

#############################################
#############################################
# 5. Configure experiment setup

# Convert search space to ray.tune configuration

#############################################
#############################################
# 6. Run experiments
