#%%
import pandas as pd 
import numpy as np
import yaml
from yaml import Loader

import argus
from argus import *

from pandarallel import pandarallel

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

train_dataset_filepath: ./datasets/jane_street/preprocessed/train.csv
valid_dataset_filepath: ./datasets/jane_street/preprocessed/valid.csv
test_dataset_filepath: ./datasets/jane_street/preprocessed/test.csv

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
            date_colname: date
            fit_encode_date: {}
    
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

# Initialization
pandarallel.initialize(nb_workers=NB_WORKERS)
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


from argus.design.utils import get_functions
                

preprocessing_functions_list = get_functions(data_config['preprocessings'], argus.PREPROCESSING_FUNCTIONS, seed=data_config['seed'])



#%%
#############################################
#############################################
# 5. Splitting data

from argus.design.data import split_data

train_df, valid_df, test_df = split_data(df, data_config)

#%%
#############################################
#############################################
# 6. Applying preprocessing and filtering

from argus.run.pipelines import ArgusPreprocessingPipeline



pipeline = ArgusPreprocessingPipeline(functions_list=preprocessing_functions_list, functions_by_name=PREPROCESSING_FUNCTIONS, verbose=True)
dfs_dict = pipeline.apply({
    'train': train_df,
    'valid': valid_df,
    'test': test_df
})

# %%
#############################################
#############################################
# 7. Writing outputs

train_df.to_csv(data_config['train_dataset_filepath'])
valid_df.to_csv(data_config['valid_dataset_filepath'])
test_df.to_csv(data_config['test_dataset_filepath'])

print('Test OK')
# %%
