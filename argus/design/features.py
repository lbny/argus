from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 

import argus
from argus.utils.data import ArgusDataset, concat

#######################################
#######################################
#######################################

DATASET_LEVEL_FUNCTION_NAMES = [
    "fit_tf_idf",
    "tf_idf",
    "add_sum_col"
]

ROW_LEVEL_FUNCTION_NAMES = []

def fit_tf_idf(dataset: ArgusDataset, params: dict) -> dict:
    tfidf = TfidfVectorizer(
        sublinear_tf=params.get('sublinear_tf') or False,
        min_df=params.get('min_df') or 1,
        max_df=params.get('max_df') or 1.0
    ) 
    tfidf.fit(dataset[argus.PANDAS][params['text_colname']])
    params['tfidf'] = tfidf
    return params

def tf_idf(dataset: ArgusDataset, params: dict) -> ArgusDataset:
    tfidf_matrix = params['tfidf'].transform(dataset[argus.PANDAS][params['text_colname']])
    dataset = concat([dataset, ArgusDataset(X_sp=tfidf_matrix)], axis=1)
    return dataset

def add_sum_col(dataset: ArgusDataset, params: dict) -> ArgusDataset:
    dataset[argus.PANDAS][params['destination_colname']] = dataset[argus.PANDAS][params['colnames']].sum(axis=1)
    return dataset
#######################################
#######################################
#######################################