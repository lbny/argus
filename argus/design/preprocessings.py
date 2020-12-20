import pandas as pd 
import numpy as np

"""
Author: Lucas Bony

This modules contains the functions used in preprocessing.
Each parameter in the configuration must correspond to function
in this module.

Instructions to add a new function:
1- Signature must respect:
(df: pd.DataFrame, args: dict) -> pd.DataFrame
"""

DATASET_LEVEL_FUNCTION_NAMES = [
    "sample",
    "encode_date"
]

ROW_LEVEL_FUNCTION_NAMES = [
    "eliminate_stopwords"
]

def sample(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    if args['subsample_ratio'] > 1:
        ratio = args['subsample_ratio']
    else:
        ratio = int(args['subsample_ratio'] * df.shape[0])
    
    df = df.sample(n=int(ratio), random_state=args['seed'])
    return df


def eliminate_stopwords(row: pd.Series, args: dict) -> pd.Series:
    s = row[args['source_colname']]
    s = s.lower().replace('je', '')
    row[args['destination_colname']] = s
    return row

def encode_date(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    ohe = args['ohe']
    colnames = [f"date_{i}" for i in ohe.categories_[0]]
    df[colnames] = ohe.transform(df[args['date_colname']].values.reshape(-1, 1))
    return df

from sklearn.preprocessing import OneHotEncoder
def fit_encode_date(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(df[args['date_colname']].values.reshape(-1, 1))
    args['ohe'] = ohe
    return args
    