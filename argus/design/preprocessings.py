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
    "sample"
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

    