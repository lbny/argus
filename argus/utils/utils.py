import pandas as pd
import numpy as np
import scipy.sparse as sp

def pd_concat(df_list: list, axis=0) -> pd.DataFrame:
    return pd.concat(df_list, axis=axis)

def np_concat(np_list: list, axis=0) -> np.array:
    return np.concatenate(np_list, axis=axis)

def sp_concat(sp_list: list, axis=0):
    if axis == 0:
        return sp.vstack(sp_list)
    elif axis == 1:
        return sp.hstack(sp_list)