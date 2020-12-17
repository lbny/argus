import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def to_ndarray(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        if data.shape[1] == 1:
            return to_array(data)
        return data.values
    if isinstance(data, list):
        return np.array(data)
    return data


def to_array(x, one_dimensional=True) -> np.array:
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) > 1:
        assert x.shape[1] == 1, "Array must be 1-dimensional"
    if one_dimensional:
        x = x.reshape(x.shape[0])
    else:
        x = x.reshape(-1, 1)
    return x