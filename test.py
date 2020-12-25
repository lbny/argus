#%%
# Data structures
import pandas as pd 
import numpy as np
import scipy.sparse as sp

from argus.utils.data import ArgusDataset

import os
import os.path as osp
# %%

##PARAMETERS
DATA_DIR = './datasets/'

# %%
# Load in dataframe
df_dict = {
    k: pd.read_csv(osp.join(DATA_DIR, 'jane_street','train_100_000.csv'), nrows=10_000) for k in ['train', 'valid', 'test']
}


# %%
X_dict = {
    k: np.empty(shape=(df_dict[k].shape[0], 0)) for k in df_dict.keys()
}

X_sp_dict = {
    k: sp.csr_matrix(np.empty(shape=(df_dict[k].shape[0], 0))) for k in df_dict.keys()
}
# %%
dataset_dict = {
    k: ArgusDataset(df=df_dict[k], X=X_dict[k], X_sp=X_sp_dict[k]) for k in df_dict.keys()
}
# %%
from argus.utils.data import concat

d = concat(dataset_dict)
# %%
d = concat(list(dataset_dict.values()), axis=1)

# %%

# %%
