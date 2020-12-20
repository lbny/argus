import argus
from argus.design import get_preprocessing_function_level
import pandas as pd 
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import os
import gc

from pandarallel import pandarallel



class ArgusPreprocessingPipeline:
    def __init__(self, functions_list: list, functions_by_name: dict, nb_workers=1, verbose=False):
        self.functions_list = functions_list
        self.functions_by_name = functions_by_name 
        self.nb_workers = nb_workers
        self.verbose = verbose

    def apply(self, df_dict: dict) -> dict:
        """
        Dataset-level transformations are applied on-the-fly
        Row-level transformations are applied in lazy mode: i.e. 
        functions are "pipelined" (and not applied) until a non-row-level
        function pops
        """
        null_dfs_dict = {}
        for key in df_dict:
            if df_dict[key] is None:
                null_dfs_dict[key] = df_dict.pop(key)
            elif isinstance(df_dict[key], pd.DataFrame):
                if df_dict[key].shape[0] == 0:
                    null_dfs_dict[key] = df_dict.pop(key)

        row_level_functions_buffer = list()

        for function, params in self.functions_list:
            
            # Detect if fitting function is required
            fitting_functions_list = [param for param in params.keys() if param.find(argus.FITTING_PREXIF) >= 0]
            assert len(fitting_functions_list) in [0, 1], "At most 1 fitting function must be provided"
                          
            # Fit object on concatenated data
            if len(fitting_functions_list) > 0:
                if self.verbose:
                    print(f"Fitting function detected: {fitting_functions_list[0]}")
                df = pd.concat(list(df_dict.values()))
                params = self.functions_by_name[fitting_functions_list[0]](df, params)
                del df
                gc.collect()

            # If function is row-level, append to buffer
            # and do not apply
            if get_preprocessing_function_level(function.__name__) == argus.ROW_LEVEL:
                row_level_functions_buffer.append((function, params))
            else:
                df_dict, row_level_functions_buffer = self._apply_row_level_functions(df_dict, row_level_functions_buffer)

                # Transform each dataset
                for key, _df in df_dict.items():
                    if self.verbose:
                        print(f'Applying {function} with parameters {params}...')
                    df_dict[key] = function(df_dict[key], params)
        if len(row_level_functions_buffer) > 0:
            df_dict, row_level_functions_buffer = self._apply_row_level_functions(df_dict, row_level_functions_buffer)
        
        df_dict.update(null_dfs_dict)
        return df_dict

    def _apply_row_level_functions(self, df_dict: dict, row_level_functions_buffer: list):
        # Check if buffer is not empty
        if len(row_level_functions_buffer) > 0:
            if self.verbose:
                print(f"Current buffer: {row_level_functions_buffer}")
            for key, _df in df_dict.items():
                df_dict[key] = self._apply_function(df_dict[key], row_level_functions_buffer)
            # Then flush
            row_level_functions_buffer = []
        return df_dict, row_level_functions_buffer

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