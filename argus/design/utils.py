import numpy as np

import pickle

def get_functions(config: dict, functions_by_name: dict, file_folder=None, seed=1234) -> list:
    """
    :config: A dictionary, keys are functions names, values are either true boolean
    either a dictionary of parameters. Parameters must be consistent with the corresponding
    function declared in function_by_name dictinary. In order to fit an object beforehand,
    declare a function with the prefix "fit_" (or any prefix declared in argus.FITTING_PREXIF)
    that will be called on the whole dataset
    :function_by_name: A dictionary, keys are function names, values the actual function objects
    :file_folder: A string, path to the folder containing the files required to apply functions
    Return:
        A list of functions with corresponding parameters
    """
    functions_list = []

    for func_dict in config:
        function_name = list(func_dict.keys())[0]
        params = func_dict[function_name]
        params['seed'] = seed
        assert function_name in functions_by_name, f"{function_name} function doesnt exist."
        functions_list.append((functions_by_name[function_name], format_parameters(params, file_folder)))
    return functions_list

def format_parameters(params: dict, file_folder: str) -> dict:
    for key, param in params.items():
        if isinstance(key, str):
            if '.' in key:
                # it has a file extension
                data = None
                try:
                    with open(osp.join(file_folder, key), 'r') as f:
                        data = f.read()
                        f.close()
                except:
                    with open(osp.join(file_folder, key), 'rb') as f:
                        data = pickle.load(f)
                        f.close()
                params[key] = data
    return params

