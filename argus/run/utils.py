from ray import tune

from argus.design.features

import numpy as np
import pandas as pd

SHARED_FEATURES_KEY = 'shared_features'
EXPERIMENT_FEATURES_KEY = 'experiment_features'
TRAINING_FUNCTIONS_KEY = 'training_functions'

PARAM_TYPE_KEY = 'type'
PARAM_VALUES_KEY = 'val'

PARAM_TYPE_FUNCTION_MAPPING = {
    'randint': tune.randint,
    'randn': tune.randn,
    'uniform': tune.uniform,
    'loguniform': tune.loguniform,
    'qrandint': tune.qrandint,
    'qrandn': tune.qrandn,
    'quniform': tune.quniform,
    'qloguniform': tune.qloguniform,
    'choice': tune.choice,
    'grid_search': tune.grid_search
}

PARAM_FIXED_TYPE_KEY = 'fixed'

PARAM_TRAINING_FUNCTION_NAME_KEY = 'training_function_name'
PARAM_TRAINING_PARAMETERS_KEY = 'training_parameters'



def to_tune_format(config: dict, keywords: list=None) -> dict:
    """Convert a dictionary to a ray.tune configuration.

    Args:
        config (dict): [Hyperparameters search configuration. 
        Format :
        - shared_features
        - experiment_features
        - training_functions
        ]

    Returns:
        dict: [description]
    """
    tune_config = dict()
    if keywords:    

        for keyword in keywords:
            features = []
            if keyword == EXPERIMENT_FEATURES_KEY:
                for feature in config[keyword]:
                    features.append(to_tune_format(feature))
                tune_config[keyword] = features
                
            if keyword == TRAINING_FUNCTIONS_KEY:    
                if len(config[keyword]) == 1:
                    # one single training function to run
                    training_function = config[keyword][0]
                    training_function_name = list(training_function.keys())[0]
                    tune_config['training_function_name'] = training_function_name
                    tune_config[PARAM_TRAINING_PARAMETERS_KEY] = to_tune_format(training_function)
                else:
                    # multipule training functions to run
                    training_function_name_list = [list(_function.keys())[0] for _function in config[keyword]]
                    training_function_index_dict = dict(zip(training_function_name_list, np.arange(len(training_function_name_list))))
                    tune_config['training_function_name'] = tune.choice(training_function_name_list)
                    
                    tune_config[PARAM_TRAINING_PARAMETERS_KEY] = tune.sample_from(lambda spec: training_function_index_dict[spec.config.training_function_name])       

    else:
        for key in config.keys():
            if is_param(config[key]):
                tune_config[key] = parse_param(config[key])
            else:
                tune_config[key] = to_tune_format(config[key])
    return tune_config

def is_param(param: dict) -> bool:
    return PARAM_TYPE_KEY in param.keys() and PARAM_VALUES_KEY in param.keys()

def is_interval_function(function_name: str) -> bool:
    return function_name[0] == 'q'

def parse_param(param: dict):
    if param[PARAM_TYPE_KEY] == PARAM_FIXED_TYPE_KEY:
        return param[PARAM_VALUES_KEY]
    elif is_interval_function(param[PARAM_TYPE_KEY]):
        return PARAM_TYPE_FUNCTION_MAPPING[param[PARAM_TYPE_KEY]](
            param[PARAM_VALUES_KEY][0],
            param[PARAM_VALUES_KEY][1],
            param[PARAM_VALUES_KEY][2]
        )
    elif not is_interval_function(param[PARAM_TYPE_KEY]):
        return PARAM_TYPE_FUNCTION_MAPPING[param[PARAM_TYPE_KEY]](
            param[PARAM_VALUES_KEY][0],
            param[PARAM_VALUES_KEY][1]
        )
    return None
        
