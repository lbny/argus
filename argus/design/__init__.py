import inspect
from . import features
from . import preprocessings

import argus

PREPROCESSING_FUNCTIONS = dict(inspect.getmembers(preprocessings))
FEATURES_FUNCTIONS = dict(inspect.getmembers(features))


def get_preprocessing_function_level(function_name: str) -> int:
    if function_name in preprocessings.DATASET_LEVEL_FUNCTION_NAMES:
        return argus.DATASET_LEVEL
    elif function_name in preprocessings.ROW_LEVEL_FUNCTION_NAMES:
        return argus.ROW_LEVEL
    return None

def get_feature_function_level(function_name: str) -> int:
    if function_name in features.DATASET_LEVEL_FUNCTION_NAMES:
        return argus.DATASET_LEVEL
    elif function_name in features.ROW_LEVEL_FUNCTION_NAMES:
        return argus.ROW_LEVEL
    return None