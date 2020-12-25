from argus import design
from argus import utils
from argus import explore
from argus import run

import inspect

from argus.utils.data import *
from argus.design import PREPROCESSING_FUNCTIONS, FEATURES_FUNCTIONS

DATASETS = ['train', 'valid', 'test']
FITTING_PREXIF = 'fit_'

# Function levels
DATASET_LEVEL = 0
ROW_LEVEL = 1

