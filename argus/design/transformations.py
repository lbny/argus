import numpy as np
from argus.utils import to_array

def to_classes(x, n_classes=2, min_value=None, max_value=None, agg_function=None, verbose=False):
    x = to_array(x)
    n_classes = int(n_classes)
    if not min_value:
        min_value = np.min(x)
    if not max_value:
        max_value = np.max(x)
    if verbose:
        print(f"Value max: {min_value}")
        print(f"Value min: {max_value}")
    if not agg_function:
        agg_function = np.mean
    
    step = (max_value - min_value) / n_classes
    classes_array = np.zeros((x.shape[0],))
    for j in range(n_classes):
        if verbose:
            print(f"Interval lower bound: {min_value + j * step}")
            print(f"Interval upper bound: {min_value + (j + 1) * step}")
        if j < n_classes - 1:
            idx = np.where((0 <= x - min_value - j * step) & (x - min_value - j * step < step))[0]
        else:
            # if last chunk, take inferior or equal (instead of strictly inferio) to max value
            # n_steps * steps =/= max_value because of float approximation
            idx = np.where((0 <= x - min_value - j * step) & (x <= max_value))[0]
        classes_array[idx] = agg_function(x[idx])
    return classes_array