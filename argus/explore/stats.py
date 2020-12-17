import numpy as np
import pandas as pd

def get_descriptive_stats(x: np.array, options=['mean', 'var'], quantiles=None, keep_na=False, verbose=False) -> dict:
    if isinstance(x, pd.Series):
        x = x.values
    elif isinstance(x, list):
        x = np.array(x)
    if len(x.shape) > 1:
        if x.shape[1] > 1:
            raise Exception("x must be a 1-dimensional array")
    x = x.reshape(-1, 1)
    if not keep_na:
        x = x[np.where(1 - np.isnan(x))[0]]
    stats = {}
    if 'mean' in options:
        stats['mean'] = np.mean(x)
    if 'var' in options:
        stats['var'] = np.var(x)
    if 'std' in options:
        stats['std'] = np.std(x)
    if 'min' in options:
        stats['min'] = np.min(x)
    if 'max' in options:
        stats['max'] = np.max(x)
    if quantiles:
        assert isinstance(quantiles, float) or isinstance(quantiles, list), "quantiles must be a float or a list"
        if isinstance(quantiles, float):
            quantiles = [quantiles]
        sorted_quantiles = sorted(quantiles)
        for k, v in zip(sorted(quantiles), np.quantile(x, sorted(quantiles))):
            stats[f"{int(k * 100)}_quantile"] = v
    if verbose:
        print(stats)
    return stats