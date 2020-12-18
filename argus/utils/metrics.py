import numpy as np

from argus.utils import to_array, to_ndarray


def find_best_treshold(y_true, y_pred, metrics=None, classes=None, scale=100, verbose=False):
    
    if classes:
        classes = to_array(classes)
    if not isinstance(metrics, list):
        metrics = [metrics]
    if len(y_true.shape) == 1:
        y_true = to_array(y_true)
        y_true = y_true.reshape(-1, 1)
    else:
        y_true = to_ndarray(y_true)
    if len(y_pred.shape) == 1:
        y_pred = to_array(y_pred)
        y_pred = y_pred.reshape(-1, 1)
    else:
        y_pred = to_ndarray(y_pred)
    assert y_true.shape == y_pred.shape, f"y_true and y_pred must have the same dimension, currently: mismatch {y_true.shape} {y_pred.shape}"
    if classes:
        assert classes.shape[0] == y_pred.shape[1], f"classes and y_true/y_pred columns must have same dimensionality: mismatch classes {classes.shape[0]}, y_pred {y_pred.shape[1]}"
    else:
        classes = np.arange(y_pred.shape[1])
        
    def search(y_true: np.array, y_pred: np.array, scale: int, metric_function):
        # y_true and y_pred as vectors
        best_score = 0
        best_threshold = 0
        for t in range(scale):
            score = metric_function(y_true, np.array(y_pred > float(t / scale), dtype=np.uint8))
            if score > best_score:
                best_threshold = float(t / scale)
                best_score = score
        return best_threshold, best_score
    
    threhsolds_dict = {}
    for j in range(y_true.shape[1]):
        for metric in metrics:
            best_threshold, best_score = search(y_true[:, j], y_pred[:, j], scale=scale, metric_function=metric)
            if verbose:
                print(f"Class {classes[j]} : best score = {best_score} with threshold = {best_threshold}")
            threhsolds_dict[classes[j]] = {"best_score": best_score, "best_threshold": best_threshold}
    return threhsolds_dict
            