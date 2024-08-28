"""Miscellaneous functions for working with the datasets."""
import numpy as np
import pandas as pd
from simulation import *

def best_solution_from_dataset(dataset: pd.DataFrame, n_ambulances: int, metric: str) -> tuple[pd.Series, pd.Series]:
    """Finds the best solution w.r.t. a metric for a given maximum number of ambulances."""
    X = dataset.drop(columns=METRICS)
    Y = dataset[METRICS]
    indices = np.where(X.sum(axis=1) <= n_ambulances)[0]
    argmin_or_argmax = np.argmin if 'response_time' in metric else np.argmax
    y = Y[metric]
    best_idx = indices[argmin_or_argmax(y[indices])]
    return X.iloc[best_idx], Y.iloc[best_idx]

def remove_outliers_at_each_ambulance_count(X, y, k=1.5):
    """For each total number of ambulances, isolate samples with that total and remove outliers.

    Parameters
    ----------
    X, y : np.ndarray
        The dataset.
    
    k : float
        y values outside the range [Q1 - k*IQR, Q3 + k*IQR] are considered outliers.
    
    Returns
    -------
    X_new, y_new : np.ndarray
        The dataset with outliers removed.
    """
    X_sum = X.sum(axis=1)
    min_ambulances = X_sum.min()
    max_ambulances = X_sum.max()
    keep = np.full(X.shape[0], False)
    for total in range(min_ambulances, max_ambulances+1):
        y_subset = y[X_sum == total]
        q1 = np.percentile(y_subset, 25)
        q3 = np.percentile(y_subset, 75)
        iqr = q3 - q1
        keep |= (
            (X_sum == total)
            & (y >= q1 - k*iqr)
            & (y <= q3 + k*iqr)
        )
    return X[keep], y[keep]

def rescale(x: np.ndarray, min: float = 0, max: float = 1) -> np.ndarray:
    """Shift and scale data to the range [min, max]."""
    return (x - x.min()) / (x.max() - x.min()) * (max - min) + min
