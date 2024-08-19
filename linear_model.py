"""Functions for preprocessing the dataset for a linear model, and solving the corresponding ambulance location problem using a greedy algorithm.

The naive approach to fitting a linear model performs no feature engineering. Instead, we expand features into binary vectors to capture diminshing returns.

If the coefficients satisfy a monotonicity property, we don't need to use a solver, and can instead use a greedy algorithm.

For our experiments, we solve a MIP model instead of using the greedy algorithm, but I'm still keeping the greedy algorithm here for reference.
"""
from typing import Any
import numpy as np

def binarize_solutions(X: np.ndarray) -> np.ndarray:
    """Custom binarization for ambulance location solutions.
    
    Equivalent to the following:
    np.array([[k < x_j for x_j in x for k in np.arange(X.max())] for x in X])
    """
    k = np.arange(X.max()).reshape(1, 1, -1)
    X_new = k < X[..., np.newaxis]
    X_new = X_new.reshape(X_new.shape[0], -1)
    return X_new

def keep_relevant_coefficients(coef: np.ndarray, n_facilities: int, facility_capacity: int) -> np.ndarray:
    """Preprocess linear model coefficients by keeping only the relevant coefficients.

    For each facility, keep only the first facility_capacity coefficients.
    """
    n_coefs_per_facility = coef.shape[0] // n_facilities
    mask = [True]*facility_capacity + [False]*(n_coefs_per_facility - facility_capacity)
    mask *= n_facilities
    return coef[mask]

def check_coefficients_monotone(coef: np.ndarray, facility_capacity: int, max: bool) -> bool:
    """Check if coefficients are monotonically increasing or decreasing for each facility.

    We expect diminishing returns as the number of ambulances at a facility increases, leading to monotonicity.
    
    Coefficients should be decreasing if maximizing, and increasing if minimizing.
    """
    n_facilities = coef.shape[0] // facility_capacity
    for j in range(n_facilities):
        facility_coefs = coef[j*facility_capacity:(j+1)*facility_capacity]
        if (
            (max and (np.diff(facility_coefs) > 0).any())
            or (not max and (np.diff(facility_coefs) < 0).any())
        ):
            return False
    return True

def linear_model_greedy_solution(model: Any, n_facilities: int, facility_capacity: int, n_ambulances: int, max: bool) -> np.ndarray:
    """Once the linear model is trained, find the optimal solution using a greedy algorithm.
    
    Raises error if coefficients do not satisfy monotonicity property.
    """
    coef = keep_relevant_coefficients(model.coef_, n_facilities, facility_capacity)
    assert check_coefficients_monotone(coef, facility_capacity, max)
    return _greedy_solution(coef, n_facilities, n_ambulances, max)

def _greedy_solution(coef: np.ndarray, n_facilities: int, n_ambulances: int, max: bool) -> np.ndarray:
    """Locate ambulances in order to optimize linear model's prediction.
    
    If coefficients are monotone for each facility, the greedy solution turns out to be optimal.
    """
    facility_capacity = coef.shape[0] // n_facilities
    if max:
        coef = -coef
    station_order = np.argsort(coef) // facility_capacity
    return np.bincount(station_order[:n_ambulances], minlength=n_facilities)
