#!/usr/bin/env python3
"""Python script to generate settings<Process>.csv files for HTCondor.

In principle, if we had infinite computational resources, we would
generate all possible solutions and run the simulation for each one.
Specifically, we would only need to evaluate solutions that use all
available ambulances, i.e., a fixed total. Therefore, the most basic way
of generating solutions is to fix the total number of ambulances and
sample uniformly at random from the space of solutions using exactly
that number of ambulances.

However, this may not be the best way to curate the dataset. Consider
the following strategies:
- Sparsity: Randomly fix some stations to have zero ambulances.
- Random total number of ambulances: Self explanatory.
- "Consecutive" solutions: For a given solution, pick a random station
    and add ambulances to it to get new solutions.

Sparsity may be useful as it could allow the model to learn more about
individual stations. The other two strategies may be useful in teaching
the model to behave monotonically, i.e., adding ambulances improves
coverage. If every solution uses the same total number of ambulances,
then the model may end up learning that certain stations are "good" and
others are "bad." Adding ambulances to a bad station implies taking them
away from a good station, furthermore the model may incorrectly learn
that adding ambulances to a bad station reduces coverage. Using
consecutive solutions directly teaches monotonicity.

Note the following when using consecutive solutions:
- This script groups consecutive solutions together. To retrieve only
    the starting solutions, use something like `X[::n_consecutive]`.
- When using scikit-learn's train_test_split, use shuffle=False to keep
    consecutive solutions together. The solutions are already sampled
    randomly, so shuffling is not necessary.
"""
import os
import random
import argparse
from typing import Optional
import numpy as np

# This function generates all possible solutions using exactly n ambulances
# To see total number of solutions without generating them:
# import math; math.comb(n+k-1, k-1)
# For Toronto (n=234, k=46), there are 2.209e52 solutions
# Deprecated but kept in case it is needed in the future (for small instances, can simulate all possible solutions)
def generate_all_solutions(n, k):
    """Generate all k-tuples of non-negative integers whose sum is n.
    
    Source: https://stackoverflow.com/a/7748851
    """
    if k == 1:
        yield (n,)
    else:
        for i in range(n+1):
            for j in generate_all_solutions(n-i, k-1):
                yield (i,) + j

def generate_random_solutions(
        ambulances: tuple[int, int],
        stations: int,
        solutions: int = 1000,
        zeros: Optional[tuple[int, int]] = None
    ) -> np.ndarray:
    """Generate random solutions for the ambulance location problem.

    The method for generating a random solution is as follows:
    1. Determine (randomly) the total number of ambulances (`ambulances` parameter).
    2. Determine (randomly) a subset of stations to be excluded from random allocation of ambulances (`zeros` parameter).
    3. Sample uniformly at random from the space of solutions with the given total number of ambulances and the given subset of stations forced to have zero ambulances.

    Parameters
    ----------
    ambulances : tuple[int, int]
        For each solution, the total number of ambulances is sampled from random.randint(*ambulances); may get something outside this range due to rounding.
    
    stations : int
        Number of stations.
    
    solutions : int, optional
        Number of solutions to sample.
    
    zeros : tuple[int, int], optional
        If provided, for each solution, the number of stations forced to have zero ambulances is sampled from random.randint(*zeros).
    
    Returns
    -------
    np.ndarray of shape (solutions, stations)
        Randomly generated solutions.
    """
    X = []
    for _ in range(solutions):
        # p ~ Dirichlet([1, 1, ..., 1]) is sampled uniformly from the standard simplex
        # A trick: to ignore some components (i.e., force them to be zeros), set corresponding alphas to np.finfo(float).eps
        alpha = np.ones(stations)
        if zeros is not None:
            zeros_idx = random.sample(range(stations), random.randint(*zeros))
            alpha[zeros_idx] = np.finfo(float).eps
        p = np.random.dirichlet(alpha=alpha)
        # Generate a solution by rounding q*p where q = total number of ambulances
        x = np.round(random.randint(*ambulances)*p)
        X.append(x)
    return np.array(X)

def generate_consecutive_solutions(starting_solutions: np.ndarray, new_solutions: int = 1) -> np.ndarray:
    """Generate "consecutive" solutions for starting solutions.

    For each starting solution, pick a random station and add ambulances to it to get consecutive solutions.

    Parameters
    ----------
    starting_solutions : np.ndarray of shape (n, p)
        Starting solutions to generate "consecutive" solutions for.
    
    new_solutions : int, optional
        Number of new solutions to generate for each starting solution.
    
    Returns
    -------
    np.ndarray of shape (new_solutions*n, p)
        Consecutive solutions. Includes starting solutions; every (new_solutions+1)-th solution is a starting solution.
    """
    X = []
    p = starting_solutions.shape[1]
    for x in starting_solutions:
        X.append(x)
        station_idx = random.randint(0, p-1)
        for _ in range(new_solutions):
            x = x.copy()
            x[station_idx] += 1
            X.append(x)
    return np.array(X)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_id', type=int, required=True,
                        help="Region ID (1 = Toronto, 4 = Muskoka); sets total number of ambulances if `ambulances` argument is not provided")
    parser.add_argument('--n_jobs', type=int, default=200,
                        help="Number of jobs to run, results in one settings<Process>.csv file per job")
    parser.add_argument('--solutions_per_job', type=int, default=500,
                        help="Number of solutions per job, equivalent to number of lines per settings<Process>.csv file")
    parser.add_argument('--ambulances', type=int, nargs=2, default=None,
                        help="For each solution, the total number of ambulances is sampled from random.randint(*ambulances); may get something outside this range due to rounding")
    parser.add_argument('--zeros', type=int, nargs=2, default=None,
                        help="For each solution, the number of stations forced to have zero ambulances is sampled from random.randint(*zeros)")
    parser.add_argument('--n_consecutive', type=int, default=1,
                        help="Number of consecutive solutions to generate (1 means no consecutive solutions, all solutions are random)")
    parser.add_argument('--settings_dir', type=str, default='sim_settings',
                        help="Directory to save settings<Process>.csv files")
    args = parser.parse_args()

    # Check that the total number of solutions (n_jobs*solutions_per_job) is divisible by n_consecutive
    if (args.n_jobs*args.solutions_per_job) % args.n_consecutive != 0:
        raise ValueError("Total number of solutions (n_jobs*solutions_per_job) must be divisible by n_consecutive")
    
    # Set n_ambulances and n_stations
    if args.region_id == 1:  # Toronto
        # Source: https://www.toronto.ca/wp-content/uploads/2021/04/9765-Annual-Report-2020-web-final-compressed.pdf
        n_ambulances = 234
        n_stations = 46
    elif args.region_id == 4:  # Muskoka
        # TODO: Check number of ambulances makes sense
        n_ambulances = 30
        n_stations = 5
    else:
        raise ValueError("region_id not supported")
    
    os.makedirs(args.settings_dir, exist_ok=True)
    generated_solutions = set()  # Keep track of unique solutions

    # Generate starting solutions
    n_starting_solutions = (args.n_jobs*args.solutions_per_job) // args.n_consecutive
    if args.ambulances is None:
        args.ambulances = (n_ambulances, n_ambulances)
    solutions = generate_random_solutions(args.ambulances, n_stations, n_starting_solutions, args.zeros)

    # Generate consecutive solutions
    if args.n_consecutive > 1:
        solutions = generate_consecutive_solutions(solutions, args.n_consecutive-1)
    
    # Save solutions to settings<Process>.csv files
    for i, some_solutions in enumerate(np.split(solutions, args.n_jobs)):
        for soln in some_solutions:
            generated_solutions.add(tuple(soln))
        np.savetxt(os.path.join(args.settings_dir, f'settings{i}.csv'), some_solutions, delimiter=',', fmt='%d')
    print(f"{len(generated_solutions)}/{args.n_jobs*args.solutions_per_job} solutions are unique")
