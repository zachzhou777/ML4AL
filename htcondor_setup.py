#!/usr/bin/env python3
"""Python script to generate settings<Process>.csv files for HTCondor.

There are two ways to generate solutions:
1. generate_all_solutions: This function generates all possible
    solutions, from which we can randomly sample a subset without
    replacement. This is only feasible when n and k are small, as the
    number of solutions is n+k-1 choose k-1.
2. generate_random_solutions: This function generates random solutions.
    It does not guarantee solutions are unique, but for large n and k,
    it is unlikely to generate duplicates.

When n and k are small (calculate n+k-1 choose k-1 to see if the total
number of solutions is reasonable), generate_all_solutions should be
used as it guarantees unique solutions. When n and k are large,
generate_random_solutions should be used as it is computationally
infeasible to generate all solutions, and the larger n and k are, the
less likely generate_random_solutions is to generate duplicates.

Usage:
    python htcondor_setup.py --region_id <int> --n_jobs <int> --solutions_per_job <int> --settings_dir <str> [--generate_all]
"""
import os
import csv
import random
import argparse
import numpy as np

# This function generates all possible solutions, so it should be used only when n and k are small
# To see total number of solutions without generating them:
# import math; math.comb(n+k-1, k-1)
# For Toronto (n=234, k=46), there are 2.209e52 solutions
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

# This function generates random solutions, so it can be used when n and k are large
# It does not guarantee solutions are unique, but for large n and k, it is unlikely to generate duplicates
def generate_random_solutions(n, k, m=1000, zeros=0):
    """Generate m random k-tuples of non-negative integers whose sum is n.

    The tricky part is that the sampling must be done uniformly at random.

    TODO: Docstring needs to be updated, still not decided on how to generate random solutions

    Parameters
    ----------
    n : int | tuple[int, int]
        If int, sum of each k-tuple. If tuple, interval to sample uniformly from (endpoints included).
    
    k : int
        Length of each k-tuple.
    
    m : int, optional
        Number of k-tuples to generate.
    
    zeros : float | int, optional
        If float, proportion of zeros in each k-tuple. If int, number of zeros in each k-tuple.
    
    Returns
    -------
    np.ndarray of shape (m, k)
        Each row is a k-tuple of non-negative integers whose sum is n.
    """
    # Sample p ~ Dirichlet([1, 1, ..., 1]); p is sampled uniformly at random from the (k-1)-simplex
    # TODO: A possibly useful trick, to force some components to be zeros, set corresponding alpha to np.finfo(float).eps
    p = np.random.dirichlet(alpha=np.ones(k), size=m)

    # n*p is roughly what we want, but need to be careful about rounding to ensure sum is n
    fractional_part, integral_part = np.modf(n*p)
    x = integral_part.astype(int)

    # Only round up the largest fractional parts until the sum is n
    deficit = n - x.sum(axis=1)
    round_up = np.argpartition(-fractional_part, deficit, axis=1)
    for i in range(m):
        x[i, round_up[i, :deficit[i]]] += 1

    return x

# TODO: Right now random solutions sample {x : sum(x) = n, x >= 0, x int}. We need to sample differently. Some strategies:
# 1. Add argument for sparsity, force some components to be zeros. Pick some bases to have ambulances, others get zero.
# 2. Add argument to allow total number of ambulances to be random. May help neural net behave monotonically, i.e., taking a solution and adding ambulances shouldn't make coverage worse.
# 3. Add argument to add "consecutive" solutions: if x is a solution, then x+(0,...,0,1,0,...,0) is also a solution. For each solution, pick a random station and add "consecutive" solutions. Should help with monotonicity.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_id', type=int, required=True,
                        help="Region ID (1 = Toronto, 4 = Muskoka)")
    parser.add_argument('--n_jobs', type=int, default=100,
                        help="Number of jobs to run, results in one settings<Process>.csv file per job")
    parser.add_argument('--solutions_per_job', type=int, default=1000,
                        help="Number of solutions per job, equivalent to number of lines per settings<Process>.csv file")
    parser.add_argument('--settings_dir', type=str, default='sim_settings',
                        help="Directory to save settings<Process>.csv files")
    parser.add_argument('--generate_all', action='store_true',
                        help="Whether to use generate_all_solutions or generate_random_solutions (see comments in code for explanation of which to use)")
    args = parser.parse_args()

    # Set n_ambulances (n) and n_stations (k)
    if args.region_id == 1:  # Toronto
        # Source: https://www.toronto.ca/wp-content/uploads/2021/04/9765-Annual-Report-2020-web-final-compressed.pdf
        n_ambulances = 75  #234  # TODO: Either switch back to 234 or add argument for n_ambulances
        n_stations = 46
    elif args.region_id == 4:  # Muskoka
        # TODO: Check number of ambulances makes sense
        n_ambulances = 30
        n_stations = 5
    else:
        raise ValueError("region_id not supported")
    
    os.makedirs(args.settings_dir, exist_ok=True)
    if args.generate_all:
        # Generate all solutions and randomly sample a subset
        all_solutions = list(generate_all_solutions(n_ambulances, n_stations))
        some_solutions = random.sample(all_solutions, args.n_jobs*args.solutions_per_job)
        # Write settings<Process>.csv files
        for i in range(args.n_jobs):
            with open(os.path.join(args.settings_dir, f'settings{i}.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(some_solutions[args.solutions_per_job*i:args.solutions_per_job*(i+1)])
    else:
        generated_solutions = set()  # Keep track of unique solutions
        for i in range(args.n_jobs):
            some_solutions = generate_random_solutions(n_ambulances, n_stations, args.solutions_per_job)
            for soln in some_solutions:
                generated_solutions.add(tuple(soln))
            np.savetxt(os.path.join(args.settings_dir, f'settings{i}.csv'), some_solutions, delimiter=',', fmt='%d')
        print(f"{len(generated_solutions)}/{args.n_jobs*args.solutions_per_job} solutions are unique")
