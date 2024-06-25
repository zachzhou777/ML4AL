#!/usr/bin/env python3
"""Python script to generate settings<Process>.csv files for HTCondor.

TODO: Update this docstring. Most of the information is outdated.

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
import argparse
import itertools
import numpy as np

def generate_random_solutions(
    min_ambulances: int,
    max_ambulances: int,
    n_stations: int,
    alpha: float,
    n_solutions: int
) -> np.ndarray:
    """Generate random solutions for the ambulance location problem.

    To generate a single solution with n_ambulances ambulances, we first
    sample a probability vector p from a Dirichlet distribution, then
    scale p by n_ambulances, and finally round the result while ensuring
    the vector sums to exactly n_ambulances.

    Parameters
    ----------
    min_ambulances, max_ambulances : int
        Minimum and maximum number of ambulances for each solution.

        When generating solutions, the number of ambulances cycles
        through the range of possible values.
    
    n_stations : int
        Number of stations.
    
    alpha : float
        The value of each concentration parameter in the Dirichlet
        distribution.

        Setting alpha=1 results in sampling p from the standard simplex
        uniformly at random. This (I believe) leads to sampling from the
        solution space (non-negative integer vectors whose sum is
        n_ambulances) uniformly at random (it is not obvious whether
        rounding preserves uniformity).
        
        Higher values of alpha make the distribution more concentrated
        around the uniform distribution. This leads to sampling
        solutions with close to the same number of ambulances at each
        station.
    
    n_solutions : int, optional
        Number of solutions to sample.
    
    Returns
    -------
    np.ndarray of shape (n_solutions, n_stations)
        Randomly generated solutions.
    """
    alpha = np.full(n_stations, alpha)
    cycle_iter = itertools.cycle(range(min_ambulances, max_ambulances+1))
    X = []
    for p in np.random.dirichlet(alpha=alpha, size=n_solutions):
        n_ambulances = next(cycle_iter)
        # Separate n_ambulances*p into integer and fractional parts
        remainder, x = np.modf(n_ambulances*p)
        x = x.astype(int)
        # x.sum() almost surely falls short of n_ambulances
        deficit = n_ambulances - x.sum()
        # Round largest `deficit` remainders
        round_up = np.argsort(-remainder)[:deficit]
        x[round_up] += 1
        X.append(x)
    return np.array(X)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_id', type=int, required=True,
                        help="Region ID (1 = Toronto, 4 = Muskoka); sets min_ambulances, max_ambulances, and n_stations")
    parser.add_argument('--alpha', type=float, default=2,
                        help="Shared concentration parameter for Dirichlet distribution")
    parser.add_argument('--n_jobs', type=int, default=200,
                        help="Number of jobs to run, results in one settings<Process>.csv file per job")
    parser.add_argument('--solutions_per_job', type=int, default=500,
                        help="Number of solutions per job, equivalent to number of lines per settings<Process>.csv file")
    parser.add_argument('--settings_dir', type=str, default='sim_settings',
                        help="Directory to save settings<Process>.csv files")
    args = parser.parse_args()
    
    # Set min_ambulances, max_ambulances, and n_stations
    if args.region_id == 1:  # Toronto
        min_ambulances = 50
        max_ambulances = 100
        n_stations = 46
    elif args.region_id == 4:  # Muskoka
        # TODO: May need to change these values
        min_ambulances = 10
        max_ambulances = 50
        n_stations = 5
    else:
        raise ValueError("region_id not supported")
    
    os.makedirs(args.settings_dir, exist_ok=True)
    generated_solutions = set()  # Keep track of unique solutions

    # Generate solutions
    solutions = generate_random_solutions(min_ambulances, max_ambulances, n_stations, args.alpha, args.n_jobs*args.solutions_per_job)
    
    # Save solutions to settings<Process>.csv files
    for i, some_solutions in enumerate(np.split(solutions, args.n_jobs)):
        for soln in some_solutions:
            generated_solutions.add(tuple(soln))
        np.savetxt(os.path.join(args.settings_dir, f'settings{i}.csv'), some_solutions, delimiter=',', fmt='%d')
    print(f"{len(generated_solutions)}/{args.n_jobs*args.solutions_per_job} solutions are unique")
