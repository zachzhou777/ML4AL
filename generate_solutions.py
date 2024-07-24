#!/usr/bin/env python3
"""Script to generate solutions<Process>.csv files for HTCondor."""
import os
import argparse
import itertools
import numpy as np

def generate_random_solutions(
    min_ambulances: int,
    max_ambulances: int,
    n_stations: int,
    alpha: float,
    n_solutions: int,
    seed: int | None = None
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
    
    n_solutions : int
        Number of solutions to sample.
    
    seed : int or None, optional
        Random seed.
    
    Returns
    -------
    np.ndarray of shape (n_solutions, n_stations)
        Randomly generated solutions.
    """
    rng = np.random.default_rng(seed)
    alpha = np.full(n_stations, alpha)
    cycle_iter = itertools.cycle(range(min_ambulances, max_ambulances+1))
    X = []
    for p in rng.dirichlet(alpha, n_solutions):
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
    parser.add_argument(
        '--region_id', type=int, required=True,
        help="Region ID; sets min_ambulances, max_ambulances, and n_stations"
    )
    parser.add_argument(
        '--alpha', type=float, default=3,
        help="Shared concentration parameter for Dirichlet distribution"
    )
    parser.add_argument(
        '--solutions_per_total_ambulance_count', type=int, default=10000,
        help="Number of random solutions to generate for each possible total ambulance count"
    )
    parser.add_argument(
        '--n_jobs', type=int, default=100,
        help="Number of jobs to run, results in one solutions<Process>.csv file per job"
    )
    parser.add_argument(
        '--solutions_dir', type=str, default='sim_solutions',
        help="Directory to save solutions<Process>.csv files"
    )
    args = parser.parse_args()

    region_settings = {
        1: (40, 60, 46),
        3: (12, 22, 17),
        5: (17, 33, 20)
    }
    min_ambulances, max_ambulances, n_stations = region_settings[args.region_id]

    n_solutions = (max_ambulances - min_ambulances + 1) * args.solutions_per_total_ambulance_count
    solutions = generate_random_solutions(
        min_ambulances,
        max_ambulances,
        n_stations,
        args.alpha,
        n_solutions
    )
    
    # Save solutions to solutions<Process>.csv files
    os.makedirs(args.solutions_dir, exist_ok=True)
    generated_solutions = set()  # Keep track of unique solutions
    for i, some_solutions in enumerate(np.split(solutions, args.n_jobs)):
        for soln in some_solutions:
            generated_solutions.add(tuple(soln))
        np.savetxt(
            os.path.join(args.solutions_dir, f'solutions{i}.csv'),
            some_solutions, delimiter=',', fmt='%d'
        )
    print(f"{len(generated_solutions)}/{n_solutions} solutions are unique")
