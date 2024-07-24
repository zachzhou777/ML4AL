#!/usr/bin/env python3
"""Script to create dataset after running HTCondor."""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from simulation import METRICS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solutions_dir', type=str, default='sim_solutions',
                        help="Directory containing solutions<Process>.csv files")
    parser.add_argument('--results_dir', type=str, default='sim_results',
                        help="Directory containing results<Process>.csv files")
    parser.add_argument('--dataset_file', type=str, default='dataset.csv',
                        help="Filename of the dataset to be created")
    args = parser.parse_args()

    solutions = []
    results = []
    n_jobs = len([
        filename for filename in os.listdir(args.solutions_dir)
        if filename.startswith('solutions')
    ])
    for i in trange(n_jobs):
        filename = os.path.join(args.solutions_dir, f'solutions{i}.csv')
        solutions_i = np.loadtxt(filename, delimiter=',')
        solutions.append(solutions_i)
        filename = os.path.join(args.results_dir, f'results{i}.csv')
        results_i = np.loadtxt(filename, delimiter=',')
        # If there is a NaN, then there were runs with zero arrivals
        # Since this is a very rare event, we'll raise an error if it occurs
        if np.isnan(results_i).any():
            raise ValueError(f"{filename} contains NaN values")
        results.append(results_i)
    solutions = np.vstack(solutions)
    results = np.vstack(results)
    dataset = np.hstack((solutions, results))
    col_names = [f'station{i}' for i in range(solutions.shape[1])] + METRICS
    dataset = pd.DataFrame(dataset, columns=col_names)
    dataset = dataset.convert_dtypes()  # Convert station columns to ints
    dataset.to_csv(args.dataset_file, index=False)
