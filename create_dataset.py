#!/usr/bin/env python3
"""Python script to create dataset from the results<Process>.csv files that were created using HTCondor.

Usage:
    python create_dataset.py --region_id <int> [--n_jobs <int>] [--n_replications <int>] [--results_dir <str>]
"""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import trange

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_id', type=int, required=True,
                        help="Region ID (1 = Toronto, 4 = Muskoka)")
    parser.add_argument('--n_jobs', type=int, default=100,
                        help="Number of settings<Process>.csv files")
    parser.add_argument('--n_replications', type=int, default=10,
                        help="Number of replications per job")
    parser.add_argument('--results_dir', type=str, default='sim_results',
                        help="Directory containing results<Process>.csv files")
    parser.add_argument('--shuffle', type=bool, default=True,
                        help="Whether to shuffle the dataset before saving to file (useful if jobs sample the solution space differently)")
    parser.add_argument('--dataset_file', type=str, default='dataset.csv',
                        help="Filename of the dataset to be created")
    args = parser.parse_args()

    # Set n_stations and n_demand_nodes based on region_id
    if args.region_id == 1:  # Toronto
        n_stations = 46
        n_demand_nodes = 67
    elif args.region_id == 4:  # Muskoka
        n_stations = 5
        n_demand_nodes = 62
    else:
        raise ValueError("region_id not supported")
    
    station_col_names = [f'station{i}' for i in range(n_stations)]
    covered_total_col_names = sum([[f'covered{i}', f'total{i}'] for i in range(n_demand_nodes)], [])
    coverage_col_names = [f'coverage{i}' for i in range(n_demand_nodes)]
    header = station_col_names + covered_total_col_names

    dataset = []
    for i in trange(args.n_jobs):
        results = pd.read_csv(os.path.join(args.results_dir, f'results{i}.csv'), names=header)

        # Group every n_replications rows together. Within a group:
        # - station columns are the same, so take the first row
        # - covered and total columns are summed
        results['solution'] = np.arange(len(results)) // args.n_replications
        station_cols = results.groupby('solution')[station_col_names].first()
        coverage_cols = results.groupby('solution')[covered_total_col_names].sum()

        # Compute coverage columns, merge with station columns
        for i in range(n_demand_nodes):
            coverage_cols[f'coverage{i}'] = coverage_cols[f'covered{i}'] / coverage_cols[f'total{i}']
        agg_results = pd.concat([station_cols, coverage_cols[coverage_col_names]], axis=1)

        dataset.append(agg_results)
    
    dataset = pd.concat(dataset)

    # Impute NaNs with column mean
    dataset.fillna(dataset[coverage_col_names].mean(), inplace=True)

    # Shuffle rows
    if args.shuffle:
        dataset = dataset.sample(frac=1)

    dataset.to_csv(args.dataset_file, index=False)
