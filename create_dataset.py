#!/usr/bin/env python3
"""Python script to create dataset from the results<Process>.csv files that were created using HTCondor."""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region_id', type=int, required=True,
                        help="Region ID (1 = Toronto, 4 = Muskoka)")
    parser.add_argument('--n_jobs', type=int, default=200,
                        help="Number of settings<Process>.csv files")
    parser.add_argument('--n_replications', type=int, default=3,
                        help="Number of replications per job")
    parser.add_argument('--results_dir', type=str, default='sim_results',
                        help="Directory containing results<Process>.csv files")
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
    
    sim_input_col_names = [f'solution_{i}' for i in range(n_stations)]
    sim_output_col_names = sum([[f'n_covered_{i}', f'response_time_{i}', f'n_arrivals_{i}'] for i in range(n_demand_nodes)], [])
    coverage_col_names = [f'coverage_{i}' for i in range(n_demand_nodes)]
    avg_response_time_col_names = [f'avg_response_time_{i}' for i in range(n_demand_nodes)]
    header = sim_input_col_names + sim_output_col_names
    output_col_names = coverage_col_names + avg_response_time_col_names

    dataset = []
    for i in trange(args.n_jobs):
        results = pd.read_csv(os.path.join(args.results_dir, f'results{i}.csv'), names=header)

        # Group every n_replications rows together. Within a group:
        # - Input (solution) columns are the same, so take the first row
        # - Output (n_covered, response_time, etc.) columns are summed
        results['soln_id'] = np.arange(len(results)) // args.n_replications
        station_cols = results.groupby('soln_id')[sim_input_col_names].first()
        output_cols = results.groupby('soln_id')[sim_output_col_names].sum()

        # Compute coverage and avg_response_time columns, merge with station columns
        for i in range(n_demand_nodes):
            output_cols[f'coverage_{i}'] = output_cols[f'n_covered_{i}'] / output_cols[f'n_arrivals_{i}']
            output_cols[f'avg_response_time_{i}'] = output_cols[f'response_time_{i}'] / output_cols[f'n_arrivals_{i}']
        agg_results = pd.concat([station_cols, output_cols[output_col_names]], axis=1)

        dataset.append(agg_results)
    
    dataset = pd.concat(dataset)

    # Impute NaNs with column mean
    dataset.fillna(dataset[output_col_names].mean(), inplace=True)

    dataset.to_csv(args.dataset_file, index=False)
