#!/usr/bin/env python3
"""Python script to create dataset.csv from the results<Process>.csv files that were created using HTCondor.

Usage:
    python create_dataset.py
"""
import os
import numpy as np
import pandas as pd
from tqdm import trange

# Change these values as needed
n_stations = 8
n_demand_nodes = 62
n_jobs = 100
n_replications = 10

station_col_names = [f'station{i}' for i in range(n_stations)]
covered_total_col_names = sum([[f'covered{i}', f'total{i}'] for i in range(n_demand_nodes)], [])
coverage_col_names = [f'coverage{i}' for i in range(n_demand_nodes)]
header = station_col_names + covered_total_col_names

dataset = []
for i in trange(n_jobs):
    results = pd.read_csv(os.path.join('sim_results', f'results{i}.csv'), names=header)

    # Group every n_replications rows together. Within a group:
    # - station columns are the same, so take the first row
    # - covered and total columns are summed
    results['solution'] = np.arange(len(results)) // n_replications
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

dataset.to_csv('dataset.csv', index=False)
