#!/usr/bin/env python3
"""Python script to be run from HTCondor.

Runs each solution from settings<Process>.csv and writes the results to results<Process>.csv.

Usage:
    python run_job.py <Process>
"""
import sys
import csv
import pandas as pd
from simulation import Simulation

solutions = []
with open(f'settings{sys.argv[1]}.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        solution = [int(x) for x in row]
        solutions.append(solution)

sim = Simulation.load_instance('simulation.pkl')
n_stations = sim.average_response_times.shape[0]
all_results = []
for solution in solutions:
    results = sim.run(solution)
    solution_cols = pd.DataFrame([solution]*sim.n_replications)
    results = pd.concat([solution_cols, results], axis=1)
    all_results.append(results)
all_results = pd.concat(all_results)
all_results.to_csv(f'results{sys.argv[1]}.csv', index=False, header=False)
