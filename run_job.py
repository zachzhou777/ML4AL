#!/usr/bin/env python3
"""Python script to be run from HTCondor.

Runs each solution from solutions<Process>.csv and writes the results to results<Process>.csv.

Usage:
    python run_job.py <Process>
"""
import sys
import pickle
import numpy as np

solutions = np.loadtxt(f'solutions{sys.argv[1]}.csv', delimiter=',')
with open('simulation.pkl', 'rb') as f:
    sim = pickle.load(f)
results = []
for solution in solutions:
    result = sim.run(solution)
    result = np.nanmean(result, axis=0)
    results.append(result)
results = np.array(results)
np.savetxt(f'results{sys.argv[1]}.csv', results, delimiter=',')
