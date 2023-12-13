#!/usr/bin/env python3
"""Python script to generate settings<Process>.csv files for HTCondor.

Usage:
    python htcondor_setup.py
"""
import csv
import random

# Change these values as needed
n_jobs = 100
solutions_per_job = 5000
n_ambulances = 30
n_stations = 8

# Try to generate all solutions and store them in a list (if this fails, we'll need to use something like itertools.islice)
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

all_solutions = list(generate_all_solutions(n_ambulances, n_stations))
print(f'Total number of solutions: {len(all_solutions)}')

# Create settings<Process>.csv files
some_solutions = random.sample(all_solutions, n_jobs*solutions_per_job)
for i in range(n_jobs):
    with open(f'settings{i}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(some_solutions[solutions_per_job*i:solutions_per_job*(i+1)])
