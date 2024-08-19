#!/usr/bin/env python3
"""Script to run all MIP models and write results to file."""
import csv
import pickle
import time
import numpy as np
import pandas as pd
from gurobipy import GRB
from ems_data import *
from simulation import *
from dataset import *
from linear_model import *
from neural_network import *
from mip_models import *

RESULTS_FILE = 'results.csv'
TOTAL_AMBULANCE_COUNTS = {
    1: [43, 50, 57],
    3: [14, 17, 20],
    5: [20, 25, 30]
}
FACILITY_CAPACITY = 4
TIME_LIMIT = 600
SUCCESS_PROB = 0.8
# Estimated using best solution from dataset w.r.t coverage_9min for Toronto and Peel, coverage_15min for Simcoe
BUSY_FRACTION = {
    1: {43: 0.5378511773938689, 50: 0.4369429489820591, 57: 0.3874100103626275},
    3: {14: 0.288004162638955, 17: 0.2288075320373579, 20: 0.1880189046214391},
    5: {20: 0.3565014408893782, 25: 0.2769021029392569, 30: 0.2293679658137418}
}
# Estimated using best solution from dataset w.r.t response_time_mean
SERVICE_RATE = {
    1: {43: 22.94662688807342, 50: 23.824098402950032, 57: 24.11136495764099},
    3: {14: 19.377337143021244, 17: 20.266597902930563, 20: 20.77655572881174},
    5: {20: 21.448296831622255, 25: 21.869929046629107, 30: 21.966665014684228}
}

if __name__ == '__main__':
    with open(RESULTS_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['instance', 'model_name', 'solution', 'total_runtime', 'gurobi_runtime', 'mipgap', 'objval', 'objbound', *METRICS])
    
    with open('ems_data.pkl', 'rb') as f:
        ems_data = pickle.load(f)
    
    for region_id in [1, 3, 5]:
        ems_data.region_id = region_id
        demand_nodes = EMSData.read_patient_locations(region_id, test_id=0)
        n_stations = len(ems_data.stations)
        n_demand_nodes = len(demand_nodes)
        distance = Simulation.driving_distance(demand_nodes, ems_data.stations)
        sim = Simulation(ems_data, n_days=100, n_replications=5)
        dataset = pd.read_csv(f'dataset{region_id}.csv')
        if region_id in [1, 5]:
            coverage_metric = 'coverage_9min'
            threshold = MEXCLP_THRESHOLD_9MIN
        else:
            coverage_metric = 'coverage_15min'
            threshold = MEXCLP_THRESHOLD_15MIN

        for n_ambulances in TOTAL_AMBULANCE_COUNTS[region_id]:
            instance = f'{REGION_ID_TO_NAME[region_id]}-{n_ambulances}'

            # Best solutions from dataset
            rows = []
            for metric in [coverage_metric, 'survival_rate', 'response_time_mean']:
                solution, results = best_solution_from_dataset(dataset, n_ambulances, metric)
                rows.append(
                    [instance, f'Best-{metric}', solution.tolist(), None, None, None, None, None]
                    + [results[metric] for metric in METRICS]
                )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            # MEXCLP
            start = time.perf_counter()
            solution, model = mexclp(
                n_ambulances=n_ambulances,
                distance=distance,
                threshold=threshold,
                busy_fraction=BUSY_FRACTION[region_id][n_ambulances],
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'MEXCLP', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # p-Median with queueing
            start = time.perf_counter()
            solution, model = pmedian_with_queueing(
                n_ambulances=n_ambulances,
                distance=distance,
                arrival_rate=ems_data.avg_calls_per_day,
                service_rate=SERVICE_RATE[region_id][n_ambulances],
                success_prob=SUCCESS_PROB,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'pMQ', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # Linear-Coverage
            with open(f'trained_ml_models/{REGION_ID_TO_NAME[region_id].lower()}_coverage_linear.pkl', 'rb') as f:
                ridge = pickle.load(f)
            start = time.perf_counter()
            solution, model = linear_based_model(
                n_ambulances=n_ambulances,
                optimization_sense=GRB.MAXIMIZE,
                coef=keep_relevant_coefficients(ridge.coef_, n_stations, FACILITY_CAPACITY),
                intercept=ridge.intercept_,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'Linear-Coverage', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # Linear-Survival
            with open(f'trained_ml_models/{REGION_ID_TO_NAME[region_id].lower()}_survival_linear.pkl', 'rb') as f:
                ridge = pickle.load(f)
            start = time.perf_counter()
            solution, model = linear_based_model(
                n_ambulances=n_ambulances,
                optimization_sense=GRB.MAXIMIZE,
                coef=keep_relevant_coefficients(ridge.coef_, n_stations, FACILITY_CAPACITY),
                intercept=ridge.intercept_,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'Linear-Survival', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # Linear-Median
            with open(f'trained_ml_models/{REGION_ID_TO_NAME[region_id].lower()}_median_linear.pkl', 'rb') as f:
                ridge = pickle.load(f)
            start = time.perf_counter()
            solution, model = linear_based_model(
                n_ambulances=n_ambulances,
                optimization_sense=GRB.MINIMIZE,
                coef=keep_relevant_coefficients(ridge.coef_, n_stations, FACILITY_CAPACITY),
                intercept=ridge.intercept_,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'Linear-Median', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # DT-Coverage
            with open(f'trained_ml_models/{REGION_ID_TO_NAME[region_id].lower()}_coverage_dt.pkl', 'rb') as f:
                dt = pickle.load(f)
            start = time.perf_counter()
            solution, model = dt_based_model(
                n_ambulances=n_ambulances,
                optimization_sense=GRB.MAXIMIZE,
                dt=dt,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'DT-Coverage', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # DT-Survival
            with open(f'trained_ml_models/{REGION_ID_TO_NAME[region_id].lower()}_survival_dt.pkl', 'rb') as f:
                dt = pickle.load(f)
            start = time.perf_counter()
            solution, model = dt_based_model(
                n_ambulances=n_ambulances,
                optimization_sense=GRB.MAXIMIZE,
                dt=dt,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'DT-Survival', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            # DT-Median
            with open(f'trained_ml_models/{REGION_ID_TO_NAME[region_id].lower()}_median_dt.pkl', 'rb') as f:
                dt = pickle.load(f)
            start = time.perf_counter()
            solution, model = dt_based_model(
                n_ambulances=n_ambulances,
                optimization_sense=GRB.MINIMIZE,
                dt=dt,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'DT-Median', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # MLP-Coverage
            weights, biases = MLP.load_npz(f'trained_ml_models/{REGION_ID_TO_NAME[region_id].lower()}_coverage.npz')
            start = time.perf_counter()
            solution, model = mlp_based_model(
                n_ambulances=n_ambulances,
                optimization_sense=GRB.MAXIMIZE,
                weights=weights,
                biases=biases,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'MLP-Coverage', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # MLP-Survival
            weights, biases = MLP.load_npz(f'trained_ml_models/{REGION_ID_TO_NAME[region_id].lower()}_survival.npz')
            start = time.perf_counter()
            solution, model = mlp_based_model(
                n_ambulances=n_ambulances,
                optimization_sense=GRB.MAXIMIZE,
                weights=weights,
                biases=biases,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'MLP-Survival', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # MLP-Median
            weights, biases = MLP.load_npz(f'trained_ml_models/{REGION_ID_TO_NAME[region_id].lower()}_median.npz')
            start = time.perf_counter()
            solution, model = mlp_based_model(
                n_ambulances=n_ambulances,
                optimization_sense=GRB.MINIMIZE,
                weights=weights,
                biases=biases,
                facility_capacity=FACILITY_CAPACITY,
                time_limit=TIME_LIMIT,
                verbose=False
            )
            total_runtime = time.perf_counter() - start
            results = sim.run(solution).mean()
            row = (
                [instance, 'MLP-Median', solution.tolist(), total_runtime, model.Runtime, model.MIPGap, model.ObjVal, model.ObjBound]
                + [results[metric] for metric in METRICS]
            )
            with open(RESULTS_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
