import math
from typing import Optional
import numpy as np
import pandas as pd
from scipy import optimize
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.torch import add_sequential_constr

# Precomputed distances (km) such that response time at most 9 or 15 minutes with probability 90%
MEXCLP_THRESHOLD_9MIN = 3.64
MEXCLP_THRESHOLD_15MIN = 11.55

def mexclp(
    n_ambulances: int,
    distance: np.ndarray,
    threshold: float,
    busy_fraction: float,
    demand: Optional[np.ndarray] = None,
    facility_capacity: Optional[int] = None,
    time_limit: Optional[float] = None,
    verbose: bool = False
) -> tuple[np.ndarray, float]:
    """Solve the MEXCLP model (Daskin, 1983).

    Parameters
    ----------
    n_ambulances : int
        Total number of ambulances.
    
    distance : np.ndarray of shape (n_demand_nodes, n_facilities)
        Distance matrix.
    
    threshold : float
        Maximum distance for a facility to cover a demand node.
    
    busy_fraction : float
        Average fraction of time an ambulance is busy.
    
    demand : np.ndarray of shape (n_demand_nodes,), optional
        Demand weights. If None, assume uniform demand.

        Demand is normalized so that the objective function is a probability.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    np.ndarray of shape (n_facilities,)
        Number of ambulances located at each facility.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    n_demand_nodes, n_facilities = distance.shape
    if demand is None:
        demand = np.ones(n_demand_nodes)
    demand = np.array(demand, dtype=float)
    demand /= demand.sum()
    
    model = gp.Model()
    model.Params.LogToConsole = verbose
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    x = model.addVars(n_facilities, lb=0, ub=facility_capacity, vtype=GRB.INTEGER)
    y = model.addVars(n_demand_nodes, n_ambulances, vtype=GRB.BINARY)
    model.setObjective(
        gp.quicksum(
            demand[i] * (1-busy_fraction)*busy_fraction**k * y[i, k]
            for k in range(n_ambulances) for i in range(n_demand_nodes)
        ),
        GRB.MAXIMIZE
    )
    model.addConstrs((
        x.sum(j for j in range(n_facilities) if distance[i, j] < threshold) >= y.sum(i, '*')
        for i in range(n_demand_nodes)
    ))
    model.addConstr(x.sum() <= n_ambulances)
    model.optimize()

    return np.array([int(x[j].X + 0.5) for j in range(n_facilities)])

def pmedian_with_queuing(
    n_ambulances: int,
    distance: np.ndarray,
    arrival_rate: float,
    service_rate: float,
    success_prob: float,
    demand: Optional[np.ndarray] = None,
    facility_capacity: Optional[int] = None,
    time_limit: Optional[float] = None,
    verbose: bool = False
) -> np.ndarray | None:
    """Solve a model that combines the p-median model with queuing
    constraints. Based on Marianov and Serra (2002) and Boutilier and
    Chan (2022).

    Note that the model may be infeasible if the arrival rate is too
    high relative to the service rate, in which case this function will
    return None.

    Parameters
    ----------
    n_ambulances : int
        Total number of ambulances.
    
    distance : np.ndarray of shape (n_demand_nodes, n_facilities)
        Distance matrix.
    
    arrival_rate : float
        Arrival rate (calls per unit time).
    
    service_rate : float
        Service rate of each server (calls per unit time).
    
    success_prob : float
        Desired success probability for queueing constraints.
    
    demand : np.ndarray of shape (n_demand_nodes,), optional
        Demand weights. If None, assume uniform demand.

        Demand is normalized so that the objective function is expected
        distance.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    np.ndarray of shape (n_facilities,) | None
        Number of ambulances located at each facility. If the model is
        infeasible, return None.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    n_demand_nodes, n_facilities = distance.shape
    if demand is None:
        demand = np.ones(n_demand_nodes)
    demand = np.array(demand, dtype=float)
    demand /= demand.sum()
    arrival_rates = arrival_rate * demand
    rho = compute_rho(max_servers=facility_capacity, success_prob=success_prob)

    model = gp.Model()
    model.Params.LogToConsole = verbose
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    x = model.addVars(n_facilities, facility_capacity, vtype=GRB.BINARY)
    y = model.addVars(n_demand_nodes, n_facilities, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    model.setObjective(
        gp.quicksum(
            demand[i]*distance[i, j]*y[i, j]
            for i in range(n_demand_nodes) for j in range(n_facilities)
        ),
        GRB.MINIMIZE
    )
    model.addConstrs((y.sum(i, '*') == 1 for i in range(n_demand_nodes)))
    model.addConstrs((y[i, j] <= x[j, 0] for i in range(n_demand_nodes) for j in range(n_facilities)))
    model.addConstrs((x[j, k] <= x[j, k-1] for j in range(n_facilities) for k in range(1, facility_capacity)))
    model.addConstrs((
        gp.quicksum(arrival_rates[i]*y[i, j] for i in range(n_demand_nodes))
        <= service_rate*gp.quicksum((rho[k+1] - rho[k])*x[j, k] for k in range(facility_capacity))
        for j in range(n_facilities)
    ))
    model.addConstr(x.sum() <= n_ambulances)
    model.optimize()
    if model.status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD}:
        return None
    
    x = np.array([[x[j, k].X for k in range(facility_capacity)] for j in range(n_facilities)])
    solution = np.rint(x.sum(axis=1)).astype(int)
    return solution

def compute_rho(max_servers: int, success_prob: float) -> list[float]:
    """Compute rho values needed by queuing constraints.

    Based on M/G/c/c queues, i.e., the Erlang loss model.
    
    Parameters
    ----------
    max_servers : int
        Maximum number of servers to consider.
    
    success_prob : float
        Desired success probability for queuing constraints.
    
    Returns
    -------
    np.ndarray of shape (max_servers+1,)
        rho[0] is 0. For k > 0, rho[k] is the rho value for k servers.
    """
    # Function used by scipy.optimize.root_scalar
    def rho_eq(rho, n_servers):
        lhs = (
            (rho**n_servers / math.factorial(n_servers))
            / sum((rho**k)/math.factorial(k) for k in range(n_servers+1))
        )
        rhs = 1 - success_prob
        return lhs - rhs
    
    rho = np.zeros(max_servers+1)
    for n_servers in range(1, max_servers+1):
        eq = lambda rho: rho_eq(rho, n_servers=n_servers)
        # Will raise a ValueError if 2*n_servers is not an upper bound
        result = optimize.root_scalar(eq, bracket=[0.0, 2*n_servers])
        rho[n_servers] = result.root
    
    return rho

def mlp_based_model(
    n_ambulances: int,
    optimization_sense: int,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    facility_capacity: Optional[int] = None,
    use_gurobi_ml: bool = False,
    scale_relu: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False
) -> np.ndarray:
    """Solve the MLP-based model.

    If solving a minimization problem, the final hidden layer's
    activation function is assumed to be ReLU. If solving a maximization
    problem, the final hidden layer's activation function is assumed to
    be the modified sigmoid function (sigmoid(z) for z > 0, 0.25*z + 0.5
    otherwise). Assume the final layer's weights are positive in order
    to exploit convexity/concavity of the final hidden layer's
    activation function and avoid introducing a binary variable for each
    unit in this layer.

    Parameters
    ----------
    n_ambulances : int
        Total number of ambulances.
    
    optimization_sense : {GRB.MINIMIZE, GRB.MAXIMIZE}
        Whether to solve a minimization or maximization problem.
    
    weights, biases : list[np.ndarray]
        Weights and biases of the MLP.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    use_gurobi_ml : bool, optional
        Use the Gurobi Machine Learning package to embed everything up
        to the last hidden layer (not including the last hidden layer's
        activation function).
    
    scale_relu : bool, optional
        How to model the ReLU activation function for hidden layers.
        Ignored if use_gurobi_ml is True.
        - If False, use the original approach seen in other papers.
        - If True, scale the net inputs to ReLU units so that they are
        in [-1, 1], then rescale the outputs.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    np.ndarray of shape (n_facilities,)
        Number of ambulances located at each facility.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    n_facilities = weights[0].shape[1]
    final_hidden_size = weights[-1].shape[1]

    model = gp.Model()
    model.Params.LogToConsole = verbose
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    x = model.addMVar(n_facilities, lb=0, ub=facility_capacity, vtype=GRB.INTEGER)
    z = model.addMVar(final_hidden_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    y = model.addMVar(final_hidden_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    # Objective function and constraints to embed the last hidden layer's activation function and the last linear layer
    if optimization_sense == GRB.MAXIMIZE:
        embed_modified_sigmoid(model, z, y)
    elif optimization_sense == GRB.MINIMIZE:
        # Embed last ReLU layer
        model.addConstr(y >= 0)
        model.addConstr(y >= z)
    else:
        raise ValueError("optimization_sense must be GRB.MINIMIZE or GRB.MAXIMIZE")
    model.setObjective(weights[-1][0]@y + biases[-1][0], optimization_sense)
    # Embed MLP up to the last hidden layer's activation function
    if use_gurobi_ml:
        sequential_layers = []
        for weight, bias in zip(weights[:-1], biases[:-1]):
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                linear.weight.copy_(torch.tensor(weight))
                linear.bias.copy_(torch.tensor(bias))
            sequential_layers.extend([linear, nn.ReLU()])
        sequential_layers.pop()  # Remove last ReLU
        sequential_model = nn.Sequential(*sequential_layers)
        add_sequential_constr(model, sequential_model, x, z)
        pass
    else:
        M_minus, M_plus = compute_bounds(weights[:-1], biases[:-1], n_ambulances)
        embed_mlp(model, weights[:-1], biases[:-1], x, z, M_minus, M_plus, scale_relu)
    model.addConstr(x.sum() <= n_ambulances)
    model.optimize()
    
    return np.rint(x.X).astype(int)

def embed_mlp(
    gurobi_model: gp.Model,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    input_vars: gp.MVar,
    output_vars: gp.MVar,
    M_minus: list[np.ndarray],
    M_plus: list[np.ndarray],
    scale_relu: bool = False
):
    """Embed an MLP in a Gurobi model.

    Assumes the MLP uses the ReLU activation function for hidden layers,
    and no activation function for the output layer. For our use case,
    we use this function to embed everything up to the final hidden
    layer (not including the final hidden layer's activation function).
    Thus, when we call this function, `weights` and `biases` exclude the
    last linear layer, and `output_vars` corresponds to the final hidden
    layer's pre-activation values.

    Parameters
    ----------
    gurobi_model : gp.Model
        Gurobi model.
    
    weights, biases : list[np.ndarray]
        Weights and biases of the MLP.
    
    input_vars : gp.MVar
        Decision variables used as input to the MLP.
    
    output_vars : gp.MVar
        Decision variables used as output from the MLP.
    
    M_minus : list[np.ndarray]
        Lower bounds for the net input to each hidden unit.
    
    M_plus : list[np.ndarray]
        Upper bounds for the net input to each hidden unit.
    
    scale_relu : bool, optional
        How to model the ReLU function for the hidden layers.
        - If False, use the original approach seen in other papers.
        - If True, scale the net inputs to ReLU units so that they are in [-1, 1], then rescale the outputs.
    """
    n_layers = len(weights)
    layer_dims = [weights[0].shape[1]] + [weight.shape[0] for weight in weights]

    # To make indexing consistent with paper, lists may start with None to give the illusion of 1-based indexing
    weights = [None] + weights
    biases = [None] + biases
    M_minus = [None] + M_minus
    M_plus = [None] + M_plus

    z = [None] + [gurobi_model.addMVar(dim, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) for dim in layer_dims[1:]]
    a = [None] + [gurobi_model.addMVar(dim, vtype=GRB.BINARY) for dim in layer_dims[1:-1]]
    h = [gurobi_model.addMVar(dim, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) for dim in layer_dims[:-1]]

    gurobi_model.addConstr(h[0] == input_vars)
    gurobi_model.addConstr(output_vars == z[-1])
    gurobi_model.addConstrs((z[ell] == weights[ell]@h[ell-1] + biases[ell] for ell in range(1, n_layers+1)))

    if scale_relu:
        M = [None] + [np.maximum(M_plus[ell], -M_minus[ell]) for ell in range(1, n_layers)]

        z_hat = [None] + [gurobi_model.addMVar(dim, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS) for dim in layer_dims[1:-1]]
        h_hat = [None] + [gurobi_model.addMVar(dim, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS) for dim in layer_dims[1:-1]]

        gurobi_model.addConstrs((h_hat[ell] <= a[ell] for ell in range(1, n_layers)))
        gurobi_model.addConstrs((h_hat[ell] >= z_hat[ell] for ell in range(1, n_layers)))
        gurobi_model.addConstrs((h_hat[ell] <= z_hat[ell] + (1 - a[ell]) for ell in range(1, n_layers)))
        gurobi_model.addConstrs((z_hat[ell] == z[ell]/M[ell] for ell in range(1, n_layers)))
        gurobi_model.addConstrs((h[ell] == M[ell]*h_hat[ell] for ell in range(1, n_layers)))
    else:
        gurobi_model.addConstrs((h[ell] <= M_plus[ell]*a[ell] for ell in range(1, n_layers)))
        gurobi_model.addConstrs((h[ell] >= z[ell] for ell in range(1, n_layers)))
        gurobi_model.addConstrs((h[ell] <= z[ell] - M_minus[ell]*(1 - a[ell]) for ell in range(1, n_layers)))

    # If lower and upper bounds have the same sign, can fix variables
    for ell in range(1, n_layers):
        active_units = M_minus[ell] > 0
        gurobi_model.addConstr(a[ell][active_units] == 1)
        gurobi_model.addConstr(h[ell][active_units] == z[ell][active_units])

        inactive_units = M_plus[ell] < 0
        gurobi_model.addConstr(a[ell][inactive_units] == 0)
        gurobi_model.addConstr(h[ell][inactive_units] == 0)

def compute_bounds(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    n_ambulances: int,
    facility_capacity: Optional[int] = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Find big-M values for embedding an MLP for ambulance location in a MIP model.

    Only use interval arithmetic to compute bounds.

    Parameters
    ----------
    weights, biases : list[np.ndarray]
        Weights and biases of the MLP.
    
    n_ambulances : int
        Total number of ambulances.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    Returns
    -------
    M_minus : list[np.ndarray]
        Lower bounds for the net input to each hidden unit.
    
    M_plus : list[np.ndarray]
        Upper bounds for the net input to each hidden unit.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    
    M_minus = []
    M_plus = []

    first_layer = True
    for weight, bias in zip(weights[:-1], biases[:-1]):
        # For the first hidden layer, leverage the following assumptions on the input x:
        # - x is nonnegative
        # - x sums to n_ambulances
        # - Each component of x is at most facility_capacity
        if first_layer:
            n_stations = weight.shape[1]
            # If n_ambulances = 7, facility_capacity = 3, n_stations = 5, then x = [3, 3, 1, 0, 0]
            x = np.zeros(n_stations)
            n_stations_full = n_ambulances // facility_capacity
            x[:n_stations_full] = facility_capacity
            if n_stations_full < n_stations:
                x[n_stations_full] = n_ambulances % facility_capacity
            weight_sorted = np.sort(weight, axis=1)
            M_minus_ell = np.minimum(weight_sorted, 0)@x + bias
            weight_sorted = np.fliplr(weight_sorted)
            M_plus_ell = np.maximum(weight_sorted, 0)@x + bias
            first_layer = False
        
        # For subsequent hidden layers, assume units can independently achieve zero or maximum activation
        else:
            max_activations = np.maximum(0, M_plus[-1])
            M_minus_ell = np.minimum(0, weight)@max_activations + bias
            M_plus_ell = np.maximum(0, weight)@max_activations + bias
        
        M_minus.append(M_minus_ell)
        M_plus.append(M_plus_ell)
    
    return M_minus, M_plus

def embed_modified_sigmoid(
    gurobi_model: gp.Model,
    input_vars: gp.MVar,
    output_vars: gp.MVar,
    tangent_points: Optional[np.ndarray] = None
):
    """Embed the modified sigmoid function applied at the last hidden
    layer in a Gurobi model.
    
    We model the function as a piecewise linear function. In general,
    modeling piecewise linear functions requires introducing additional
    binary variables, but because the function is concave and the MIP
    has to maximize it, it is sufficient to use only linear constraints.
    
    Parameters
    ----------
    gurobi_model : gp.Model
        Gurobi model.
    
    input_vars : gp.MVar
        Decision variables used as input to the function.
    
    output_vars : gp.MVar
        Decision variables used as output from the function.
    
    tangent_points : np.ndarray, optional
        Points where tangent lines are used to approximate the function.
    """
    gurobi_model.addConstr(output_vars <= 1)
    if tangent_points is None:
        tangent_points = np.linspace(0, 5, 11)
    s = lambda z: 1/(1 + np.exp(-z))
    ds = lambda z: s(z)*(1 - s(z))
    gurobi_model.addConstrs((
        output_vars <= ds(z)*input_vars + s(z) - ds(z)*z
        for z in tangent_points
    ))
