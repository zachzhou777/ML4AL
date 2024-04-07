import math
from typing import Optional
import numpy as np
from scipy import optimize
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.torch import add_sequential_constr
from ems_data import EMSData
from neural_network import MLP

def mexclp(
        demand: np.ndarray,
        distance: np.ndarray,
        threshold: float,
        n_ambulances: int,
        busy_fraction: float,
        facility_capacity: Optional[int] = None,
        time_limit: float = None,
        verbose: bool = False
    ) -> list[int]:
    """Solve the MEXCLP model (Daskin, 1983).

    Parameters
    ----------
    demand : np.ndarray of shape (n_demand_nodes,)
        Demand weights.
    
    distance : np.ndarray of shape (n_demand_nodes, n_facilities)
        Distance matrix.
    
    threshold : float
        Maximum distance for a facility to cover a demand node.
    
    n_ambulances : int
        Total number of ambulances.
    
    busy_fraction : float
        Average fraction of time an ambulance is busy.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    list[int]
        Number of ambulances located at each facility.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    
    n_demand_nodes, n_facilities = distance.shape

    model = gp.Model()
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = verbose

    x = model.addVars(n_facilities, lb=0, ub=facility_capacity, vtype=GRB.INTEGER)
    y = model.addVars(n_demand_nodes, n_ambulances, vtype=GRB.BINARY)

    demand = np.array(demand, dtype=float)
    demand /= demand.sum()  # Normalize so objective function is expected distance
    model.setObjective(
        gp.quicksum(
            demand[i]*(1-busy_fraction)*busy_fraction**k * y[i, k]
            for i in range(n_demand_nodes) for k in range(n_ambulances)
        ),
        GRB.MAXIMIZE
    )

    model.addConstrs((
        x.sum(j for j in range(n_facilities) if distance[i, j] < threshold) >= y.sum(i, '*')
        for i in range(n_demand_nodes)
    ))
    model.addConstr(x.sum() <= n_ambulances)

    model.optimize()

    return [int(x[j].X) for j in range(n_facilities)]

def compute_rho(max_servers: int = 100, success_prob: float = 0.95) -> list[float]:
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
    list[float]
        rho[0] is 0. For k > 0, rho[k] is the value of rho for k servers.
    """
    # Function used by scipy.optimize.root_scalar
    def rho_eq(rho, n_servers):
        lhs = ((rho**n_servers)/math.factorial(n_servers)) / sum((rho**k)/math.factorial(k) for k in range(n_servers+1))
        rhs = 1 - success_prob
        return lhs - rhs
    
    rho = [0]
    for n_servers in range(1, max_servers+1):
        eq = lambda rho: rho_eq(rho, n_servers=n_servers)
        result = optimize.root_scalar(eq, bracket=[0.0, 2*n_servers])  # 2*servers is usually enough, will raise a ValueError if not
        rho.append(result.root)
    
    return rho

def pmedian_with_queuing(
        demand: list[float],
        distance: np.ndarray,
        n_ambulances: int,
        arrival_rate: float,
        service_rate: float,
        rho: list[float],
        time_limit: float = None,
        verbose: bool = False
    ) -> list[int]:
    """Solve a model that combines the p-median model with queuing constraints.

    Similar to Boutilier and Chan (2022).

    Parameters
    ----------
    demand : np.ndarray of shape (n_demand_nodes,)
        Demand weights.
    
    distance : np.ndarray of shape (n_demand_nodes, n_facilities)
        Distance matrix.
    
    n_ambulances : int
        Total number of ambulances.
    
    arrival_rate : float
        System-wide arrival rate. Units are unimportant, only needs to be on the same scale as service_rate.
    
    service_rate : float
        Service rate of each server. Units are unimportant, only needs to be on the same scale as arrival_rate.
    
    rho : list[float]
        Rho values for queuing constraints. rho[0] should be 0. For k > 0, rho[k] should be the value of rho for k servers. Facility capacity is len(rho) - 1.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    x : np.ndarray of shape (n_facilities,)
        Number of servers located at each facility.
    
    y : np.ndarray of shape (n_demand_nodes, n_facilities)
        y[i, j] is the fraction of demand at node i that is served by facility j.
    """
    n_demand_nodes, n_facilities = distance.shape
    facility_capacity = len(rho) - 1  # Subtract 1 because rho[0] is 0

    model = gp.Model()
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = verbose

    demand = np.array(demand, dtype=float)
    demand /= demand.sum()  # Normalize so objective function is expected distance
    arrival_rates = arrival_rate * demand  # Note that demand is normalized

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

    return (np.array([int(x.sum(j, '*').getValue()) for j in range(n_facilities)]),
            np.array([[y[i, j].X for j in range(n_facilities)] for i in range(n_demand_nodes)]))

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

    TODO: Move call to calculate_bounds() inside this function, remove M_minus, M_plus arguments?

    Parameters
    ----------
    gurobi_model : gp.Model
        Gurobi model.
    
    weights : list[np.ndarray]
        Weight matrices of the MLP to be embedded.
    
    biases : list[np.ndarray]
        Bias vectors of the MLP to be embedded.
    
    input_vars : gp.MVar
        Decision variables used as input to the MLP.
    
    output_vars : gp.MVar
        Decision variables used as output from the MLP.
    
    M_minus : list[np.ndarray]
        Lower bounds for the net input to each hidden unit.
    
    M_plus : list[np.ndarray]
        Upper bounds for the net input to each hidden unit.
    
    scale_relu : bool, optional
        How to model the ReLU activation function in the constraints.
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

def calculate_bounds(
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        n_ambulances: int,
        facility_capacity: Optional[int] = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Find big-M values for embedding an MLP for ambulance location in a MIP model.

    Only use interval arithmetic to compute bounds.

    TODO: Use facility_capacity to get tighter bounds

    Parameters
    ----------
    weights : list[np.ndarray]
        Weight matrices of each linear layer.
    
    biases : list[np.ndarray]
        Bias vectors of each linear layer.
    
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

    first_iter = True
    for weight, bias in zip(weights[:-1], biases[:-1]):
        # For the first hidden layer, leverage the following assumptions on the input x:
        # - x is nonnegative
        # - x sums to n_ambulances
        # - TODO: Each component of x is at most facility_capacity
        if first_iter:
            M_minus_ell = weight.min(axis=1)*n_ambulances + bias
            M_plus_ell = weight.max(axis=1)*n_ambulances + bias
            first_iter = False
        
        # For subsequent hidden layers, assume units can independently achieve zero or maximum activation
        else:
            max_activations = np.maximum(0, M_plus[-1])
            M_minus_ell = np.minimum(0, weight)@max_activations + bias
            M_plus_ell = np.maximum(0, weight)@max_activations + bias
        
        M_minus.append(M_minus_ell)
        M_plus.append(M_plus_ell)
    
    return M_minus, M_plus

def embed_sigmoid(
        gurobi_model: gp.Model,
        input_vars: gp.MVar,
        output_vars: gp.MVar,
        tangent_points: Optional[np.ndarray] = None
    ):
    """Embed the sigmoid function in a Gurobi model.

    Only suitable under the following conditions:
    - Must be the output layer.
    - Optimization model must be a maximization problem.
    - Output probabilities are generally expected to be greater than 0.5.

    Parameters
    ----------
    gurobi_model : gp.Model
        Gurobi model.
    
    input_vars : gp.MVar
        Decision variables used as input to the sigmoid.
    
    output_vars : gp.MVar
        Decision variables used as output from the sigmoid.
    
    tangent_points : np.ndarray, optional
        Points where tangent lines are used to approximate the sigmoid function.
    """
    if tangent_points is None:
        tangent_points = np.linspace(0, 5, 11)
    sigmoid = lambda z: 1/(1 + np.exp(-z))
    sigmoid_derivative = lambda z: sigmoid(z)*(1 - sigmoid(z))
    gurobi_model.addConstrs((output_vars <= sigmoid_derivative(z0)*input_vars + sigmoid(z0) - sigmoid_derivative(z0)*z0 for z0 in tangent_points))

def coverage_mlp(
        demand: list[float],
        mlp: MLP,
        n_ambulances: int,
        facility_capacity: Optional[int] = None,
        use_gurobi_ml: bool = False,
        scale_relu: bool = False,
        warm_start: list[int] = None,
        time_limit: float = None,
        verbose: bool = False,
    ) -> list[int]:
    """Solve the Coverage-MLP model.

    Parameters
    ----------
    demand : list[float]
        Weight of each demand node.
    
    mlp : MLP
        MLP predicting coverage probabilities given a solution.
    
    n_ambulances : int
        Total number of ambulances.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    use_gurobi_ml : bool, optional
        Use the Gurobi Machine Learning package to embed the MLP in the MIP model.
    
    scale_relu : bool, optional
        How to model the ReLU activation function in the constraints. Ignored if use_gurobi_ml is True.
        - If False, use the original approach seen in other papers.
        - If True, scale the net inputs to ReLU units so that they are in [-1, 1], then rescale the outputs.
    
    warm_start : list[int], optional
        Initial solution.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    list[int]
        Number of ambulances located at each facility.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    
    linear_layers = [layer for layer in mlp if isinstance(layer, nn.Linear)]
    weights = [layer.weight.detach().cpu().numpy() for layer in linear_layers]
    biases = [layer.bias.detach().cpu().numpy() for layer in linear_layers]

    model = gp.Model()
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = verbose

    n_demand_nodes = len(demand)
    n_facilities = mlp[0].in_features

    x = model.addMVar(n_facilities, lb=0, ub=facility_capacity, vtype=GRB.INTEGER)
    z = model.addMVar(n_demand_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    y = model.addMVar(n_demand_nodes, lb=-GRB.INFINITY, ub=1.0, vtype=GRB.CONTINUOUS)

    # Demand-weighted coverage
    demand = np.array(demand, dtype=float)
    demand /= demand.sum()  # Normalize so objective function is expected distance
    model.setObjective(demand@y, GRB.MAXIMIZE)

    # Limit total number of ambulances
    model.addConstr(x.sum() <= n_ambulances)
    
    # Embed the MLP
    if use_gurobi_ml:
        # Ideally we would pass mlp directly to add_sequential_constr(), but it doesn't like Dropout layers
        sequential_model = nn.Sequential(*(layer for layer in mlp if not isinstance(layer, nn.Dropout)))
        add_sequential_constr(model, sequential_model, x, z)
    else:
        M_minus, M_plus = calculate_bounds(weights, biases, n_ambulances)
        embed_mlp(model, weights, biases, x, z, M_minus, M_plus, scale_relu=scale_relu)
    embed_sigmoid(model, z, y)

    # Warm start
    if warm_start is not None:
        for j in range(n_facilities):
            x[j].Start = warm_start[j]

    model.optimize()
    
    return [int(x[j].X) for j in range(n_facilities)]

def median_mlp(
        demand: list[float],
        mlp: MLP,
        n_ambulances: int,
        facility_capacity: Optional[int] = None,
        use_gurobi_ml: bool = False,
        scale_relu: bool = False,
        warm_start: list[int] = None,
        time_limit: float = None,
        verbose: bool = False,
    ) -> list[int]:
    """Solve the Median-MLP model.

    Parameters
    ----------
    demand : list[float]
        Weight of each demand node.
    
    mlp : MLP
        MLP predicting coverage probabilities given a solution.
    
    n_ambulances : int
        Total number of ambulances.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    use_gurobi_ml : bool, optional
        Use the Gurobi Machine Learning package to embed the MLP in the MIP model.
    
    scale_relu : bool, optional
        How to model the ReLU activation function in the constraints. Ignored if use_gurobi_ml is True.
        - If False, use the original approach seen in other papers.
        - If True, scale the net inputs to ReLU units so that they are in [-1, 1], then rescale the outputs.
    
    warm_start : list[int], optional
        Initial solution.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    list[int]
        Number of ambulances located at each facility.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    
    linear_layers = [layer for layer in mlp if isinstance(layer, nn.Linear)]
    weights = [layer.weight.detach().cpu().numpy() for layer in linear_layers]
    biases = [layer.bias.detach().cpu().numpy() for layer in linear_layers]

    model = gp.Model()
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = verbose

    n_demand_nodes = len(demand)
    n_facilities = mlp[0].in_features

    x = model.addMVar(n_facilities, lb=0, ub=facility_capacity, vtype=GRB.INTEGER)
    y_mlp = model.addMVar(n_demand_nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    y = model.addMVar(n_demand_nodes, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)

    # Demand-weighted average distance
    demand = np.array(demand, dtype=float)
    demand /= demand.sum()  # Normalize so objective function is expected distance
    model.setObjective(demand@y, GRB.MINIMIZE)

    # Limit total number of ambulances
    model.addConstr(x.sum() <= n_ambulances)
    
    # Embed the MLP
    if use_gurobi_ml:
        # Ideally we would pass mlp directly to add_sequential_constr(), but it doesn't like Dropout layers
        sequential_model = nn.Sequential(*(layer for layer in mlp if not isinstance(layer, nn.Dropout)))
        add_sequential_constr(model, sequential_model, x, y_mlp)
    else:
        M_minus, M_plus = calculate_bounds(weights, biases, n_ambulances)
        embed_mlp(model, weights, biases, x, y_mlp, M_minus, M_plus, scale_relu=scale_relu)
    model.addConstr(y >= y_mlp)

    # Warm start
    if warm_start is not None:
        for j in range(n_facilities):
            x[j].Start = warm_start[j]

    model.optimize()
    
    return [int(x[j].X) for j in range(n_facilities)]
