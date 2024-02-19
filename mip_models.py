import numpy as np
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.torch import add_sequential_constr
from ems_data import EMSData
from neural_network import MLP

def compute_coverage(data: EMSData):
    """Compute the `coverage` parameter for MEXCLP.

    Parameters
    ----------
    data : EMSData
        EMS data instance.
    
    Returns
    -------
    list[list[int]]
        `coverage` parameter for MEXCLP.
    """
    demand_nodes = data.demand_nodes[['x', 'y']].values
    stations = data.stations[['x', 'y']].values
    n_demand_nodes = demand_nodes.shape[0]
    n_stations = stations.shape[0]

    euc_dist = lambda p, q: np.linalg.norm(p - q)
    travel_time = lambda p, q: EMSData.travel_time(euc_dist(p, q)/1000)
    covered = lambda i, j: travel_time(stations[j], demand_nodes[i]) < data.response_time_threshold

    coverage = [[j for j in range(n_stations) if covered(i, j)] for i in range(n_demand_nodes)]

    return coverage

def mexclp(
        demand: list[float],
        coverage: list[list[int]],
        n_ambulances: int,
        busy_fraction: float,
        time_limit: float = None,
        verbose: bool = False
    ) -> list[int]:
    """Solve the MEXCLP model (Daskin, 1983).

    Parameters
    ----------
    demand : list[float]
        Weight of each demand node.
    
    coverage : list[list[int]]
        For each demand node, a list of facilities covering it.
    
    n_ambulances : int
        Total number of ambulances.
    
    busy_fraction : float
        Average fraction of time an ambulance is busy.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    list[int]
        Number of ambulances located at each facility.
    """
    n_demand_nodes = len(demand)
    n_facilities = max(max(c, default=0) for c in coverage) + 1  # default=0 in case c is empty

    model = gp.Model()
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = verbose

    x = model.addVars(n_facilities, lb=0, ub=n_ambulances, vtype=GRB.INTEGER)
    y = model.addVars(n_demand_nodes, n_ambulances, vtype=GRB.BINARY)

    model.setObjective(
        gp.quicksum(
            demand[i]*(1-busy_fraction)*busy_fraction**k * y[i, k]
            for i in range(n_demand_nodes) for k in range(n_ambulances)
        ),
        GRB.MAXIMIZE
    )

    model.addConstrs((x.sum(coverage[i]) >= y.sum(i, '*') for i in range(n_demand_nodes)))
    model.addConstr(x.sum() <= n_ambulances)

    model.optimize()

    return [int(x[j].X) for j in range(n_facilities)]

def mexclp_mlp(
        demand: list[float],
        mlp: MLP,
        n_ambulances: int,
        extra_constraints: tuple[np.ndarray, np.ndarray] = None,
        use_gurobi_ml: bool = False,
        scale_relu: bool = False,
        warm_start: list[int] = None,
        time_limit: float = None,
        verbose: bool = False,
    ) -> list[int]:
    """Solve the MEXCLP-MLP model.

    Parameters
    ----------
    demand : list[float]
        Weight of each demand node.
    
    mlp : MLP
        MLP predicting coverage probabilities given a solution.
    
    n_ambulances : int
        Total number of ambulances.
    
    extra_constraints : tuple(np.ndarray, np.ndarray), optional
        Extra constraints on x of the form Ax <= b. Given as a tuple (A, b).
    
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
    linear_layers = [layer for layer in mlp if isinstance(layer, nn.Linear)]
    weights = [None] + [layer.weight.detach().cpu().numpy() for layer in linear_layers]
    biases = [None] + [layer.bias.detach().cpu().numpy() for layer in linear_layers]
    n_layers = len(linear_layers)
    layer_dims = [weights[1].shape[1]] + [weight.shape[0] for weight in weights[1:]]

    model = gp.Model()
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = verbose

    x = model.addMVar(layer_dims[0], lb=0, ub=n_ambulances, vtype=GRB.INTEGER)
    y = model.addMVar(layer_dims[-1], lb=-GRB.INFINITY, ub=1.0, vtype=GRB.CONTINUOUS)

    demand = np.array(demand)
    model.setObjective(demand@y, GRB.MAXIMIZE)

    model.addConstr(x.sum() <= n_ambulances)

    if extra_constraints is not None:
        A, b = extra_constraints
        model.addConstr(A@x <= b)
    
    if use_gurobi_ml:
        # Ideally we would pass mlp directly to add_sequential_constr, but it doesn't like Dropout layers
        sequential_model = nn.Sequential(*(layer for layer in mlp if not isinstance(layer, nn.Dropout)))

        # z still needs to be a list[gurobipy.MVar] because we use z[-1] to model the output layer later on
        z = [model.addMVar(layer_dims[-1], lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)]

        add_sequential_constr(model, sequential_model, x, z[-1])
    else:
        M_minus, M_plus = calculate_bounds(weights, biases, n_ambulances)

        # Set z[0], a[0] to None to make indexing consistent with paper
        z = [None] + [model.addMVar(dim, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) for dim in layer_dims[1:]]
        a = [None] + [model.addMVar(dim, vtype=GRB.BINARY) for dim in layer_dims[1:-1]]
        h = [model.addMVar(dim, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) for dim in layer_dims[:-1]]

        model.addConstr(h[0] == x)
        model.addConstrs((z[ell] == weights[ell]@h[ell-1] + biases[ell] for ell in range(1, n_layers+1)))

        if scale_relu:
            M = [None] + [np.maximum(M_plus[ell], -M_minus[ell]) for ell in range(1, n_layers)]

            # Set z_hat[0], h_hat[0] to None to make indexing consistent with paper
            z_hat = [None] + [model.addMVar(dim, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS) for dim in layer_dims[1:-1]]
            h_hat = [None] + [model.addMVar(dim, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS) for dim in layer_dims[1:-1]]

            model.addConstrs((h_hat[ell] <= a[ell] for ell in range(1, n_layers)))
            model.addConstrs((h_hat[ell] >= z_hat[ell] for ell in range(1, n_layers)))
            model.addConstrs((h_hat[ell] <= z_hat[ell] + (1 - a[ell]) for ell in range(1, n_layers)))
            model.addConstrs((z_hat[ell] == z[ell]/M[ell] for ell in range(1, n_layers)))
            model.addConstrs((h[ell] == M[ell]*h_hat[ell] for ell in range(1, n_layers)))
        else:
            model.addConstrs((h[ell] <= M_plus[ell]*a[ell] for ell in range(1, n_layers)))
            model.addConstrs((h[ell] >= z[ell] for ell in range(1, n_layers)))
            model.addConstrs((h[ell] <= z[ell] - M_minus[ell]*(1 - a[ell]) for ell in range(1, n_layers)))

        # If lower and upper bounds have the same sign, can fix variables
        for ell in range(1, n_layers):
            active_units = M_minus[ell] > 0
            model.addConstr(a[ell][active_units] == 1)
            model.addConstr(h[ell][active_units] == z[ell][active_units])

            inactive_units = M_plus[ell] < 0
            model.addConstr(a[ell][inactive_units] == 0)
            model.addConstr(h[ell][inactive_units] == 0)
    
    sigmoid = lambda z: 1/(1 + np.exp(-z))
    sigmoid_derivative = lambda z: sigmoid(z)*(1 - sigmoid(z))
    tangent_points = np.linspace(0, 5, 11)
    model.addConstrs((y <= sigmoid_derivative(z0)*z[-1] + sigmoid(z0) - sigmoid_derivative(z0)*z0 for z0 in tangent_points))

    # Warm start
    if warm_start is not None:
        for j in range(layer_dims[0]):
            x[j].Start = warm_start[j]

    model.optimize()
    
    return [int(x[j].X) for j in range(layer_dims[0])]

def calculate_bounds(
        weights: list[np.ndarray | None],
        biases: list[np.ndarray | None],
        n_ambulances: int
    ) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Find big-M values for the MEXCLP-MLP model.

    Only use interval arithmetic to compute bounds.

    Parameters
    ----------
    weights : list[np.ndarray | None]
        Weights of each linear layer. weights[0] is None to make indexing consistent with paper.
    
    biases : list[np.ndarray | None]
        Biases of each linear layer. biases[0] is None to make indexing consistent with paper.
    
    n_ambulances : int
        Total number of ambulances.
    
    Returns
    -------
    M_minus : list[np.ndarray]
        Lower bounds for each hidden layer.
        - M_minus[0] is None to make indexing consistent with paper. 
        - M_minus[ell] contains the lower bounds for the net inputs to the units in layer ell.
    
    M_plus : list[np.ndarray]
        Upper bounds for each hidden layer.
        - M_plus[0] is None to make indexing consistent with paper.
        - M_plus[ell] contains the upper bounds for the net inputs to the units in layer ell.
    """
    M_minus = [None]
    M_plus = [None]

    n_layers = len(weights) - 1  # -1 because weights[0] is None
    for ell in range(1, n_layers):
        # For the first hidden layer, leverage the fact that input x is nonnegative and sums to n_ambulances
        if ell == 1:
            M_minus_ell = weights[1].min(axis=1)*n_ambulances + biases[1]
            M_plus_ell = weights[1].max(axis=1)*n_ambulances + biases[1]
        
        # For subsequent hidden layers, assume units can independently achieve zero or maximum activation
        else:
            max_activations = np.maximum(0, M_plus[ell-1])
            M_minus_ell = np.minimum(0, weights[ell])@max_activations + biases[ell]
            M_plus_ell = np.maximum(0, weights[ell])@max_activations + biases[ell]
        
        M_minus.append(M_minus_ell)
        M_plus.append(M_plus_ell)
    
    return M_minus, M_plus
