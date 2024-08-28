import math
from typing import Optional, Any
import numpy as np
from scipy import optimize
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.sklearn import add_decision_tree_regressor_constr
from gurobi_ml.lightgbm import add_lgbm_booster_constr
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
) -> tuple[np.ndarray, gp.Model]:
    """Solve the MEXCLP model.

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
    solution : np.ndarray of shape (n_facilities,)
        Number of ambulances located at each facility.
    
    model : gp.Model
        Gurobi model.
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

    return np.array([int(x[j].X + 0.5) for j in range(n_facilities)]), model

def pmedian_with_queueing(
    n_ambulances: int,
    distance: np.ndarray,
    arrival_rate: float,
    service_rate: float,
    success_prob: float,
    demand: Optional[np.ndarray] = None,
    facility_capacity: Optional[int] = None,
    time_limit: Optional[float] = None,
    verbose: bool = False
) -> tuple[np.ndarray | None, gp.Model]:
    """Solve a model that combines the p-median model with queueing
    constraints.

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
    solution : np.ndarray of shape (n_facilities,) | None
        Number of ambulances located at each facility. If the model is
        infeasible, return None.
    
    model : gp.Model
        Gurobi model.
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
    return solution, model

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

def linear_based_model(
    n_ambulances: int,
    optimization_sense: int,
    coef: np.ndarray,
    intercept: float,
    facility_capacity: int,
    time_limit: Optional[float] = None,
    verbose: bool = False
) -> tuple[np.ndarray, gp.Model]:
    """Solve an ML-based ambulance location model which embeds a linear model.

    Parameters
    ----------
    n_ambulances : int
        Total number of ambulances.
    
    optimization_sense : {GRB.MINIMIZE, GRB.MAXIMIZE}
        Whether to solve a minimization or maximization problem.
    
    coef : np.ndarray of shape (n_facilities*facility_capacity,)
        Coefficients of the linear model.
    
    intercept : float
        Intercept of the linear model.
    
    facility_capacity : int
        Maximum number of ambulances allowed at any facility.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    solution : np.ndarray of shape (n_facilities,)
        Number of ambulances located at each facility.
    
    model : gp.Model
        Gurobi model.
    """
    n_facilities = coef.shape[0] // facility_capacity

    model = gp.Model()
    model.Params.LogToConsole = verbose
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    x = model.addMVar((n_facilities, facility_capacity), vtype=GRB.BINARY)
    y = model.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    model.setObjective(y, optimization_sense)
    model.addConstr(x.sum() <= n_ambulances)
    model.addConstr(y == coef@x.reshape(-1) + intercept)
    model.addConstrs((x[j, k] <= x[j, k-1] for j in range(n_facilities) for k in range(1, facility_capacity)))
    model.optimize()
    
    solution = np.rint(x.X.sum(axis=1)).astype(int)
    return solution, model

def dt_based_model(
    n_ambulances: int,
    optimization_sense: int,
    dt: DecisionTreeRegressor,
    facility_capacity: Optional[int] = None,
    use_gurobi_ml: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False
) -> tuple[np.ndarray, gp.Model]:
    """Solve an ML-based ambulance location model which embeds a decision tree regressor.

    Parameters
    ----------
    n_ambulances : int
        Total number of ambulances.
    
    optimization_sense : {GRB.MINIMIZE, GRB.MAXIMIZE}
        Whether to solve a minimization or maximization problem.
    
    dt : DecisionTreeRegressor
        Decision tree regressor to embed.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    use_gurobi_ml : bool, optional
        Use the Gurobi Machine Learning package to embed the ML model.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    solution : np.ndarray of shape (n_facilities,)
        Number of ambulances located at each facility.
    
    model : gp.Model
        Gurobi model.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    n_facilities = dt.n_features_in_

    model = gp.Model()
    model.Params.LogToConsole = verbose
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    x = model.addMVar(n_facilities, lb=0, ub=facility_capacity, vtype=GRB.INTEGER)
    y = model.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    model.setObjective(y, optimization_sense)
    model.addConstr(x.sum() <= n_ambulances)
    if use_gurobi_ml:
        # Gurobi ML does not automatically round thresholds down and set epsilon to 1 for integer input_vars
        branch_nodes = (dt.tree_.children_left > 0).nonzero()[0]
        old_thresholds = dt.tree_.threshold[branch_nodes]
        dt.tree_.threshold[branch_nodes] = np.floor(old_thresholds)
        add_decision_tree_regressor_constr(model, dt, x, y, epsilon=1)
        dt.tree_.threshold[branch_nodes] = old_thresholds
    else:
        embed_decision_tree(model, dt, x, y)
    model.optimize()
    
    return np.rint(x.X).astype(int), model

def gbm_based_model(
    n_ambulances: int,
    optimization_sense: int,
    gbm: lgb.Booster,
    facility_capacity: Optional[int] = None,
    use_gurobi_ml: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False
) -> tuple[np.ndarray, gp.Model]:
    """Solve an ML-based ambulance location model which embeds a LightGBM model.

    Parameters
    ----------
    n_ambulances : int
        Total number of ambulances.
    
    optimization_sense : {GRB.MINIMIZE, GRB.MAXIMIZE}
        Whether to solve a minimization or maximization problem.
    
    gbm : lgb.Booster
        LightGBM model to embed.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    use_gurobi_ml : bool, optional
        Use the Gurobi Machine Learning package to embed the ML model.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    solution : np.ndarray of shape (n_facilities,)
        Number of ambulances located at each facility.
    
    model : gp.Model
        Gurobi model.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    n_facilities = gbm.num_feature()

    model = gp.Model()
    model.Params.LogToConsole = verbose
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    x = model.addMVar(n_facilities, lb=0, ub=facility_capacity, vtype=GRB.INTEGER)
    y = model.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    model.setObjective(y, optimization_sense)
    model.addConstr(x.sum() <= n_ambulances)
    if use_gurobi_ml:
        add_lgbm_booster_constr(model, gbm, x, y)
    else:
        embed_lightgbm_model(model, gbm, x, y)
    model.optimize()
    
    return np.rint(x.X).astype(int), model

def mlp_based_model(
    n_ambulances: int,
    optimization_sense: int,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    facility_capacity: Optional[int] = None,
    use_gurobi_ml: bool = False,
    time_limit: Optional[float] = None,
    verbose: bool = False
) -> tuple[np.ndarray, gp.Model]:
    """Solve an ML-based ambulance location model which embeds a multilayer perceptron.

    Parameters
    ----------
    n_ambulances : int
        Total number of ambulances.
    
    optimization_sense : {GRB.MINIMIZE, GRB.MAXIMIZE}
        Whether to solve a minimization or maximization problem.
    
    weights, biases : list[np.ndarray]
        Weights and biases of the MLP to embed.
    
    facility_capacity : int, optional
        Maximum number of ambulances allowed at any facility.
    
    use_gurobi_ml : bool, optional
        Use the Gurobi Machine Learning package to embed the ML model.
    
    time_limit : float, optional
        Time limit in seconds.
    
    verbose : bool, optional
        Print Gurobi output.
    
    Returns
    -------
    solution : np.ndarray of shape (n_facilities,)
        Number of ambulances located at each facility.
    
    model : gp.Model
        Gurobi model.
    """
    if facility_capacity is None:
        facility_capacity = n_ambulances
    n_facilities = weights[0].shape[1]

    model = gp.Model()
    model.Params.LogToConsole = verbose
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    x = model.addMVar(n_facilities, lb=0, ub=facility_capacity, vtype=GRB.INTEGER)
    y = model.addMVar(1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    model.setObjective(y, optimization_sense)
    model.addConstr(x.sum() <= n_ambulances)
    if use_gurobi_ml:
        # To be compatible with Gurobi ML package, need to convert to a sequential model
        sequential_layers = []
        for weight, bias in zip(weights, biases):
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                linear.weight.copy_(torch.tensor(weight))
                linear.bias.copy_(torch.tensor(bias))
            sequential_layers.extend([linear, nn.ReLU()])
        sequential_layers.pop()  # Remove last ReLU
        sequential_model = nn.Sequential(*sequential_layers)
        add_sequential_constr(model, sequential_model, x, y)
    else:
        M_minus, M_plus = compute_mlp_bounds(weights, biases, n_ambulances, facility_capacity)
        embed_multilayer_perceptron(model, weights, biases, x, y, M_minus, M_plus)
    model.optimize()
    
    return np.rint(x.X).astype(int), model

def embed_decision_tree(
    gurobi_model: gp.Model,
    decision_tree: DecisionTreeRegressor | dict[str, np.ndarray],
    input_vars: gp.MVar,
    output_vars: gp.MVar,
    epsilon: float = 0.01
):
    """Embed a decision tree regressor in a Gurobi model.

    Parameters
    ----------
    gurobi_model : gp.Model
        Gurobi model.
    
    decision_tree : DecisionTreeRegressor | dict[str, np.ndarray]
        Decision tree.

        If a dictionary, must contain the following key-value pairs:
        - 'children_left': np.ndarray of shape (n_nodes,)
        - 'children_right': np.ndarray of shape (n_nodes,)
        - 'feature': np.ndarray of shape (n_nodes,)
        - 'threshold': np.ndarray of shape (n_nodes,)
        - 'value': np.ndarray of shape (n_nodes,), (n_nodes, n_outputs), or (n_nodes, n_outputs, 1)

    input_vars : gp.MVar of shape (n_features,)
        Decision variables used as input to the decision tree regressor.

        Ensure LB and UB are finite as big-M constraints depend on these bounds.
    
    output_vars : gp.MVar of shape (n_outputs,)
        Decision variables used as output from the decision tree regressor.
    
    epsilon : float, optional
        Positive constant to model splits involving continuous features.
    """
    # Update needed before accessing variable attributes
    gurobi_model.update()
    lb = input_vars.lb
    ub = input_vars.ub
    if not (np.isfinite(lb).all() and np.isfinite(ub).all()):
        raise ValueError("Bounds must be finite")
    if isinstance(decision_tree, DecisionTreeRegressor):
        tree = decision_tree.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value
    else:
        children_left = decision_tree['children_left']
        children_right = decision_tree['children_right']
        feature = decision_tree['feature']
        threshold = decision_tree['threshold']
        value = decision_tree['value']
    if value.ndim == 3:
        value = value[:, :, 0]
    n_nodes = children_left.shape[0]
    branch_nodes = (children_left > 0).nonzero()[0]
    leaf_nodes = (children_left < 0).nonzero()[0]
    # For branch nodes testing integer features, round thresholds down and set corresponding epsilon to 1
    tests_integer_feature = np.isin(input_vars.vtype[feature[branch_nodes]], [GRB.BINARY, GRB.INTEGER])
    threshold = threshold[branch_nodes]
    threshold = np.where(tests_integer_feature, np.floor(threshold), threshold)
    epsilon = np.where(tests_integer_feature, 1, epsilon)

    w = gurobi_model.addMVar(n_nodes, lb=0.0, ub=1.0)
    w[leaf_nodes[1:]].vtype = GRB.BINARY
    
    w[0].lb = 1
    gurobi_model.addConstr(
        w[branch_nodes] == w[children_left[branch_nodes]] + w[children_right[branch_nodes]]
    )
    gurobi_model.addConstr(
        input_vars[feature[branch_nodes]] <= ub[feature[branch_nodes]]
        - (ub[feature[branch_nodes]] - threshold)*w[children_left[branch_nodes]]
    )
    gurobi_model.addConstr(
        input_vars[feature[branch_nodes]] >= lb[feature[branch_nodes]]
        + (threshold + epsilon - lb[feature[branch_nodes]])*w[children_right[branch_nodes]]
    )
    gurobi_model.addConstr(
        output_vars == value[leaf_nodes].T@w[leaf_nodes]
    )
    # Future work: fix w[t].ub = 0 for unreachable nodes t, or prune dead branches to get a smaller tree

def embed_lightgbm_model(
    gurobi_model: gp.Model,
    gbm: lgb.Booster,
    input_vars: gp.MVar,
    output_vars: gp.MVar,
    epsilon: float = 0.01
):
    """Embed a LightGBM model in a Gurobi model.

    Parameters
    ----------
    gurobi_model : gp.Model
        Gurobi model.
    
    gbm : lgb.Booster
        LightGBM model.
    
    input_vars : gp.MVar of shape (n_features,)
        Decision variables used as input to the gradient boosting machine.
    
    output_var : gp.MVar of shape (1,)
        Decision variable used as output from the gradient boosting machine.
    
    epsilon : float, optional
        Positive constant to model strict inequalities.
    """
    n_trees = gbm.num_trees()
    y = gurobi_model.addMVar(n_trees, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    for i in range(n_trees):
        lightgbm_tree = gbm.dump_model()['tree_info'][i]['tree_structure']
        dt = flatten_lightgbm_tree(lightgbm_tree)
        embed_decision_tree(gurobi_model, dt, input_vars, y[i], epsilon)
    gurobi_model.addConstr(output_vars == y.sum())

def embed_multilayer_perceptron(
    gurobi_model: gp.Model,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    input_vars: gp.MVar,
    output_vars: gp.MVar,
    M_minus: list[np.ndarray],
    M_plus: list[np.ndarray]
):
    """Embed an MLP in a Gurobi model.

    Assumes the MLP uses the ReLU activation function for hidden layers,
    and no activation function for the output layer.

    Parameters
    ----------
    gurobi_model : gp.Model
        Gurobi model.
    
    weights, biases : list[np.ndarray]
        Weights and biases of the MLP.
    
    input_vars : gp.MVar of shape (n_features,)
        Decision variables used as input to the MLP.
    
    output_vars : gp.MVar of shape (n_outputs,)
        Decision variables used as output from the MLP.
    
    M_minus : list[np.ndarray]
        Lower bounds for the net input to each hidden unit.
    
    M_plus : list[np.ndarray]
        Upper bounds for the net input to each hidden unit.
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

def compute_mlp_bounds(
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

def flatten_lightgbm_tree(tree: dict[str, Any], dummy_value: int = -1) -> dict[str, np.ndarray]:
    """Convert a LightGBM tree to a flat representation similar to scikit-learn's.

    Parameters
    ----------
    tree : dict[str, Any]
        A tree structure from a LightGBM model.

        For example, if gbm is an instance of lgb.Booster, then tree i can be accessed as
        gbm.dump_model()['tree_info'][i]['tree_structure'].

    dummy_value : int, optional
        Dummy value to pad arrays with, as arrays pertain only to either branch or leaf nodes.

    Returns
    -------
    dict[str, np.ndarray]
        Contains the following arrays:
        - children_left
        - children_right
        - feature
        - threshold
        - value
    """
    children_left = []
    children_right = []
    feature = []
    threshold = []
    value = []
    
    def recurse(subtree: dict[str, Any]) -> int:
        current_node_index = len(feature)
        if 'split_index' in subtree:
            if subtree['decision_type'] != '<=':
                raise ValueError(f"'decision_type' was {subtree['decision_type']}, expected '<='")
            feature.append(subtree['split_feature'])
            threshold.append(subtree['threshold'])
            value.append(dummy_value)
            # Have to append dummy values to overwrite later because of recursion
            children_left.append(0)
            children_right.append(0)
            children_left[current_node_index] = recurse(subtree['left_child'])
            children_right[current_node_index] = recurse(subtree['right_child'])
        else:
            children_left.append(dummy_value)
            children_right.append(dummy_value)
            feature.append(dummy_value)
            threshold.append(dummy_value)
            value.append(subtree['leaf_value'])
        return current_node_index
    
    recurse(tree)
    
    return {
        'children_left': np.array(children_left),
        'children_right': np.array(children_right),
        'feature': np.array(feature),
        'threshold': np.array(threshold),
        'value': np.array(value)
    }
