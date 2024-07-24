from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ems_data import *
from simulation import Simulation
    
def plot_locations(
    ems_data: EMSData,
    demand_nodes: np.ndarray = None,
    ax : Optional[matplotlib.axes.Axes] = None
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot patient, station, and hospital locations for a region.

    Parameters
    ----------
    ems_data : EMSData
        EMSData instance set to the desired region.
    
    demand_nodes : np.ndarray of shape (n_demand_nodes, 2), optional
        Demand nodes' UTM coordinates (easting, northing).
    
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, create new figure and axes.
    
    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes objects for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()
    ax.scatter(
        ems_data.patients[:, 0], ems_data.patients[:, 1],
        color='salmon', alpha=0.2, marker=',', label="Patient"
    )
    if demand_nodes is not None:
        ax.scatter(
            demand_nodes[:, 0], demand_nodes[:, 1],
            color='black', marker='1', s=100, label="Demand Node", alpha=0.5
        )
    ax.scatter(
        ems_data.hospitals[:, 0], ems_data.hospitals[:, 1],
        color='red', marker='P', edgecolors='black', s=200, label="Hospital"
    )
    ax.scatter(
        ems_data.stations[:, 0], ems_data.stations[:, 1],
        color='white', edgecolors='black', s=200, label="Station"
    )
    for i in range(ems_data.stations.shape[0]):
        ax.text(
            ems_data.stations[i, 0], ems_data.stations[i, 1],
            str(i), color='black', ha='center', va='center', size=8
        )
    ax.set_title(REGION_ID_TO_NAME[ems_data.region_id])
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.legend()
    return fig, ax

def plot_arrival_times(
    arrival_times: np.ndarray,
    urbanicity: Optional[str] = None,
    bin_width: int = 60,
    tick_interval: int = 240,
    ax: Optional[matplotlib.axes.Axes] = None
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot distribution of historical arrival times.

    Parameters
    ----------
    arrival_times : np.ndarray of shape (n_arrival_times,)
        Historical arrival times in minutes past midnight.
    
    urbanicity : str, optional
        Urbanicity of the region.
        
        Used in the plot title if provided.
    
    bin_width : int, optional
        Width of each bin in minutes.

        Prohibit values that are not factors of 1440 (minutes in a day).
    
    tick_interval : int, optional
        Interval between x-axis ticks in minutes.

        Prohibit values that are not factors of 1440 (minutes in a day).
    
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, create new figure and axes.
    
    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes objects for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()
    if 1440 % bin_width:
        raise ValueError("bin_width must be a factor of 1440")
    if 1440 % tick_interval:
        raise ValueError("tick_interval must be a factor of 1440")
    ax.hist(
        arrival_times,
        bins=range(0, 1441, bin_width),
        weights=np.ones_like(arrival_times)/len(arrival_times),
        edgecolor='black',
        linewidth=0.5
    )
    ticks = range(0, 1441, tick_interval)
    labels = [f"{t//60:02}:{t%60:02}" for t in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_title(f"{urbanicity.capitalize()} Arrival Time Distribution")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Proportion")
    return fig, ax

def plot_support_times(
    support_times: np.ndarray,
    urbanicity: Optional[str] = None,
    bin_width: int = 1,
    percentile: float = 99,
    ax: Optional[matplotlib.axes.Axes] = None
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot distribution of historical support times.
    
    Parameters
    ----------
    support_times : np.ndarray of shape (n_support_times,)
        Historical support times in minutes.
    
    urbanicity : str, optional
        Urbanicity of the region.
        
        Used in the plot title if provided.
    
    bin_width : int, optional
        Width of each bin in minutes.
    
    percentile : float, optional
        Maximum percentile of support times to plot.
    
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, create new figure and axes.
    
    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes objects for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()
    threshold = np.percentile(support_times, percentile)
    support_times = support_times[support_times <= threshold]
    ax.hist(
        support_times,
        bins=range(0, int(threshold) + bin_width + 1, bin_width),
        weights=np.ones_like(support_times)/len(support_times)
    )
    ax.set_title(f"{urbanicity.capitalize()} Support Time Distribution")
    ax.set_xlabel("Support Time (minutes)")
    ax.set_ylabel("Proportion")
    return fig, ax

def plot_response_times(
    sim: Simulation,
    n_dispatches: Optional[int] = None,
    ax: Optional[matplotlib.axes.Axes] = None
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot response times throughout the last simulation run."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    y = sim._response_times[:n_dispatches]
    x = range(len(y))
    ax.plot(x, y)
    ax.set_title("Response Times Throughout Simulation")
    ax.set_xlabel("Dispatch")
    ax.set_ylabel("Response Time (minutes)")
    return fig, ax

def plot_metric_by_ambulance_count(
    X: np.ndarray,
    y: np.ndarray,
    ylabel: str,
    showfliers: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None
):
    """For each total ambulance count, plot a box plot of the metric for
    the corresponding samples.
    
    Metrics generally improve as ambulance count increases. If this
    trend is hard to see due to outliers, then outliers should be
    removed prior to training the neural network.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    n_ambulances = X.sum(axis=1)
    min_ambulances = n_ambulances.min()
    max_ambulances = n_ambulances.max()
    data_to_plot = [
        y[n_ambulances == n]
        for n in range(min_ambulances, max_ambulances+1)
    ]
    ax.boxplot(
        data_to_plot,
        positions=range(min_ambulances, max_ambulances+1),
        widths=0.6,
        showfliers=showfliers
    )
    ax.set_xlabel("Total ambulances")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(min_ambulances, max_ambulances+1))
    ax.grid(True)
    return fig, ax
