import math
import heapq
import random
import pickle
import collections
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ems_data import EMSData

# Event.type values
ARRIVAL = 0
RETURN = 1

class Event:
    """An event in the simulation. Can represent either an arrival or an ambulance return."""
    def __init__(self, type: int, time: float, id: int):
        self.type = type  # Either ARRIVAL or RETURN
        self.time = time  # Time of event
        self.id = id  # Patient ID or station ID
    
    def __lt__(self, other: 'Event') -> bool:
        """Needed to determine next event in priority queue."""
        return self.time < other.time

class Simulation:
    """Discrete-event simulation of an EMS system.

    Parameters
    ----------
    data : EMSData
        Data for the simulation.
    
    avg_calls_per_day : int
        Average number of calls per day, a.k.a. the arrival rate.
    
    remove_outliers : bool, optional
        Whether to remove outliers when sampling travel times and support times.
    
    n_days : int, optional
        Duration of a run.
    
    n_replications : int, optional
        Number of times to replicate simulation for a given solution.
    
    Attributes
    ----------
    response_time_threshold : float
        Response time threshold in minutes.
    
    patients : pd.DataFrame
        Historical patient location data. Includes columns 'demand_node' and 'hospital'.
    
    n_demand_nodes : int
        Number of demand nodes.
    
    support_times : list[float]
        Historical support (scene, transfer, and turnover) times in minutes. Outliers (top 1%) are removed if remove_outliers is True.
    
    arrival_times : list[int]
        Historical arrival times, given as the exact minute of the day (e.g., 60*8 + 30 for 8:30 AM).
    
    median_response_times : np.ndarray of shape (n_stations, n_patients)
        median_response_times[i, j] is the median travel time in minutes from station i to patient j.
    
    median_transport_times : np.ndarray of shape (n_patients,)
        median_transport_times[i] is the median travel time in minutes from patient i to their hospital.
    
    median_return_times : np.ndarray of shape (n_hospitals, n_stations)
        median_return_times[i, j] is the median travel time in minutes from hospital i to station j.
    
    dispatch_order : np.ndarray of shape (n_patients, n_stations)
        dispatch_order[i] is the station IDs sorted by increasing travel time to patient i.
    
    avg_calls_per_day : int
        Average number of calls per day.
    
    remove_outliers : bool
        Whether to remove outliers when sampling travel times and support times.
    
    n_days : int
        Duration of a run.
    
    n_replications : int
        Number of times to replicate simulation.
    """
    def __init__(
        self,
        data: 'EMSData',
        avg_calls_per_day: int,
        remove_outliers: bool = True,
        n_days: int = 100,
        n_replications: int = 1
    ):
        # To keep Simulation instance lightweight, extract only necessary attributes from EMSData
        self.response_time_threshold = data.response_time_threshold
        self.patients = data.patients[['demand_node', 'hospital']]
        self.n_demand_nodes = len(data.demand_nodes)
        support_times = data.support_times
        if remove_outliers:
            support_times = np.array(support_times)
            support_times = support_times[support_times < np.percentile(support_times, 99)]
            support_times = support_times.tolist()
        self.support_times = support_times
        self.arrival_times = data.arrival_times
        self.median_response_times = data.median_response_times
        self.median_transport_times = data.median_transport_times
        self.median_return_times = data.median_return_times
        self.dispatch_order = data.dispatch_order

        self.avg_calls_per_day = avg_calls_per_day
        self.remove_outliers = remove_outliers
        self.n_days = n_days
        self.n_replications = n_replications
    
    def save_instance(self, filename: str):
        """Dump Simulation instance to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_instance(cls, filename: str) -> 'Simulation':
        """Load Simulation instance from a pickle file."""
        with open(filename, 'rb') as f:
            instance = pickle.load(f)
        return instance
    
    @staticmethod
    def _sample_t_distribution(df: float, retry_if_outlier: bool = True) -> float:
        """Sample from a t-distribution with df degrees of freedom.

        Variance of t-distribution infinity or undefined for df <= 2, so only allow df > 2.

        The t-distribution can produce extreme values, so this function allows for retrying if an outlier is sampled.
        """
        if df <= 2:
            raise ValueError("df must be greater than 2")
        if retry_if_outlier:
            # Try 3 times before giving up and returning 0
            for _ in range(3):
                eps = np.random.standard_t(df=df)
                if np.abs(eps) < 3*math.sqrt(df/(df-2)):  # Check if within 3 std devs
                    return eps
            return 0
        return np.random.standard_t(df=df)

    @staticmethod
    def sample_pretravel_delay(
            zero_prob: float = 0.0055,
            mu: float = 4.68,
            sigma: float = 0.38,
            tau: float = 5.37,
            retry_if_outlier: bool = True
        ) -> float:
        """Sample pre-travel delay.

        Original paper: https://doi.org/10.1287/mnsc.1090.1142

        Parameters
        ----------
        zero_prob : float, optional
            Discrete probability of zero pre-travel delay.
        
        mu, sigma : float, optional
            Parameters for nonzero part of the mixture. See original paper.
        
        tau : float, optional
            Degrees of freedom for t-distribution.
        
        retry_if_outlier : bool, optional
            Whether to retry if an outlier is sampled.

        Returns
        -------
        float
            Pre-travel delay in minutes.
        """
        if np.random.binomial(n=1, p=zero_prob):
            return 0
        eps = Simulation._sample_t_distribution(df=tau, retry_if_outlier=retry_if_outlier)
        pretravel_delay = math.exp(mu + sigma*eps)
        pretravel_delay /= 60  # Convert to minutes
        return pretravel_delay

    @staticmethod
    def sample_travel_time(
            median: float,
            b_0: float = 0.336,
            b_1: float = 0.000058,
            b_2: float = 0.0388,
            tau: float = 4.0,
            retry_if_outlier: bool = True
        ) -> float:
        """Sample travel time given median.

        The t-distribution can produce extreme values, so this function allows for retrying if an outlier is sampled.
        
        Original paper: https://doi.org/10.1287/mnsc.1090.1142

        Parameters
        ----------
        median : float
            Median travel time in minutes.
        
        b_0, b_1, b_2 : float, optional
            Used to calculate coefficient of variation. See original paper.
        
        tau : float, optional
            Degrees of freedom for t-distribution.
        
        retry_if_outlier : bool, optional
            Whether to retry if an outlier is sampled.
        
        Returns
        -------
        float
            Travel time in minutes.
        """
        cv = math.sqrt(b_0*(b_2 + 1) + b_1*(b_2 + 1)*median + b_2*median**2)/median
        eps = Simulation._sample_t_distribution(df=tau, retry_if_outlier=retry_if_outlier)
        travel_time = median*math.exp(cv*eps)
        return travel_time
    
    def run_single_replication(self, solution: list[int]) -> pd.DataFrame:
        """Run a replication of the simulation using a given solution.
        
        When recording response times, we exclude pre-travel delay
        because it is independent of the solution; we only account for
        the travel time from the station to the patient, and any
        additional time the patient must wait if all ambulances in the
        system are busy. When evaluating coverage, we include pre-travel
        delay.

        Parameters
        ----------
        solution : list[int]
            Number of ambulances located at each facility.
        
        Returns
        -------
        pd.DataFrame of shape (1, 3*n_demand_nodes)
            Contains the following columns for each demand node i:
            - n_covered_<i>: Number of arrivals covered within the response time threshold
            - response_time_<i>: Total response time (excluding pre-travel delay), summed over all arrivals
            - n_arrivals_<i>: Total number of arrivals
        """
        # Tracks number of available ambulances at each station throughout the simulation
        self._available_ambulances = list(solution)
        # Priority queue of events (arrivals and returns)
        arrivals = self._create_arrivals()
        heapq.heapify(arrivals)
        self._event_queue = arrivals
        # FIFO queue patients must wait in if no ambulances are available
        self._fifo_queue = collections.deque()

        # Log results in self._results array, use self._col2idx to map column names to indices
        cols = sum([[f'n_covered_{i}', f'response_time_{i}', f'n_arrivals_{i}'] for i in range(self.n_demand_nodes)], [])
        self._col2idx = {col: idx for idx, col in enumerate(cols)}
        self._results = np.zeros(len(cols))

        # Run simulation, self._results is updated as simulation progresses
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.type == ARRIVAL:
                self._process_arrival(event)
            elif event.type == RETURN:
                self._process_return(event)
        
        # Convert results to DataFrame
        self._results = pd.DataFrame(self._results.reshape(1, -1), columns=cols)
        self._results = self._results.astype({col: int for col in cols if col.startswith('n_')})

        return self._results
    
    def _create_arrivals(self) -> list[Event]:
        """Create all arrivals for a run up front."""
        arrival_times = []
        arrival_patient_ids = []

        # Sample number of calls for each day
        for day, n_calls in enumerate(np.random.poisson(self.avg_calls_per_day, self.n_days)):
            # Sample arrival times
            arrival_times_today = random.sample(self.arrival_times, n_calls)
            arrival_times += (np.array(arrival_times_today) + 60*24*day).tolist()
            # Sample patient IDs
            arrival_patient_ids += random.sample(self.patients.index.tolist(), n_calls)
        
        # Create Events
        arrivals = [Event(ARRIVAL, t, i) for t, i in zip(arrival_times, arrival_patient_ids)]

        return arrivals
    
    def _process_arrival(self, event: Event):
        """Dispatch the nearest ambulance; if there are no available ambulances, add patient to FIFO queue."""
        current_time = event.time
        patient_id = event.id
        for station_id in self.dispatch_order[patient_id]:
            if self._available_ambulances[station_id] > 0:
                self._dispatch(station_id, patient_id, current_time, current_time)
                return
        self._fifo_queue.append((patient_id, current_time))
    
    def _process_return(self, event: Event):
        """Increment ambulance count. If FIFO queue nonempty, dispatch ambulance."""
        current_time = event.time
        station_id = event.id
        self._available_ambulances[station_id] += 1
        # If FIFO queue nonempty, then this is the only available ambulance, so dispatch immediately
        if self._fifo_queue:
            patient_id, arrival_time = self._fifo_queue.popleft()
            self._dispatch(station_id, patient_id, current_time, arrival_time)

    def _dispatch(self, station_id: int, patient_id: int, current_time: float, arrival_time: float):
        """Decrement ambulance count, sample random times, create return Event, and record metrics."""
        # Decrement ambulance count
        self._available_ambulances[station_id] -= 1

        # Sample random times
        pretravel_delay = Simulation.sample_pretravel_delay(retry_if_outlier=self.remove_outliers)
        response_travel_time = Simulation.sample_travel_time(self.median_response_times[station_id, patient_id], retry_if_outlier=self.remove_outliers)
        transport_time = Simulation.sample_travel_time(self.median_transport_times[patient_id], retry_if_outlier=self.remove_outliers)
        return_time = Simulation.sample_travel_time(self.median_return_times[self.patients.hospital[patient_id], station_id], retry_if_outlier=self.remove_outliers)
        support_time = random.choice(self.support_times)

        # Create return Event
        return_event_time = current_time + pretravel_delay + response_travel_time + transport_time + return_time + support_time
        heapq.heappush(self._event_queue, Event(RETURN, return_event_time, station_id))

        # Record response time, whether covered, and number of calls from this demand node
        demand_node = self.patients.demand_node[patient_id]
        # Don't include pretravel_delay when recording response time, but do include time waiting in FIFO queue (current_time - arrival_time)
        response_time = response_travel_time + current_time - arrival_time
        self._results[self._col2idx[f'response_time_{demand_node}']] += response_time
        # Include pretravel_delay when evaluating coverage
        response_time += pretravel_delay
        self._results[self._col2idx[f'n_covered_{demand_node}']] += response_time < self.response_time_threshold
        self._results[self._col2idx[f'n_arrivals_{demand_node}']] += 1
    
    def run(self, solution: list[int]) -> pd.DataFrame:
        """Run multiple replications of the simulation using a given solution.

        Parameters
        ----------
        solution : list[int]
            Number of ambulances located at each facility.
        
        Returns
        -------
        pd.DataFrame of shape (n_replications, 3*n_demand_nodes)
            Contains the following columns for each demand node i:
            - n_covered_<i>: Number of arrivals covered within the response time threshold
            - response_time_<i>: Total response time (excluding pre-travel delay), summed over all arrivals
            - n_arrivals_<i>: Total number of arrivals
        """
        results = [self.run_single_replication(solution) for _ in range(self.n_replications)]
        results = pd.concat(results, ignore_index=True)
        return results
