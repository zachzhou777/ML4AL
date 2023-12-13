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
    
    avg_calls_per_day : int, optional
        Average number of calls per day.
    
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
        Historical support (scene, transfer, and turnover) times in minutes.
    
    arrival_times : list[int]
        Historical arrival times, given as the exact minute of the day (e.g., 60*8 + 30 for 8:30 AM).
    
    average_response_times : np.ndarray of shape (n_stations, n_patients)
        average_response_times[i, j] is the average travel time in minutes from station i to patient j.
    
    average_transport_times : np.ndarray of shape (n_patients,)
        average_transport_times[i] is the average travel time in minutes from patient i to their hospital.
    
    average_return_times : np.ndarray of shape (n_hospitals, n_stations)
        average_return_times[i, j] is the average travel time in minutes from hospital i to station j.
    
    dispatch_order : np.ndarray of shape (n_patients, n_stations)
        dispatch_order[i] is the station IDs sorted by increasing travel time to patient i.
    
    avg_calls_per_day : int
        Average number of calls per day.
    
    n_days : int
        Duration of a run.
    
    n_replications : int
        Number of times to replicate simulation.
    """
    def __init__(
        self,
        data: 'EMSData',
        avg_calls_per_day: int = 100,
        n_days: int = 100,
        n_replications: int = 10
    ):
        # To keep Simulation instance lightweight, extract only necessary attributes from EMSData
        self.response_time_threshold = data.response_time_threshold
        self.patients = data.patients[['demand_node', 'hospital']]
        self.n_demand_nodes = len(data.demand_nodes)
        self.support_times = data.support_times
        self.arrival_times = data.arrival_times
        self.average_response_times = data.average_response_times
        self.average_transport_times = data.average_transport_times
        self.average_return_times = data.average_return_times
        self.dispatch_order = data.dispatch_order

        self.avg_calls_per_day = avg_calls_per_day
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

    def run_single_replication(self, solution: list[int]) -> pd.Series:
        """Run a single replication of the simulation using a given solution.

        Parameters
        ----------
        solution : list[int]
            Number of ambulances located at each facility.
        
        Returns
        -------
        pd.Series
            Contains the following entries for each demand node i:
            - covered<i>: Number of patients covered within the response time threshold
            - total<i>: Total number of patients
        """
        self._available_ambulances = solution.copy()  # Copy not really necessary, but just in case
        arrivals = self._create_arrivals()
        heapq.heapify(arrivals)  # Priority queue
        self._event_queue = arrivals
        self._patient_queue = collections.deque()  # FIFO queue for when no ambulances are available
        log_cols = ['demand_node', 'response_time']
        self._log = np.empty((len(arrivals), len(log_cols)))
        self._log_idx = 0  # Used to index into self._log

        # The simulation itself; _log is populated by the end
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.type == ARRIVAL:
                self._process_arrival(event)
            elif event.type == RETURN:
                self._process_return(event)
        
        self._log = pd.DataFrame(self._log, columns=log_cols)
        self._log = self._log.astype({'demand_node': int})

        results = self._evaluate_coverage()

        return results
    
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
        """Dispatch the nearest ambulance; if there are no available ambulances, add patient to queue."""
        current_time = event.time
        patient_id = event.id
        for station_id in self.dispatch_order[patient_id]:
            if self._available_ambulances[station_id] > 0:
                self._dispatch(station_id, patient_id, current_time, current_time)
                return
        self._patient_queue.append((patient_id, current_time))
    
    def _process_return(self, event: Event):
        """Increment ambulance count. If patient queue nonempty, dispatch ambulance."""
        current_time = event.time
        station_id = event.id
        self._available_ambulances[station_id] += 1
        # If patient queue nonempty, then this is the only available ambulance, so dispatch immediately
        if self._patient_queue:
            patient_id, arrival_time = self._patient_queue.popleft()
            self._dispatch(station_id, patient_id, current_time, arrival_time)
    
    def _dispatch(self, station_id: int, patient_id: int, current_time: float, arrival_time: float):
        """Decrement ambulance count, sample random times, create return Event, and update log."""
        # Decrement ambulance count
        self._available_ambulances[station_id] -= 1

        # Sample random times
        avg_response_time = self.average_response_times[station_id, patient_id]
        avg_transport_time = self.average_transport_times[patient_id]
        hospital_id = self.patients.hospital[patient_id]
        avg_return_time = self.average_return_times[hospital_id, station_id]
        response_time = np.random.exponential(avg_response_time)
        transport_time = np.random.exponential(avg_transport_time)
        return_time = np.random.exponential(avg_return_time)
        support_time = random.choice(self.support_times)

        # Create return Event
        return_event_time = current_time + response_time + transport_time + return_time + support_time
        heapq.heappush(self._event_queue, Event(RETURN, return_event_time, station_id))

        # Update log
        self._log[self._log_idx, 0] = self.patients.demand_node[patient_id]
        # Must also factor in time between arrival and dispatch
        self._log[self._log_idx, 1] = response_time + current_time - arrival_time
        self._log_idx += 1
    
    def _evaluate_coverage(self) -> pd.Series:
        """Evaluate coverage within response time threshold.

        Assumes self._log is a pd.DataFrame with columns 'demand_node'
        and 'response_time'. For each demand node i, count the number of
        patients covered within the response time threshold (covered<i>)
        and the total number of patients (total<i>).
        """
        # results is a DataFrame indexed by demand_node and with columns 'covered' and 'total'
        self._log['covered'] = self._log.response_time <= self.response_time_threshold
        results = self._log.groupby('demand_node').covered.agg(['sum', 'count'])
        results.rename(columns={'sum': 'covered', 'count': 'total'}, inplace=True)

        # Reindex to include demand nodes with no arrivals (covered and total are 0)
        demand_nodes = range(self.n_demand_nodes)
        results = results.reindex(demand_nodes, fill_value=0)

        # Stack results to get indices 'covered<i>' and 'total<i>'
        results = results.stack()
        results.index = results.index.map(lambda x: f'{x[1]}{x[0]}')

        return results
    
    def run(self, solution: list[int]) -> pd.DataFrame:
        """Run multiple replications of the simulation using a given solution.

        Parameters
        ----------
        solution : list[int]
            Number of ambulances located at each facility.
        
        Returns
        -------
        pd.DataFrame
            Contains `self.n_replications` rows and the following columns for each demand node i:
            - covered<i>: Number of patients covered within the response time threshold
            - total<i>: Total number of patients
        """
        results = [self.run_single_replication(solution) for _ in range(self.n_replications)]
        results = pd.DataFrame(results)
        return results
