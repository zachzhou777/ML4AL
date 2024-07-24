import math
import heapq
import collections
import warnings
import numpy as np
import pandas as pd
import scipy
import scipy.spatial
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ems_data import EMSData

# Ratio of driving distance to straight-line distance
# Source: https://doi.org/10.1080/00330124.2011.583586
DETOUR_INDEX = 1.417

METRICS = [
    'coverage_9min',
    'coverage_15min',
    'survival_rate',
    'response_time_mean',
    'response_time_median',
    'response_time_90th_percentile',
    'busy_fraction',
    'service_rate'
]

class Event:
    """An event in the simulation. Can represent either an arrival or an
    ambulance return.
    """
    def __init__(self, type: str, time: float, id: int):
        self.type = type  # Either 'arrival' or 'return'
        self.time = time  # Time of event
        self.id = id  # Patient ID or station ID
    
    def __lt__(self, other: 'Event') -> bool:
        """Needed to determine next event in priority queue."""
        return self.time < other.time

class Simulation:
    """Discrete-event simulation of an EMS system.

    Parameters
    ----------
    ems_data : EMSData
        Data for the simulation.

        To keep Simulation instance lightweight, we extract only the
        necessary attributes from EMSData.
    
    n_days : int, optional
        Duration of a run.
    
    n_replications : int, optional
        Number of times to replicate simulation.
    
    outlier_fraction : float, optional
        Fraction of pre-travel delay, travel time, and support time
        distributions to remove from consideration when sampling.

        The t-distribution can produce extreme outliers, so for
        simulation, it makes sense to truncate it. We remove the
        outlier_fraction/2 smallest and largest values from the
        distribution.

        The support time empirical distribution has a long right tail,
        so we remove the outlier_fraction largest values.
    
    seed : int | None, optional
        Random seed.
    
    Attributes
    ----------
    n_days : int
        Duration of a run.
    
    n_replications : int
        Number of times to replicate simulation.
    
    outlier_fraction : float
        Fraction of pre-travel delay, travel time, and support time
        distributions to remove from consideration when sampling.
    """
    _VARS_TO_PICKLE = [
        'n_days',
        'n_replications',
        'outlier_fraction',
        '_seed',
        '_patients',
        '_stations',
        '_hospitals',
        '_nonemergent_factor',
        '_arrival_times',
        '_support_times',
        '_avg_calls_per_day'
    ]

    def __init__(
        self,
        ems_data: 'EMSData',
        n_days: int = 100,
        n_replications: int = 1,
        outlier_fraction: float = 0.01,
        seed: int | None = None
    ):
        # Attributes saved during pickling
        self.n_days = n_days
        self.n_replications = n_replications
        self.outlier_fraction = outlier_fraction
        self._seed = seed
        self._patients = ems_data.patients
        self._stations = ems_data.stations
        self._hospitals = ems_data.hospitals
        self._nonemergent_factor = ems_data.nonemergent_factor
        self._arrival_times = ems_data.arrival_times
        self._support_times = ems_data.support_times
        self._avg_calls_per_day = ems_data.avg_calls_per_day

        # Attributes not saved during pickling
        self._precompute_simulation_data()
        self._rng = np.random.default_rng(seed)
    
    def __getstate__(self) -> dict:
        """When pickling, only save necessary attributes."""
        return {var: getattr(self, var) for var in Simulation._VARS_TO_PICKLE}
    
    def __setstate__(self, state: dict):
        """When unpickling, restore attributes, including those not
        saved during pickling.
        """
        self.__dict__.update(state)
        self._precompute_simulation_data()
        self._rng = np.random.default_rng(self._seed)
    
    def run(self, solution: list[int]) -> pd.DataFrame:
        """Run multiple replications of the simulation using a given
        solution.

        The response time metrics we report do not include pre-travel
        delay because it is just noise independent of the solution; we
        only include the time spent waiting for an ambulance to become
        available if all are busy, and the travel time from the station
        to the patient. When evaluating coverage and survival rate, we
        do include pre-travel delay.

        Parameters
        ----------
        solution : list[int]
            Number of ambulances located at each facility.
        
        Returns
        -------
        pd.DataFrame of shape (n_replications, 8)
            Contains the following columns:
            - coverage_9min: Fraction of calls covered within 9 minutes
            - coverage_15min: Fraction of calls covered within 15
              minutes
            - survival_rate: Fraction of patients who survive to
              hospital discharge
            - response_time_mean: Mean response time in minutes
            - response_time_median: Median response time in minutes
            - response_time_90th_percentile: 90th percentile of response
              in minutes
            - busy_fraction: Fraction of time an ambulance is busy on
              average
            - service_rate: Service rate in calls per day
        """
        results = [
            self.run_single_replication(solution)
            for _ in range(self.n_replications)
        ]
        results = pd.concat(results, ignore_index=True)
        return results
    
    def run_single_replication(self, solution: list[int]) -> pd.DataFrame:
        """Run a single replication of the simulation using a given
        solution.

        Parameters
        ----------
        solution : list[int]
            Number of ambulances located at each facility.
        
        Returns
        -------
        pd.DataFrame of shape (1, 8)
            Contains the following columns:
            - coverage_9min: Fraction of calls covered within 9 minutes
            - coverage_15min: Fraction of calls covered within 15
              minutes
            - survival_rate: Fraction of patients who survive to
              hospital discharge
            - response_time_mean: Mean response time in minutes
            - response_time_median: Median response time in minutes
            - response_time_90th_percentile: 90th percentile of response
              in minutes
            - busy_fraction: Fraction of time an ambulance is busy on
              average
            - service_rate: Service rate in calls per day
        """
        # Generate arrival data up front
        # Arrival times and patient IDs
        arrivals = self._sample_arrivals()
        if not arrivals:
            warnings.warn("No arrivals during simulation, all metrics are NaN")
            return pd.DataFrame([{metric: np.nan for metric in METRICS}])
        # Pre-travel delays
        self._pretravel_delay_samples = (
            self.sample_pretravel_delay(size=len(arrivals))
        )
        # t-distribution samples for travel times
        self._travel_time_eps_samples = (
            self.sample_truncated_t_distribution(df=4.0, size=(len(arrivals), 3))
        )
        # Support times
        support_times = self._support_times
        outlier_threshold = np.quantile(support_times, 1 - self.outlier_fraction)
        support_times = support_times[support_times <= outlier_threshold]
        self._support_time_samples = self._rng.choice(support_times, size=len(arrivals))

        # Initialize data structures for simulation
        # For each arrival, log response time and busy time
        self._response_times = np.empty(len(arrivals))
        self._total_busy_time = 0
        # Index for pre-generated values and response times log
        self._dispatch_idx = 0
        # Track available ambulances at each station throughout simulation
        self._available_ambulances = np.rint(solution)
        # Priority queue of events (arrivals and returns)
        heapq.heapify(arrivals)
        self._event_queue = arrivals
        # FIFO queue patients must wait in if no ambulances are available
        self._fifo_queue = collections.deque()

        # Run simulation
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.type == 'arrival':
                self._process_arrival(event)
            elif event.type == 'return':
                self._process_return(event)
        # Last event is a return, its time is the end of the simulation
        self._simulation_duration = event.time

        # Compute metrics
        metrics = self._compute_metrics()
        metrics = pd.DataFrame([metrics])

        return metrics
    
    def _compute_metrics(self) -> dict[str, float]:
        """Compute metrics for the simulation."""
        n_ambulances = np.sum(self._available_ambulances)
        n_arrivals = self._response_times.shape[0]
        total_response_times = self._pretravel_delay_samples + self._response_times
        coverage_9min = np.mean(total_response_times < 9)
        coverage_15min = np.mean(total_response_times < 15)
        survival_rate = np.mean(Simulation.survival_function(total_response_times))
        response_time_mean = np.mean(self._response_times)
        response_time_median = np.median(self._response_times)
        response_time_90th_percentile = np.percentile(self._response_times, 90)
        busy_fraction = (
            self._total_busy_time / (n_ambulances * self._simulation_duration)
        )
        avg_time_per_call = self._total_busy_time / n_arrivals
        service_rate = 60*24 / avg_time_per_call
        metrics = {
            'coverage_9min': coverage_9min,
            'coverage_15min': coverage_15min,
            'survival_rate': survival_rate,
            'response_time_mean': response_time_mean,
            'response_time_median': response_time_median,
            'response_time_90th_percentile': response_time_90th_percentile,
            'busy_fraction': busy_fraction,
            'service_rate': service_rate
        }
        # Sanity check in case of future changes
        assert set(metrics.keys()) == set(METRICS)
        return metrics

    def sample_pretravel_delay(
        self,
        size: int | tuple[int] | None = None
    ) -> float | np.ndarray:
        """Sample pre-travel delay.

        Source: https://doi.org/10.1287/mnsc.1090.1142

        Parameters
        ----------
        size : int | tuple[int] | None, optional
            Output shape.

        Returns
        -------
        float | np.ndarray
            Pre-travel delay in minutes.
        """
        log_pretravel_delay = (
            self.sample_truncated_t_distribution(5.37, 4.68, 0.38, size)
        )
        pretravel_delay = np.exp(log_pretravel_delay)
        pretravel_delay *= self._rng.binomial(1, 0.9945, size)
        return pretravel_delay / 60  # Convert to minutes
    
    def sample_travel_time(
        self,
        distance: float | np.ndarray,
        size: int | tuple[int] | None = None
    ) -> float | np.ndarray:
        """Sample travel time.

        We won't call this method during the simulation because it's too
        slow to sample as we go, and we also can't easily pre-sample
        travel times because we don't know up front what distances we
        will need samples for. Instead, we precompute the median and cv
        parameters for all possible pairs of locations, pre-generate
        t-distribution samples, and use these to calculate travel times
        on the fly.

        Source: https://doi.org/10.1287/mnsc.1090.1142

        Parameters
        ----------
        distance : float | np.ndarray
            Distance in kilometers.
        
        size : int | tuple[int] | None, optional
            Output shape.
        
        Returns
        -------
        float | np.ndarray
            Travel time in minutes.
        """
        distance = np.array(distance)
        if size is None:
            size = distance.shape
        median, cv = Simulation._compute_travel_time_median_cv(distance)
        eps = self.sample_truncated_t_distribution(4.0, size=size)
        return median*np.exp(cv*eps)
    
    def sample_truncated_t_distribution(
        self,
        df: float | np.ndarray,
        loc: float | np.ndarray = 0.0,
        scale: float | np.ndarray = 1.0,
        size: int | tuple[int] | None = None
    ) -> float | np.ndarray:
        """Sample from a truncated t-distribution.

        Parameters
        ----------
        df : float | np.ndarray
            Degrees of freedom.
        
        loc, scale : float | np.ndarray, optional
            Location and scale parameters.
        
        size : int | tuple[int] | None, optional
            Output shape.
        
        Returns
        -------
        float | np.ndarray
            Drawn samples from the truncated t-distribution.
        """
        if size is None:
            size = np.broadcast(df, loc, scale).shape
        q = self._rng.uniform(
            self.outlier_fraction/2,
            1 - self.outlier_fraction/2,
            size
        )
        return scipy.stats.t.ppf(q, df, loc, scale)
    
    @staticmethod
    def driving_distance(
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
        """Compute driving distance between pairs of points.

        Parameters
        ----------
        X : np.ndarray of shape (m, 2)
            UTM coordinates of the first set of points.
        
        Y : np.ndarray of shape (n, 2)
            UTM coordinates of the second set of points.
        
        Returns
        -------
        np.ndarray of shape (m, n)
            Driving distance in kilometers between each pair of points.
        """
        return (
            scipy.spatial.distance.cdist(X, Y)  # Straight-line distance
            * DETOUR_INDEX  # Convert to driving distance
            / 1000  # Convert to kilometers
        )
    
    @staticmethod
    def survival_function(
        response_time: float | np.ndarray
    ) -> float | np.ndarray:
        """Estimate survival probability as a function of response time.

        "Survival" means "survival to hospital discharge."

        Original paper: https://doi.org/10.1067/mem.2003.266
        
        Parameters
        ----------
        response_time : float | np.ndarray
            Response time in minutes.
        
        Returns
        -------
        float | np.ndarray
            Estimated survival probability.
        """
        # Ignore overflow, happens if response time is large, result ends up being 0 anyway
        with warnings.catch_warnings(action='ignore', category=RuntimeWarning):
            return 1/(1 + np.exp(0.679 + 0.262*np.array(response_time)))
    
    def _precompute_simulation_data(self):
        """Using locations of stations, patients, and hospitals,
        precompute attributes used during simulation.
        """
        # Compute driving distances
        response_distances = Simulation.driving_distance(self._stations, self._patients)
        transport_distances = Simulation.driving_distance(self._patients, self._hospitals)
        return_distances = Simulation.driving_distance(self._hospitals, self._stations)

        # For each patient, sort stations by proximity
        self._dispatch_order = response_distances.T.argsort(axis=1)

        # Assign patients to hospitals, keep only the distance to the assigned hospital
        self._patient_to_hospital = transport_distances.argmin(axis=1)
        transport_distances = transport_distances[
            np.arange(transport_distances.shape[0]),
            self._patient_to_hospital
        ]

        # Compute travel time median and cv parameters
        self._response_median, self._response_cv = (
            Simulation._compute_travel_time_median_cv(response_distances)
        )
        self._transport_median, self._transport_cv = (
            Simulation._compute_travel_time_median_cv(transport_distances)
        )
        self._return_median, self._return_cv = (
            Simulation._compute_travel_time_median_cv(return_distances)
        )
        self._return_median *= self._nonemergent_factor
    
    @staticmethod
    def _compute_travel_time_median_cv(
        distance: float | np.ndarray,
        a: float = 41.0/60,
        v_c: float = 100.7/60,
        b_0: float = 0.336,
        b_1: float = 0.000058,
        b_2: float = 0.0388
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Compute travel time median and cv parameters for a given
        distance in kilometers.
        """
        d_c = v_c**2 / (2*a)
        median = np.where(
            distance <= 2*d_c,
            2*np.sqrt(distance)/math.sqrt(a),
            v_c/a + distance/v_c
        )
        cv = np.sqrt(b_0*(b_2 + 1) + b_1*(b_2 + 1)*median + b_2*median**2)/median
        return median, cv
    
    def _sample_arrivals(self) -> list[Event]:
        """Sample all arrivals for the simulation up front."""
        times = []
        patient_ids = []
        n_patients = len(self._patient_to_hospital)

        # Sample number of calls for each day
        for day, n_calls in enumerate(
            self._rng.poisson(self._avg_calls_per_day, self.n_days)
        ):
            # Sample arrival times
            times_today = self._rng.choice(self._arrival_times, n_calls)
            times.extend(times_today + 60*24*day)
            # Sample patient IDs
            patient_ids.extend(self._rng.choice(n_patients, n_calls))
        
        # Create Events
        arrivals = [Event('arrival', t, i) for t, i in zip(times, patient_ids)]

        return arrivals
    
    def _process_arrival(self, event: Event):
        """Dispatch the nearest ambulance; if there are no available
        ambulances, add patient to FIFO queue.
        """
        current_time = event.time
        patient_id = event.id
        for station_id in self._dispatch_order[patient_id]:
            if self._available_ambulances[station_id] > 0:
                self._dispatch(station_id, patient_id, current_time, current_time)
                return
        self._fifo_queue.append((patient_id, current_time))
    
    def _process_return(self, event: Event):
        """Increment ambulance count. If FIFO queue nonempty, dispatch
        the ambulance that just returned immediately.
        """
        current_time = event.time
        station_id = event.id
        self._available_ambulances[station_id] += 1
        # If FIFO queue nonempty, then this is the only available ambulance, so dispatch immediately
        if self._fifo_queue:
            patient_id, arrival_time = self._fifo_queue.popleft()
            self._dispatch(station_id, patient_id, current_time, arrival_time)
    
    def _dispatch(
        self,
        station_id: int,
        patient_id: int,
        current_time: float,
        arrival_time: float
    ):
        """Decrement ambulance count, sample random times, create return
        Event, and record metrics.
        """
        # Decrement ambulance count
        self._available_ambulances[station_id] -= 1

        # Sample random times (by pulling from pre-generated data)
        pretravel_delay = self._pretravel_delay_samples[self._dispatch_idx]
        eps = self._travel_time_eps_samples[self._dispatch_idx]
        support_time = self._support_time_samples[self._dispatch_idx]

        median = self._response_median[station_id, patient_id]
        cv = self._response_cv[station_id, patient_id]
        response_time = median*math.exp(cv*eps[0])

        median = self._transport_median[patient_id]
        cv = self._transport_cv[patient_id]
        transport_time = median*math.exp(cv*eps[1])

        hospital_id = self._patient_to_hospital[patient_id]
        median = self._return_median[hospital_id, station_id]
        cv = self._return_cv[hospital_id, station_id]
        return_time = median*math.exp(cv*eps[2])

        # Create return Event
        busy_time = (
            pretravel_delay  # We count pretravel delay as busy time
            + response_time
            + transport_time
            + return_time
            + support_time
        )
        return_event_time = current_time + busy_time
        heapq.heappush(
            self._event_queue,
            Event('return', return_event_time, station_id)
        )

        # Log response time (excluding pre-travel delay) and busy time
        self._response_times[self._dispatch_idx] = response_time + current_time - arrival_time
        self._total_busy_time += busy_time

        self._dispatch_idx += 1
