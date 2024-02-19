import os
import glob
import math
import pickle
import numpy as np
import pandas as pd
from pyproj import Proj
from tqdm import tqdm, trange

# TODO: Not really sure if this is the best place for these constants
# Toronto (region_id=1) data
# Source: https://www.toronto.ca/wp-content/uploads/2021/04/9765-Annual-Report-2020-web-final-compressed.pdf
TORONTO_N_AMBULANCES = 234
TORONTO_AVG_CALLS_PER_DAY = 194109/365

class EMSData:
    """Class to read and preprocess EMS data for usage in simulation and MIP models.

    Parameters
    ----------
    region_id : int
        One of 1, 2, 3, 4, 5, 6, 9, or 14.
    
    x_intervals : int, optional
        Number of grid intervals along the x-axis. Grid is used to define demand nodes.
    
    y_intervals : int, optional
        Number of grid intervals along the y-axis. Grid is used to define demand nodes.
    
    use_clinics : bool, optional
        Whether patients can be transported to clinics in addition to hospitals.
    
    data_dir : str, optional
        Directory containing data files.
    
    verbose : bool, optional
        Whether to print progress bars.
    
    Attributes
    ----------
    region_id : int
        One of 1, 2, 3, 4, 5, 6, 9, or 14.
    
    x_intervals : int
        Number of grid intervals along the x-axis. Grid is used to define demand nodes.
    
    y_intervals : int
        Number of grid intervals along the y-axis. Grid is used to define demand nodes.
    
    data_dir : str
        Directory containing data files.
    
    urbanicity : str
        Either 'Urban' or 'Rural'. Determined by region_id.
    
    response_time_threshold : float
        Response time threshold in minutes. Set to 9.0 for urban and 20.0 for rural.
    
    patients : pd.DataFrame
        Historical patient location data. Includes the following columns:
        - x: UTM easting coordinate.
        - y: UTM northing coordinate.
        - demand_node: Demand node ID.
        - hospital: Nearest hospital ID.
    
    demand_nodes : pd.DataFrame
        Demand node locations. Includes the following columns:
        - x: UTM easting coordinate.
        - y: UTM northing coordinate.
        - demand: Number of patients assigned to the demand node.
    
    stations : pd.DataFrame
        Station location data. Includes columns x and y for UTM coordinates.
    
    hospitals : pd.DataFrame
        Hospital (and clinic) location data. Includes columns x and y for UTM coordinates.
    
    times : pd.DataFrame
        Time data. Used to compute non-emergent scale, support times, and arrival times.
    
    nonemergent_scale : float
        Ratio of non-emergent response time to emergent response time based on historical data.
    
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
    """
    def __init__(
        self,
        region_id: int,
        x_intervals: int = 10,
        y_intervals: int = 10,
        use_clinics: bool = False,
        data_dir: str = 'data',
        verbose: bool = False
    ):
        self.region_id = region_id
        self.x_intervals = x_intervals
        self.y_intervals = y_intervals
        self.data_dir = data_dir
        # TODO remaining regions
        region2urbanicity = {
            1: 'Urban',
            4: 'Rural'
        }
        self.urbanicity = region2urbanicity[region_id]
        # TODO allow user to set response_time_threshold
        urbanicity2rtt = {
            'Urban': 9.0,
            'Rural': 20.0
        }
        self.response_time_threshold = urbanicity2rtt[self.urbanicity]
        self.patients = self._read_patient_location_data()
        self.stations = self._read_station_data()
        self.hospitals = self._read_hospital_data(use_clinics)
        (self.demand_nodes,
         self.patients['demand_node']) = self._define_demand_nodes()
        self.patients['hospital'] = self._assign_patients_to_hospitals(verbose)
        self.times = self._read_time_data()
        self.nonemergent_scale = self._compute_nonemergent_scale()
        self.support_times = self._compute_support_times()
        self.arrival_times = self._compute_arrival_times()
        (self.average_response_times,
         self.average_transport_times,
         self.average_return_times) = self._compute_travel_times(verbose)
        self.dispatch_order = self._determine_dispatch_order()
    
    def save_instance(self, filename: str):
        """Dump EMSData instance to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_instance(cls, filename: str) -> 'EMSData':
        """Load EMSData instance from a pickle file."""
        with open(filename, 'rb') as f:
            instance = pickle.load(f)
        return instance
    
    def _read_patient_location_data(self) -> pd.DataFrame:
        """Read patient location data files into a single DataFrame."""
        patients = pd.DataFrame()
        for filepath in glob.glob(os.path.join(self.data_dir, f'MRegion{self.region_id}KDE*.csv')):
            patients = pd.concat([patients, pd.read_csv(filepath, header=None)], ignore_index=True)
        patients.columns = ['y', 'x']
        patients = patients[['x', 'y']]
        return patients
    
    def _read_station_data(self) -> pd.DataFrame:
        """Read station (a.k.a. base) data."""
        stations = pd.read_csv(os.path.join(self.data_dir, 'Base_Locations.csv'))
        # Type values: E = EMS, F = Fire, P = Police
        stations = stations[(stations['Type'] == 'E') & (stations['Region'] == self.region_id)]
        stations = stations[['Easting', 'Northing']]
        stations.columns = ['x', 'y']
        stations.reset_index(drop=True, inplace=True)
        return stations
    
    def _read_hospital_data(self, use_clinics: bool) -> pd.DataFrame:
        """Read hospital (and clinic) data."""
        # All provider data
        providers = pd.read_csv(os.path.join(self.data_dir, 'provider_data.csv'))
        providers = providers[['X', 'Y', 'SERV_TYPE']]
        
        # Convert latitude (Y) and longitude (X) to UTM easting (x) and northing (y)
        proj = Proj(proj='utm', zone=17, ellps='WGS84', datum='NAD83', units='m')
        providers['x'], providers['y'] = proj(providers.X, providers.Y)
        providers = providers.drop(columns=['X', 'Y'])
        
        # Drop providers too far from patients (1e4 means 10 km)
        x_min, x_max = self.patients.x.min() - 1e4, self.patients.x.max() + 1e4
        y_min, y_max = self.patients.y.min() - 1e4, self.patients.y.max() + 1e4
        providers = providers[
            (x_min <= providers.x) & (providers.x <= x_max)
            & (y_min <= providers.y) & (providers.y <= y_max)
        ]

        # Keep only hospitals (and clinics)
        serv_types = ['Hospital - Corporation', 'Hospital - Site']
        if use_clinics:
            serv_types += ['Community Health Centre', 'Independent Health Facility']
        hospitals = providers[providers.SERV_TYPE.isin(serv_types)]
        hospitals = hospitals[['x', 'y']]
        hospitals.reset_index(drop=True, inplace=True)

        return hospitals
    
    def _define_demand_nodes(self) -> tuple[pd.DataFrame, pd.Series]:
        """Using historical patient location data, define demand nodes.

        Historical demand is broken up into regions by an
        x_intervals-by-y_intervals grid. For each region where there is
        demand, the centroid of the demand points within the region is
        used as the location of a demand node. Patients are assigned to
        demand nodes based on grid region.
        """
        x = self.patients.x
        y = self.patients.y

        # Use np.nextafter because we use strict inequalities later (see in_region)
        x_min, x_max = x.min(), np.nextafter(x.max(), x.max()+1)
        y_min, y_max = y.min(), np.nextafter(y.max(), y.max()+1)
        x_gridlines = np.linspace(x_min, x_max, self.x_intervals+1)
        y_gridlines = np.linspace(y_min, y_max, self.y_intervals+1)

        centroids = []
        demand = []
        patient2node = pd.Series(0, index=self.patients.index)
        demand_node_id = 0
        for i in range(self.x_intervals):
            for j in range(self.y_intervals):
                in_region = ((x_gridlines[i] <= x) & (x < x_gridlines[i + 1])
                             & (y_gridlines[j] <= y) & (y < y_gridlines[j + 1]))
                if in_region.any():
                    centroids.append(self.patients[in_region][['x', 'y']].mean())
                    demand.append(in_region.sum())
                    patient2node[in_region] = demand_node_id
                    demand_node_id += 1
        demand_nodes = pd.DataFrame(centroids)
        demand_nodes['demand'] = demand

        return demand_nodes, patient2node

    def _assign_patients_to_hospitals(self, verbose: bool) -> pd.Series:
        """For simulation, we assume patients are always transported to their nearest hospital."""
        patients = self.patients[['x', 'y']].values
        hospitals = self.hospitals[['x', 'y']].values
        euc_dist = lambda p, q: np.linalg.norm(p - q)
        nearest_hospital = lambda i: np.argmin([euc_dist(patients[i], hospitals[j]) for j in range(hospitals.shape[0])])
        patient2hospital = pd.Series([nearest_hospital(i) for i in trange(patients.shape[0], desc="Assigning patients to hospitals", disable=not verbose)])
        return patient2hospital
    
    def _read_time_data(self) -> pd.DataFrame:
        """Read Times_<urbanicity>data.csv.

        Keep only the columns needed to compute non-emergent scale, support times, and arrival times.
        """
        time_data = pd.read_csv(os.path.join(self.data_dir,  f'Times_{self.urbanicity}data.csv'))
        time_data = time_data[[
            'Response_Time', 'OG', 'eResponse_23-[ResponseModetoScene]',  # Non-emergent scale
            'Scene_Time', 'Transfer_Time', 'Turnover_Time',  # Support times
            'eTimes_01-[PSAPCallDate/Time]'  # Arrival times
        ]]
        return time_data
    
    def _compute_nonemergent_scale(self) -> float:
        """Compute ratio of non-emergent response time to emergent response time based on historical data."""
        data = self.times
        data = data.dropna(subset='Response_Time')
        data = data[data.OG == 0]  # TODO ask Eric what OG means
        emergent_times = data.loc[data['eResponse_23-[ResponseModetoScene]'] == 'Emergent (Immediate Response)', 'Response_Time']
        nonemergent_times = data.loc[data['eResponse_23-[ResponseModetoScene]'] == 'Non-Emergent', 'Response_Time']
        nonemergent_scale = nonemergent_times.mean() / emergent_times.mean()
        return nonemergent_scale
    
    def _compute_support_times(self) -> list[float]:
        """Compute support times."""
        data = self.times[['Scene_Time', 'Transfer_Time', 'Turnover_Time']]
        data = data.dropna()
        data = data[(data > 0.0).all(axis=1)]  # Drop rows with zeros
        # Keep all three times together since they're likely correlated
        support_times = data.sum(axis=1).tolist()
        return support_times
    
    def _compute_arrival_times(self) -> list[int]:
        """Compute exact minute of the day of historical arrivals."""
        arrival_times = self.times['eTimes_01-[PSAPCallDate/Time]']
        arrival_times = arrival_times.dropna()
        arrival_times = pd.to_datetime(arrival_times, format='%Y-%m-%d %H:%M:%S')
        arrival_times = 60*arrival_times.dt.hour + arrival_times.dt.minute
        arrival_times = arrival_times.tolist()
        return arrival_times
    
    @staticmethod
    def travel_time(d: float, a: float = 41.0, v_c: float = 100.7) -> float:
        """Compute (median) travel time.

        Original paper: https://doi.org/10.1287/mnsc.1090.1142

        Parameters
        ----------
        d : float
            Distance in km.
        
        a : float, optional
            Acceleration in km/hr/min.
        
        v_c : float, optional
            Cruising speed in km/hr.

        Returns
        -------
        float
            Travel time in minutes.
        """
        a *= 60  # Convert to km/hr/hr
        d_c = v_c**2 / (2*a)
        travel_time = 2*math.sqrt(d/a) if d <= 2*d_c else v_c/a + d/v_c
        travel_time *= 60  # Convert to minutes
        return travel_time

    def _compute_travel_times(self, verbose: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For every pair of locations where travel may occur, compute average travel time.

        Computes average response, transport, and return times.
        """
        patients = self.patients[['x', 'y']].values
        patient2hospital = self.patients.hospital.values
        stations = self.stations[['x', 'y']].values
        hospitals = self.hospitals[['x', 'y']].values
        n_patients = patients.shape[0]
        n_stations = stations.shape[0]
        n_hospitals = hospitals.shape[0]

        euc_dist = lambda p, q: np.linalg.norm(p - q)
        # UTM coordinates are in meters, EMSData.travel_time expects km
        travel_time = lambda p, q: EMSData.travel_time(euc_dist(p, q)/1000)

        average_response_times = np.empty((n_stations, n_patients))
        average_transport_times = np.empty(n_patients)
        average_return_times = np.empty((n_hospitals, n_stations))
        n_iter = n_stations*n_patients + n_patients + n_hospitals*n_stations
        with tqdm(total=n_iter, desc="Computing travel times", disable=not verbose) as pbar:
            for i in range(n_stations):
                for j in range(n_patients):
                    average_response_times[i, j] = travel_time(stations[i], patients[j])
                    pbar.update()
            for i in range(n_patients):
                average_transport_times[i] = travel_time(patients[i], hospitals[patient2hospital[i]])
                pbar.update()
            for i in range(n_hospitals):
                for j in range(n_stations):
                    average_return_times[i, j] = self.nonemergent_scale * travel_time(hospitals[i], stations[j])
                    pbar.update()
        
        return average_response_times, average_transport_times, average_return_times
    
    def _determine_dispatch_order(self) -> np.ndarray:
        """For each patient, order stations based on proximity."""
        dispatch_order = np.argsort(self.average_response_times.T)
        return dispatch_order
