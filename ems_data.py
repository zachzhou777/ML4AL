import os
import glob
import math
import pickle
import numpy as np
import pandas as pd
from pyproj import Proj

class EMSData:
    """Class to read and preprocess EMS data for usage in simulation and MIP models.

    Parameters
    ----------
    urbanicity : str, optional
        Either 'Urban' or 'Rural'.
    
    region_id : int, optional
        1 is urban, 4 is rural, Eric has data for others if needed. TODO merge with urbanicity?
    
    x_intervals : int, optional
        Number of grid intervals along the x-axis. Grid is used to define demand nodes.
    
    y_intervals : int, optional
        Number of grid intervals along the y-axis. Grid is used to define demand nodes.
    
    station_limit : int, optional
        Maximum number of stations. Sample a subset if needed.
    
    hospital_limit : int, optional
        Maximum number of hospitals. Sample a subset if needed.
    
    clinic_limit : int, optional
        Maximum number of clinics. Sample a subset if needed.
    
    data_dir : str, optional
        Directory containing data files.
    
    Attributes
    ----------
    urbanicity : str
        Either 'Urban' or 'Rural'.
    
    region_id : int
        1 is urban, 4 is rural.
    
    x_intervals : int
        Number of grid intervals along the x-axis. Grid is used to define demand nodes.
    
    y_intervals : int
        Number of grid intervals along the y-axis. Grid is used to define demand nodes.
    
    data_dir : str
        Directory containing data files.
    
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
        urbanicity: str = 'Rural',
        region_id: int = 4,
        x_intervals: int = 10,
        y_intervals: int = 10,
        station_limit: int = 10,
        hospital_limit: int = 10,
        clinic_limit: int = 0,
        data_dir: str = 'data'
    ):
        self.urbanicity = urbanicity
        self.region_id = region_id
        self.x_intervals = x_intervals
        self.y_intervals = y_intervals
        self.data_dir = data_dir

        urbanicity2rtt = {
            'Urban': 9.0,
            'Rural': 20.0
        }
        self.response_time_threshold = urbanicity2rtt[urbanicity]
        self.patients = self._read_patient_location_data()
        self.stations = self._read_station_data(station_limit)
        self.hospitals = self._read_hospital_data(hospital_limit, clinic_limit)
        (self.demand_nodes,
         self.patients['demand_node']) = self._define_demand_nodes()
        self.patients['hospital'] = self._assign_patients_to_hospitals()
        self.times = self._read_time_data()
        self.nonemergent_scale = self._compute_nonemergent_scale()
        self.support_times = self._compute_support_times()
        self.arrival_times = self._compute_arrival_times()
        (self.average_response_times,
         self.average_transport_times,
         self.average_return_times) = self._compute_travel_times()
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
    
    def _read_station_data(self, station_limit: int) -> pd.DataFrame:
        """Read station (a.k.a. base) data. Sample a subset of stations if there are too many."""
        stations = pd.read_csv(os.path.join(self.data_dir, 'Base_Locations.csv'))
        stations = stations[stations['Type'] == 'E']
        stations = stations[['Easting', 'Northing']]
        stations.columns = ['x', 'y']

        # Only consider stations within box defined by patient locations
        x_min, x_max = self.patients.x.min(), self.patients.x.max()
        y_min, y_max = self.patients.y.min(), self.patients.y.max()
        stations = stations[
            (x_min <= stations.x) & (stations.x <= x_max)
            & (y_min <= stations.y) & (stations.y <= y_max)
        ]

        if len(stations) > station_limit:
            stations = stations.sample(n=station_limit)
        
        stations.reset_index(drop=True, inplace=True)
        
        return stations
    
    def _read_hospital_data(self, hospital_limit: int, clinic_limit: int) -> pd.DataFrame:
        """Read hospital (and clinic) data. Sample a subset of hospitals (and clinics) if there are too many."""
        # All provider data
        providers = pd.read_csv(os.path.join(self.data_dir, 'provider_data.csv'))
        providers = providers[['X', 'Y', 'SERV_TYPE']]
        
        # Convert latitude (Y) and longitude (X) to UTM easting (x) and northing (y)
        proj = Proj(proj='utm', zone=17, ellps='WGS84', datum='NAD83', units='m')
        providers['x'], providers['y'] = proj(providers.X, providers.Y)
        providers = providers.drop(columns=['X', 'Y'])
        
        # Only consider providers within box defined by patient locations
        x_min, x_max = self.patients.x.min(), self.patients.x.max()
        y_min, y_max = self.patients.y.min(), self.patients.y.max()
        providers = providers[
            (x_min <= providers.x) & (providers.x <= x_max)
            & (y_min <= providers.y) & (providers.y <= y_max)
        ]

        # Hospital data
        hospitals = providers[providers.SERV_TYPE.isin(['Hospital - Corporation', 'Hospital - Site'])]
        if len(hospitals) > hospital_limit:
            hospitals = hospitals.sample(n=hospital_limit)
        hospitals = hospitals[['x', 'y']]

        # Clinic data
        clinics = providers[providers.SERV_TYPE.isin(['Community Health Centre', 'Independent Health Facility'])]
        if len(clinics) > clinic_limit:
            clinics = clinics.sample(n=clinic_limit)
        clinics = clinics[['x', 'y']]
        
        # Merge hospital and clinic data
        hospitals = pd.concat([hospitals, clinics])
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

    def _assign_patients_to_hospitals(self) -> pd.Series:
        """For simulation, we assume patients are always transported to their nearest hospital."""
        patients = self.patients[['x', 'y']].values
        hospitals = self.hospitals[['x', 'y']].values
        euc_dist = lambda p, q: np.linalg.norm(p - q)
        nearest_hospital = lambda i: np.argmin([euc_dist(patients[i], hospitals[j]) for j in range(hospitals.shape[0])])
        patient2hospital = pd.Series([nearest_hospital(i) for i in range(patients.shape[0])])
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

    def _compute_travel_times(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        for i in range(n_stations):
            for j in range(n_patients):
                average_response_times[i, j] = travel_time(stations[i], patients[j])
        average_transport_times = np.empty(n_patients)
        for i in range(n_patients):
            average_transport_times[i] = travel_time(patients[i], hospitals[patient2hospital[i]])
        average_return_times = np.empty((n_hospitals, n_stations))
        for i in range(n_hospitals):
            for j in range(n_stations):
                average_return_times[i, j] = self.nonemergent_scale * travel_time(hospitals[i], stations[j])
        
        return average_response_times, average_transport_times, average_return_times
    
    def _determine_dispatch_order(self) -> np.ndarray:
        """For each patient, order stations based on proximity."""
        dispatch_order = np.argsort(self.average_response_times.T)
        return dispatch_order
