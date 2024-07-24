import os
from typing import Optional
import numpy as np
import pandas as pd
import scipy
from pyproj import Proj

REGION_ID_TO_NAME = {
    1: 'Toronto',
    2: 'Durham',
    3: 'Simcoe',
    4: 'Muskoka & Parry Sound',
    5: 'Peel',
    6: 'Hamilton',
    9: 'Halton',
    14: 'York'
}

# Urban means at least 1000 people/mi^2 (386 people/km^2)
URBAN_REGION_IDS = [1, 5, 6, 9, 14]
RURAL_REGION_IDS = [2, 3, 4]

# Number of OHCAs per year by region
# Used in conjunction with OHCA prevalence to estimate arrival rate
# Source: https://doi.org/10.1287/msom.2022.1092
OHCAS_PER_YEAR = {
    1: 2977,
    2: 570,
    3: 440,
    4: 73,
    5: 848,
    6: 618,
    9: 355,
    14: 666
}

# Estimated proportion of calls requiring transport that are OHCAs
# Denominator is Toronto transport volume in 2020
# Source: https://www.toronto.ca/wp-content/uploads/2021/04/9765-Annual-Report-2020-web-final-compressed.pdf
OHCA_PREVALENCE = 2977 / 194109

class EMSData:
    """Class to read, preprocess, and store EMS data.

    Instantiating this class will read and preprocess all data files,
    then store the data internally. Because reading the data files is so
    time-consuming, it is recommended to only instantiate this class
    once, then pickle the object for future use.

    To load data for a specific region, simply set the region_id
    attribute. This will update the remaining attributes with the data
    for that region.

    Parameters
    ----------
    use_clinics : bool, optional
        Whether patients can be transported to clinics in addition to
        hospitals.
    
    data_dir : str, optional
        Directory containing data files.
    
    Attributes
    ----------
    region_id : int
        Region ID for which to load data into attributes.

        1. Toronto
        2. Durham
        3. Simcoe
        4. Muskoka
        5. Peel
        6. Hamilton
        9. Halton
        14. York

        Setting region_id to one of the above values will update the
        remaining attributes with data for that region.
    
    urbanicity : {'urban', 'rural'}
        Urbanicity.

    patients : np.ndarray of shape (n_patients, 2)
        Simulated patient locations in UTM coordinates.
    
    stations : np.ndarray of shape (n_stations, 2)
        Station locations in UTM coordinates.
    
    hospitals : np.ndarray of shape (n_hospitals, 2)
        Hospital (and clinic) locations in UTM coordinates.
    
    nonemergent_factor : float
        Ratio of non-emergent response time to emergent response time.
    
    arrival_times : np.ndarray of shape (n_arrival_times,)
        Historical arrival times, given as the minute past midnight
        (e.g., 60*8 + 30 for 8:30 AM).
    
    support_times : np.ndarray of shape (n_support_times,)
        Historical support times in minutes.

        Support time consists of scene time, transfer time, and turnover
        time. Since these three times are likely correlated, we sum them
        rather than keep them separate.
    
    avg_calls_per_day : float
        Daily arrival rate of all EMS calls.

        Estimated as the number of OHCAs per year divided by the
        proportion of all calls requiring transport that are OHCAs.
    """
    # Attributes to set when region_id is set
    _REGION_ATTRS = [
        'patients',
        'stations',
        'hospitals',
        'avg_calls_per_day'
    ]
    _URBANICITY_ATTRS = [
        'nonemergent_factor',
        'arrival_times',
        'support_times',
    ]

    def __init__(
        self,
        use_clinics: bool = False,
        data_dir: str = 'data'
    ):
        self._region_data = {}
        self._urbanicity_data = {}
        for region_id in REGION_ID_TO_NAME:
            patients = self.read_patient_locations(region_id, data_dir)
            stations = self.read_station_locations(region_id, data_dir)
            hospitals = self.read_hospital_locations(
                use_clinics, patients, data_dir
            )
            avg_calls_per_day = (OHCAS_PER_YEAR[region_id] / OHCA_PREVALENCE) / 365
            self._region_data[region_id] = {
                'patients': patients,
                'stations': stations,
                'hospitals': hospitals,
                'avg_calls_per_day': avg_calls_per_day
            }
        for urbanicity in ['urban', 'rural']:
            nemsis_data = self.read_nemsis_data(urbanicity, data_dir)
            nonemergent_factor = self.compute_nonemergent_factor(nemsis_data)
            arrival_times = self.compute_arrival_times(nemsis_data)
            support_times = self.compute_support_times(nemsis_data)
            self._urbanicity_data[urbanicity] = {
                'nonemergent_factor': nonemergent_factor,
                'arrival_times': arrival_times,
                'support_times': support_times
            }
    
    @property
    def region_id(self) -> int:
        return self._region_id
    
    @region_id.setter
    def region_id(self, region_id: int):
        if region_id not in REGION_ID_TO_NAME:
            raise ValueError("Invalid region_id")
        self._region_id = region_id
        if region_id in RURAL_REGION_IDS:
            self.urbanicity = 'rural'
        else:
            self.urbanicity = 'urban'
        for attr in EMSData._REGION_ATTRS:
            setattr(self, attr, self._region_data[region_id][attr])
        for attr in EMSData._URBANICITY_ATTRS:
            setattr(self, attr, self._urbanicity_data[self.urbanicity][attr])

    @staticmethod
    def read_patient_locations(
        region_id: int,
        data_dir: str = 'data',
        test_id: Optional[int] = None
    ) -> np.ndarray:
        """Read simulated patient locations for a region.

        Parameters
        ----------
        region_id : int
            Region ID.
        
        data_dir : str, optional
            Directory containing data files.
        
        test_id : int, optional
            Test ID of the file to read. If None, read all files for the
            region.
        
        Returns
        -------
        np.ndarray of shape (n_patients, 2)
            Patients' UTM coordinates (easting, northing).
        """
        patients = []
        test_ids = range(100) if test_id is None else [test_id]
        for test_id in test_ids:
            filename = os.path.join(data_dir, f'MRegion{region_id}KDEtest{test_id}.csv')
            patients.append(np.loadtxt(filename, delimiter=','))
        patients = np.vstack(patients)
        patients = patients[:, [1, 0]]  # CSV files have (northing, easting) format
        return patients

    @staticmethod
    def read_station_locations(
        region_id: int,
        data_dir: str = 'data'
    ) -> np.ndarray:
        """Read station locations for a region.

        Parameters
        ----------
        region_id : int
            Region ID.
        
        data_dir : str, optional
            Directory containing data files.
        
        Returns
        -------
        np.ndarray of shape (n_stations, 2)
            Stations' UTM coordinates (easting, northing).
        """
        stations = pd.read_csv(os.path.join(data_dir, 'Base_Locations.csv'))
        stations = stations.loc[
            # Type values: E = EMS, F = Fire, P = Police
            # To simplify the problem, we only consider EMS stations
            (stations['Type'] == 'E')
            & (stations['Region'] == region_id),
            ['Easting', 'Northing']
        ]
        return stations.to_numpy()
    
    @staticmethod
    def read_hospital_locations(
        use_clinics: bool = False,
        patients: Optional[np.ndarray] = None,
        data_dir: str = 'data'
    ) -> np.ndarray:
        """Read hospital (and optionally clinic) locations.

        Optionally keep only the hospitals that are the closest one to
        any patient.

        Parameters
        ----------
        use_clinics : bool, optional
            Whether to consider clinics as hospitals.
        
        patients : pd.DataFrame of shape (n_patients, 2), optional
            Simulated patient locations in UTM coordinates.
        
        data_dir : str, optional
            Directory containing data files.
        
        Returns
        -------
        np.ndarray of shape (n_hospitals, 2)
            Hospitals' UTM coordinates (easting, northing).
        """
        # Read all provider data, keep only hospitals (and clinics)
        providers = pd.read_csv(os.path.join(data_dir, 'provider_data.csv'))
        serv_types = ['Hospital - Corporation', 'Hospital - Site']
        if use_clinics:
            serv_types += ['Community Health Centre', 'Independent Health Facility']
        hospitals = providers.loc[providers.SERV_TYPE.isin(serv_types), ['X', 'Y']]
        
        # Convert latitude (Y) and longitude (X) to UTM coordinates
        proj = Proj(proj='utm', zone=17, ellps='WGS84', datum='NAD83', units='m')
        hospitals = np.column_stack(proj(hospitals.X, hospitals.Y))
        
        # Optionally keep only the hospitals that are the closest one to any patient
        if patients is not None:
            transport_distances = scipy.spatial.distance.cdist(patients, hospitals)
            closest_hospitals_ids = np.unique(transport_distances.argmin(axis=1))
            hospitals = hospitals[closest_hospitals_ids]

        return hospitals
    
    @staticmethod
    def read_nemsis_data(
        urbanicity: str,
        keep_all_columns: bool = False,
        data_dir: str = 'data'
    ) -> pd.DataFrame:
        """Read NEMSIS data for a region.

        Optionally, keep only the columns needed to compute non-emergent
        factor, arrival times, and support times.

        Parameters
        ----------
        urbanicity : {'urban', 'rural'}
            Urbanicity of the region.
        
        keep_all_columns : bool, optional
            Whether to keep all columns in the data file.
        
        data_dir : str, optional
            Directory containing data files.
        
        Returns
        -------
        pd.DataFrame
            NEMSIS data.
        """
        nemsis_data = pd.read_csv(os.path.join(data_dir,  f'nemsis_{urbanicity}.csv'))
        if not keep_all_columns:
            nemsis_data = nemsis_data[[
                # Non-emergent factor
                'Response_Time',
                'OG',
                'eResponse_23-[ResponseModetoScene]',
                # Arrival times
                'eTimes_01-[PSAPCallDate/Time]',
                # Support times
                'Scene_Time',
                'Transfer_Time',
                'Turnover_Time'
            ]]
        return nemsis_data
    
    @staticmethod
    def compute_nonemergent_factor(nemsis_data: pd.DataFrame) -> float:
        """Compute ratio of non-emergent response time to emergent
        response time.

        Parameters
        ----------
        nemsis_data : pd.DataFrame
            NEMSIS data.
        
        Returns
        -------
        float
            Ratio of non-emergent response time to emergent response
            time.
        """
        data = nemsis_data.dropna(subset='Response_Time')
        data = data[data.OG == 0]  # TODO ask Eric what OG means
        emergent_times = data.loc[
            data['eResponse_23-[ResponseModetoScene]']
            == 'Emergent (Immediate Response)', 'Response_Time'
        ]
        nonemergent_times = data.loc[
            data['eResponse_23-[ResponseModetoScene]']
            == 'Non-Emergent', 'Response_Time'
        ]
        nonemergent_factor = nonemergent_times.mean() / emergent_times.mean()
        return nonemergent_factor
    
    @staticmethod
    def compute_arrival_times(nemsis_data: pd.DataFrame) -> np.ndarray:
        """Compute exact minute of the day of historical arrivals.
        
        Parameters
        ----------
        nemsis_data : pd.DataFrame
            NEMSIS data.
        
        Returns
        -------
        np.ndarray of shape (n_arrival_times,)
            Historical arrival times in minutes past midnight.
        """
        arrival_times = nemsis_data['eTimes_01-[PSAPCallDate/Time]']
        arrival_times = arrival_times.dropna()
        arrival_times = pd.to_datetime(arrival_times, format='%Y-%m-%d %H:%M:%S')
        arrival_times = 60*arrival_times.dt.hour + arrival_times.dt.minute
        return arrival_times.to_numpy()
    
    @staticmethod
    def compute_support_times(nemsis_data: pd.DataFrame) -> np.ndarray:
        """Compute historical support times in minutes.
        
        Parameters
        ----------
        nemsis_data : pd.DataFrame
            NEMSIS data.
        
        Returns
        -------
        np.ndarray of shape (n_support_times,)
            Historical support times in minutes.
        """
        data = nemsis_data[['Scene_Time', 'Transfer_Time', 'Turnover_Time']]
        data = data.dropna()
        data = data[(data > 0.0).all(axis=1)]  # Drop rows with zeros
        support_times = data.sum(axis=1).to_numpy()
        return support_times
