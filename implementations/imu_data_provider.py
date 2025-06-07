# implentations/imu_data_provider.py
 
import numpy as np
from typing import List, Tuple
from core.data_provider import TimeSeriesDataProvider

class IMUDataProvider(TimeSeriesDataProvider):
    """
    A specialized data provider for IMU sensor data, storing accelerometer
    and gyroscope readings together with their timestamps.
    """

    def __init__(self, maxlen: int = 2000):
        super().__init__(maxlen)

    def add(self, timestamp: float, acc: np.ndarray, gyro: np.ndarray):
        data = np.stack([acc, gyro])  # shape: (2, 3)
        super().add(timestamp, data)

    def get_all(self) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        return [(t, d[0], d[1]) for t, d in super().get_all()]

    def get_latest(self) -> Tuple[float, np.ndarray, np.ndarray]:
        t, d = super().get_latest()
        return t, d[0], d[1]

    def get_latest_n(self, n: int) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        return [(t, d[0], d[1]) for t, d in super().get_latest_n(n)]