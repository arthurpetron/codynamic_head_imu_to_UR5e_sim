# core/codynamic_simulator.py

from abc import ABC, abstractmethod
from typing import Protocol, List
import numpy as np
from core.data_provider import TimeSeriesDataProvider

class Integrator(Protocol):
    def integrate_from_state(self, x0: np.ndarray, dt: float) -> np.ndarray:
        ...

class ResidualEvaluator(Protocol):
    def evaluate(self, sim_state: np.ndarray, obs_data: np.ndarray) -> float:
        ...

class CodynamicSimulator(ABC):
    """
    Abstract interface for any update engine that supports:
    - Rewinding to past simulator state (based on sensor data)
    - Integrating forward over logarithmic windows
    - Selecting the most causally consistent trajectory
    """

    @abstractmethod
    def rewind_and_update(self, 
                          t_groundtruth: float, 
                          provider: TimeSeriesDataProvider):
        pass

    @abstractmethod
    def get_current_state(self, t_query: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_residuals(self) -> List[float]:
        pass