# core/history.py

from core.data_provider import TimeSeriesDataProvider
import numpy as np
from typing import List, Tuple
import math

class SimulatorStateHistory(TimeSeriesDataProvider):
    """
    TimeSeriesDataProvider with interpolation and log-distributed retrievals.
    """
    def __init__(self, maxlen: int = 2000):
        super().__init__(maxlen)
        self.residuals: List[Tuple[float, float]] = []

    def add(self, t: float, x: np.ndarray, residual: float = None):
        super().add(t, x)
        if residual is not None:
            self.residuals.append((t, residual))

    def get_residuals(self) -> List[float]:
        """Return just the residuals, in time order."""
        return [r for _, r in self.residuals]

    def get_state_at(self, t: float) -> np.ndarray:
        pair = self.interpolate(t)
        return pair[1] if pair else None

    def truncate_after(self, t: float):
        self.data = [(ts, val) for ts, val in self.data if ts <= t]

    def get_log_backwards_range(self, t: float, base: float = math.e, depth: int = 6) -> list:
        return [t - base**(-i) for i in range(1, depth + 1)]