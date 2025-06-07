# core/data_provider.py

from threading import Lock
from bisect import bisect_left
from collections import deque
from typing import List, Tuple, Optional
import numpy as np


class TimeSeriesDataProvider:
    """
    A buffer for timestamped nD data. Supports non-monotonic insertion,
    causal reprocessing, and interpolated queries.
    Thread-safe.
    """

    def __init__(self, maxlen: int = 2000):
        self._history: deque[Tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._lock = Lock()

    def add(self, timestamp: float, data: np.ndarray):
        with self._lock:
            self._history.append((timestamp, data))

    def get_all(self) -> Optional[List[Tuple[float, np.ndarray]]]:
        with self._lock:
            return list(self._history) if self._history else None

    def get_latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self._lock:
            return self._history[-1] if self._history else None

    def get_latest_n(self, n: int) -> List[Tuple[float, np.ndarray]]:
        with self._lock:
            return list(self._history)[-n:]

    def get_interpolated(self, t_query: float) -> Optional[np.ndarray]:
        with self._lock:
            # Just take a snapshot of the history, release lock immediately
            history = list(self._history)

        if not history:
            return None

        # Now compute safely without lock
        sorted_history = sorted(history, key=lambda pair: pair[0])
        timestamps = [t for t, _ in sorted_history]
        index = bisect_left(timestamps, t_query)

        if index == 0:
            return sorted_history[0][1]
        elif index == len(timestamps):
            return sorted_history[-1][1]

        t0, x0 = sorted_history[index - 1]
        t1, x1 = sorted_history[index]
        alpha = (t_query - t0) / (t1 - t0)
        return (1 - alpha) * x0 + alpha * x1