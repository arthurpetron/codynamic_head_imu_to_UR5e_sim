# core/state_space.py

from abc import ABC, abstractmethod


class StateSpaceModel(ABC):
    """
    Defines the physics for a particular dynamical system (e.g. head).
    """

    @abstractmethod
    def compute_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Return dx/dt given current state x and input u (e.g. torque)."""
        pass

    @abstractmethod
    def apply_kalman_update(self, x: np.ndarray, accel_meas: np.ndarray) -> np.ndarray:
        """Apply a measurement update given accelerometer reading."""
        pass