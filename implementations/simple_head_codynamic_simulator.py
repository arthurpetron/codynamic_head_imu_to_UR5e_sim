# implementations/simple_head_codynamic_simulator.py

import numpy as np
from typing import List
from core.codynamic_simulator import CodynamicSimulator
from core.data_provider import TimeSeriesDataProvider
from core.state_space import StateSpaceModel
from core.history import SimulatorStateHistory


class SimpleHeadCodynamicSimulator(CodynamicSimulator):
    def __init__(self,
                 model: StateSpaceModel,
                 history: SimulatorStateHistory,
                 x0: np.ndarray,
                 dt: float = 0.004):
        self.model = model
        self.history = history
        self.x = x0.copy() if x0 is not None else np.zeros(6)
        self.dt = dt
        self.torque_input = np.zeros(3)
        self.residuals: List[float] = []
        self.sim_time: float = history.get_all()[-1][0] if history.get_all() else 0.0

        self.P = np.eye(6) * 0.01
        self.Q = np.eye(6) * 1e-4
        self.R = np.eye(3) * 1e-2

    def get_current_state(self, t_query: float) -> np.ndarray:
        return self.history.get_state_at(t_query)

    def get_residuals(self) -> List[float]:
        return self.simulator_history.get_residuals()

    def rewind_and_update(self, t_groundtruth: float, provider: TimeSeriesDataProvider):
        log_times = self.history.get_log_backwards_range(t_groundtruth)
        best_residual = float("inf")
        best_state = None
        best_time = None

        imu_data = provider.get_all()

        for t_rewind in log_times:
            x_rewind = self.history.get_state_at(t_rewind)
            if x_rewind is None:
                continue

            x = x_rewind.copy()
            sim_time = t_rewind
            residual_sum = 0.0
            P = self.P.copy()

            for t, acc, gyro in imu_data:
                if t <= t_rewind:
                    continue
                dt = t - sim_time
                if dt <= 0:
                    continue

                torque = self.model.I * (gyro - x[3:6]) / self.dt
                dxdt = self.model.compute_dynamics(x, torque)
                x = x + dxdt * dt
                sim_time = t

                accel_pred = self.model.estimate_accel(x, torque)
                residual_sum += np.linalg.norm(acc - accel_pred)

                x, P = self.model.apply_kalman_update(x, acc, torque, P, self.Q, self.R)

            if residual_sum < best_residual:
                best_residual = residual_sum
                best_state = x
                best_time = sim_time

        if best_state is not None and best_time is not None:
            self.x = best_state
            self.sim_time = best_time
            self.history.add(best_time, best_state.copy())
            self.residuals.append(best_residual)