# implementations/simple_head_state_model.py

import numpy as np
from core.state_space import StateSpaceModel

class SimpleHeadStateModel(StateSpaceModel):
    def __init__(self, I: np.ndarray, k: np.ndarray, gamma: np.ndarray, r_imu: np.ndarray):
        self.I = I
        self.k = k
        self.gamma = gamma
        self.r_imu = r_imu

    def compute_dynamics(self, x: np.ndarray, torque: np.ndarray) -> np.ndarray:
        theta = x[0:3]
        omega = x[3:6]
        alpha = (torque - self.gamma * omega - self.k * theta) / self.I
        dxdt = np.zeros(6)
        dxdt[0:3] = omega
        dxdt[3:6] = alpha
        return dxdt

    def estimate_accel(self, x: np.ndarray, torque: np.ndarray) -> np.ndarray:
        theta = x[0:3]
        omega = x[3:6]
        alpha = (torque - self.gamma * omega - self.k * theta) / self.I
        term1 = np.cross(alpha, self.r_imu)
        term2 = np.cross(omega, np.cross(omega, self.r_imu))
        return term1 + term2

    def apply_kalman_update(self, x: np.ndarray, accel_meas: np.ndarray, torque: np.ndarray,
                            P: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        accel_pred = self.estimate_accel(x, torque)
        y = accel_meas - accel_pred

        H = self.compute_accel_jacobian(x, torque)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x_new = x + K @ y
        P_new = (np.eye(6) - K @ H) @ P + Q
        return x_new, P_new

    def compute_accel_jacobian(self, x: np.ndarray, torque: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        H = np.zeros((3, 6))
        for i in range(6):
            x_perturbed = x.copy()
            x_perturbed[i] += epsilon
            a_perturbed = self.estimate_accel(x_perturbed, torque)
            a_nominal = self.estimate_accel(x, torque)
            H[:, i] = (a_perturbed - a_nominal) / epsilon
        return H