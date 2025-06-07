# systems/head_to_ur5e_sampler.py

import numpy as np
from core.codynamic_simulator import CodynamicSimulator
from systems.ur5e_control_interface import UR5eControlInterface

class HeadToUR5eSampler:
    def __init__(self,
                 simulator: CodynamicSimulator,
                 ur5e_interface: UR5eControlInterface,
                 default_z_force: float = 0.0):
        self.simulator = simulator
        self.ur5e = ur5e_interface
        self.default_z_force = default_z_force

    def sample(self, t_query: float) -> np.ndarray:
        """
        Returns: joint_angles (np.ndarray of shape (6,))
        """
        state = self.simulator.get_current_state(t_query)
        if state is None:
            raise ValueError(f"No simulator state available at t={t_query:.3f}")

        theta = state[0:3]  # orientation (Euler angles)
        z_force = self.default_z_force  # can be inferred later from dynamics
        self.ur5e.update_from_head_orientation(theta, z_force)

        tcp_pos = self.ur5e.tcp_position
        tcp_quat = self.ur5e.compute_tcp_orientation()
        return self.ur5e.compute_joint_angles()