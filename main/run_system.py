# main/run_system.py

import time
import numpy as np
from threading import Thread
from implementations.simple_head_state_model import SimpleHeadStateModel
from implementations.simple_head_codynamic_simulator import SimpleHeadCodynamicSimulator
from implementations.imu_data_provider import IMUDataProvider
from implementations.imu_state import IMUStateHistory
from inputs.imu_receive_server import UDPIMUServerThread
from core.history import SimulatorStateHistory
from systems.ur5e_control_interface import UR5eControlInterface
from systems.head_to_ur5e_sampler import HeadToUR5eSampler

# 1. Create head physics model
I = np.array([0.01, 0.01, 0.01])
k = np.array([1.5, 2.0, 1.8])
gamma = np.array([0.2, 0.3, 0.25])
r_imu = np.array([0.0, 0.0, 0.1])
model = SimpleHeadStateModel(I, k, gamma, r_imu)

# 2. Create initial state and history
x0 = np.zeros(6)
sim_history = SimulatorStateHistory()
t0 = time.time()
sim_history.add(t0, x0)

# 3. Set up IMU data server
imu_state = IMUStateHistory()
imu_server = UDPIMUServerThread(imu_state)
imu_server.start()

# 4. Initialize simulator + control interface
sim = SimpleHeadCodynamicSimulator(model, sim_history, x0)
ur5e = UR5eControlInterface()
sampler = HeadToUR5eSampler(sim, ur5e)

# 5. Connect: wrap IMUStateHistory in TimeSeriesDataProvider interface
class LiveIMUProvider(IMUDataProvider):
    def get_all(self):
        return imu_state.get_all()

imu_provider = LiveIMUProvider()

# 6. Sampling loop with callback
def send_data_to_ur5e(callback, sampling_rate_hz=30.0):
    period = 1.0 / sampling_rate_hz

    def run():
        while True:
            now = time.time()
            try:
                sim.rewind_and_update(now, imu_provider)
                joint_angles = sampler.sample(now)
                callback(joint_angles)
            except Exception as e:
                print(f"[WARN] Sampling error: {e}")
            time.sleep(period)

    thread = Thread(target=run, daemon=True)
    thread.start()

if __name__ == "__main__":
    def print_joint_angles(joint_angles):
        print("Joint Angles:", np.round(joint_angles, 3))

    send_data_to_ur5e(callback=print_joint_angles, sampling_rate_hz=30)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down IMU server...")
        imu_server.stop()
        imu_server.join()