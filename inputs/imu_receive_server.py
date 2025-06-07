import socket
import threading
import numpy as np
import socket
import json

from ezmsg.util.messagecodec import MessageDecoder
from ezmsg.util.messages.axisarray import AxisArray
from implementations.imu_data_provider import IMUDataProvider

# Debug flag
DEBUG = True

# Configuration
udp_ip = "0.0.0.0"   # Listen on all interfaces
udp_port = 9001     # Port to listen on

class UDPIMUServerThread(threading.Thread):
    def __init__(self, imu_state: IMUDataProvider, port: int = udp_port):
        super().__init__(daemon=True)
        self.imu_state = imu_state
        self.port = port
        self._stop_event = threading.Event()

    def run(self):
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Bind the socket to the address and port
        sock.bind((udp_ip, udp_port))

        print(f"Listening on UDP port {udp_port}...")

        while not self._stop_event.is_set():
            try:
                data, _ = sock.recvfrom(4096)
                decoded_data = json.loads(data.decode(), cls=MessageDecoder)
                if isinstance(decoded_data, AxisArray):
                    self._handle_packet(decoded_data)
            except Exception as e:
                print(f"[UDP Error] {e}")

        sock.close()

    def axisarray_to_imu_dicts(aa: AxisArray) -> list[dict[str, float]]:
        """
        Converts an AxisArray with shape (N, 6) into a list of dicts with labeled keys.
        Assumes time axis is a LinearAxis, and channels are [acc_x, acc_y, acc_z, gyro_roll, gyro_pitch, gyro_yaw].
        """
        time_axis = aa.axes['time']
        offset = float(time_axis.offset)
        gain = float(time_axis.gain)
        labels = ['acc_x', 'acc_y', 'acc_z', 'gyro_roll', 'gyro_pitch', 'gyro_yaw']

        return [
            {'timestamp': offset + gain * i, **{label: float(val) for label, val in zip(labels, row)}}
            for i, row in enumerate(aa.data)
        ]

    def _handle_packet(self, data: bytes):
        try:
            time_axis = data.axes['time']
            offset = float(time_axis.offset)
            gain = float(time_axis.gain)

            for i, row in enumerate(data.data):
                t = offset + gain * i
                acc = row[0:3]
                gyro = row[3:6]
                self.imu_state.add(t, acc, gyro)
        except Exception as e:
            print(f"[Parse Error] Could not parse packet: {e}")

    def stop(self):
        self._stop_event.set()