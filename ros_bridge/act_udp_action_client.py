"""
Minimal UDP client that sends 7-DoF actions to the ROS mock bridge.

This file depends ONLY on the Python standard library — it must NOT import rclpy.
"""

import json
import socket

from ros_bridge.common import UDP_HOST, UDP_PORT


class UdpActionClient:
    """Send 7D actions as JSON over UDP to udp_to_ros_action_node."""

    def __init__(self, host=UDP_HOST, port=UDP_PORT):
        self._host = host
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_action(self, action_7d):
        """Send a 7-element action vector.

        Args:
            action_7d: list | numpy.ndarray | torch.Tensor of 7 floats.

        Raises:
            ValueError: if action length != 7.
        """
        # Normalise torch / numpy / list → plain list of floats
        if hasattr(action_7d, "detach"):
            action_7d = action_7d.detach().cpu().numpy().tolist()
        elif hasattr(action_7d, "tolist"):
            action_7d = action_7d.tolist()

        if len(action_7d) != 7:
            raise ValueError(
                f"Action must have length 7, got {len(action_7d)}: {action_7d}"
            )

        payload = json.dumps({"action": [float(v) for v in action_7d]})
        self._sock.sendto(payload.encode("utf-8"), (self._host, self._port))

    def close(self):
        self._sock.close()
