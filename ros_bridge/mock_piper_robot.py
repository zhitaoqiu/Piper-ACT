"""
Minimal mock of PiperRobot that sends actions via UDP and tracks local state.

This file depends ONLY on Python stdlib + ros_bridge.act_udp_action_client.
It does NOT import rclpy or any Piper SDK.
"""

import time
import numpy as np

from ros_bridge.act_udp_action_client import UdpActionClient
from ros_bridge.common import MOCK_INITIAL_QPOS


class MockPiperRobot:
    """Drop-in replacement for PiperRobot that routes actions via UDP.

    interface matches what deploy.py needs:
      - connect()
      - get_joint_positions() -> list[float] (7)
      - set_joint_positions(positions, velocity_pct=None)  # velocity_pct ignored
      - disable()   # no-op
      - enable()    # no-op
    """

    def __init__(self, can_port=None, disable_torque_on_disconnect=False):
        _ = can_port, disable_torque_on_disconnect  # accepted for api compat
        self._udp = UdpActionClient()
        self._qpos = list(MOCK_INITIAL_QPOS)

    def connect(self):
        print("  [mock] MockPiperRobot connected (UDP -> ros_bridge).")
        return self

    def get_joint_positions(self):
        return list(self._qpos)

    def set_joint_positions(self, positions, velocity_pct=None):
        _ = velocity_pct
        # Normalize
        if hasattr(positions, "tolist"):
            positions = positions.tolist()
        positions = [float(v) for v in positions]
        if len(positions) != 7:
            raise ValueError(f"set_joint_positions expects 7 values, got {len(positions)}")
        self._qpos = positions
        self._udp.send_action(positions)

    def disable(self):
        pass

    def enable(self):
        pass

    def disconnect(self):
        self._udp.close()
        print("  [mock] MockPiperRobot disconnected.")
