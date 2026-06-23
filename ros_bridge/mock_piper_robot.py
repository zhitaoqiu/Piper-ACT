"""
Minimal mock of PiperRobot that sends actions via UDP and tracks local state.

This file depends ONLY on Python stdlib + ros_bridge.act_udp_action_client
(+ optionally ros_bridge.act_udp_state_client for real-state mode).
It does NOT import rclpy or any Piper SDK.
"""

import time
import numpy as np

from ros_bridge.act_udp_action_client import UdpActionClient
from ros_bridge.act_udp_state_client import ActUdpStateClient
from ros_bridge.common import MOCK_INITIAL_QPOS


class MockPiperRobot:
    """Drop-in replacement for PiperRobot that routes actions via UDP.

    interface matches what deploy.py needs:
      - connect()
      - get_joint_positions() -> list[float] (7)
      - set_joint_positions(positions, velocity_pct=None)  # velocity_pct ignored
      - disable()   # no-op
      - enable()    # no-op

    When use_real_state=True, get_joint_positions() returns the latest
    real arm state from ros_state_udp_publisher_node (via UDP).
    send_action() still routes through UdpActionClient unchanged.
    """

    def __init__(self, can_port=None, disable_torque_on_disconnect=False,
                 use_real_state=False):
        _ = can_port, disable_torque_on_disconnect
        self._udp = UdpActionClient()
        self._use_real_state = use_real_state
        self._state_client = None
        self._qpos = list(MOCK_INITIAL_QPOS)

    def connect(self):
        if self._use_real_state:
            self._state_client = ActUdpStateClient(fallback=self._qpos)
            self._state_client.start()
            # Wait up to 3 s for first real state to arrive.
            deadline = time.monotonic() + 3.0
            while not self._state_client.is_ready() and time.monotonic() < deadline:
                time.sleep(0.05)
            if self._state_client.is_ready():
                self._qpos = self._state_client.get_joint_positions()
                print(f"  [mock] Real-state UDP client ready.  qpos={[f'{v:.4f}' for v in self._qpos]}")
            else:
                print("  [mock] WARNING: real-state UDP timeout (3 s), "
                      "using fallback qpos. Is ros_state_udp_publisher_node running?")
        else:
            print("  [mock] MockPiperRobot connected (UDP -> ros_bridge).")
        return self

    def get_joint_positions(self):
        if self._use_real_state and self._state_client is not None:
            try:
                self._qpos = self._state_client.get_joint_positions()
            except RuntimeError:
                pass
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
        if self._state_client is not None:
            self._state_client.stop()
        self._udp.close()
        print("  [mock] MockPiperRobot disconnected.")
