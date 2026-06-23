"""
Minimal UDP client that receives real 7-DoF joint state from the ROS bridge.

This file depends ONLY on Python stdlib — safe for ACT conda environment.
It does NOT import rclpy or any Piper SDK.
"""

import json
import socket
import threading

DEFAULT_STATE_PORT = 50052


class ActUdpStateClient:
    """Receive real joint state via UDP from ros_state_udp_publisher_node."""

    def __init__(self, host="127.0.0.1", port=DEFAULT_STATE_PORT, fallback=None):
        self._host = host
        self._port = port
        self._fallback = fallback
        self._lock = threading.Lock()
        self._latest = None
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _recv_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self._host, self._port))
        except OSError:
            pass
        sock.settimeout(0.5)
        while self._running:
            try:
                data, _ = sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                pos = msg.get("position")
                if isinstance(pos, list) and len(pos) == 7:
                    with self._lock:
                        self._latest = [float(v) for v in pos]
            except socket.timeout:
                continue
            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                continue
        sock.close()

    def get_joint_positions(self):
        with self._lock:
            if self._latest is not None:
                return list(self._latest)
        if self._fallback is not None:
            return list(self._fallback)
        raise RuntimeError(
            "No real state received from UDP yet. "
            "Is ros_state_udp_publisher_node running?"
        )

    def is_ready(self):
        with self._lock:
            return self._latest is not None
