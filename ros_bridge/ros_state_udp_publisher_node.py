#!/usr/bin/env python3
"""
ROS node — subscribes /piper/joint_states, publishes 7D position via UDP.

Non-ROS consumers (e.g. ACT conda env) receive real arm state without
importing rclpy.  This node NEVER writes to hardware.

Run (system Python):
    source /opt/ros/humble/setup.bash
    python3 ros_bridge/ros_state_udp_publisher_node.py
"""

import argparse
import json
import socket
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

UDP_STATE_HOST = "127.0.0.1"
UDP_STATE_PORT = 50052


class RosStateUdpPublisherNode(Node):
    def __init__(self, host: str, port: int):
        super().__init__("ros_state_udp_publisher")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)
        self._latest = None

        self._sub = self.create_subscription(
            JointState, "/piper/joint_states", self._on_js, 10
        )
        self._timer = self.create_timer(1.0 / 30.0, self._tick)
        self.get_logger().info(
            f"Publishing /piper/joint_states -> UDP {host}:{port} @ 30 Hz"
        )

    def _on_js(self, msg: JointState):
        if len(msg.position) != 7:
            return
        self._latest = [float(v) for v in msg.position]

    def _tick(self):
        if self._latest is None:
            return
        payload = json.dumps({"position": self._latest})
        self._sock.sendto(payload.encode("utf-8"), self._addr)

    def destroy_node(self):
        self._sock.close()
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(
        description="ROS /piper/joint_states -> UDP bridge (read-only)"
    )
    parser.add_argument("--host", default=UDP_STATE_HOST)
    parser.add_argument("--port", type=int, default=UDP_STATE_PORT)
    parsed = parser.parse_args(args)

    rclpy.init(args=sys.argv)
    node = RosStateUdpPublisherNode(parsed.host, parsed.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
