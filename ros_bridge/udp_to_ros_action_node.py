#!/usr/bin/env python3
"""
ROS 2 node: listens for UDP JSON actions and publishes to /policy/target_joint_raw.

Run with system Python (NOT the ACT conda environment):
    conda deactivate
    source /opt/ros/humble/setup.bash
    python3 ros_bridge/udp_to_ros_action_node.py
"""

import json
import math
import select
import socket
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ros_bridge.common import JOINT_NAMES, UDP_HOST, UDP_PORT


class UdpToRosActionNode(Node):
    def __init__(self):
        super().__init__("udp_to_ros_action_node")
        self._pub = self.create_publisher(JointState, "/policy/target_joint_raw", 10)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((UDP_HOST, UDP_PORT))
        self._sock.setblocking(False)
        self.get_logger().info(
            f"Listening on UDP {UDP_HOST}:{UDP_PORT}, "
            f"publishing to /policy/target_joint_raw"
        )
        self._timer = self.create_timer(0.01, self._tick)

    def _tick(self):
        while True:
            ready, _, _ = select.select([self._sock], [], [], 0)
            if not ready:
                break
            try:
                data, addr = self._sock.recvfrom(4096)
            except OSError:
                break
            if not data:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                self.get_logger().warning(f"Invalid JSON from {addr}")
                continue
            action = msg.get("action")
            if action is None or len(action) != 7:
                self.get_logger().warning(
                    f"Bad action from {addr}: length={len(action) if action else 'None'}"
                )
                continue
            if any(math.isnan(v) or math.isinf(v) for v in action):
                self.get_logger().warning(f"NaN/Inf in action from {addr}")
                continue
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = JOINT_NAMES
            js.position = [float(v) for v in action]
            self._pub.publish(js)

    def destroy_node(self):
        self._sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UdpToRosActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
