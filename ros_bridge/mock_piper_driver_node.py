#!/usr/bin/env python3
"""
ROS 2 mock Piper driver node.

Subscribes to /piper/command_joint_safe and publishes /piper/joint_states at 50 Hz.
This is a pure mock — it does NOT import or call any Piper SDK or CAN driver.

Run with system Python (NOT the ACT conda environment):
    conda deactivate
    source /opt/ros/humble/setup.bash
    python3 ros_bridge/mock_piper_driver_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ros_bridge.common import JOINT_NAMES, MOCK_INITIAL_QPOS


class MockPiperDriverNode(Node):
    def __init__(self):
        super().__init__("mock_piper_driver_node")

        self._qpos = list(MOCK_INITIAL_QPOS)

        self._pub = self.create_publisher(JointState, "/piper/joint_states", 10)
        self.create_subscription(
            JointState, "/piper/command_joint_safe", self._on_command, 10
        )

        # 50 Hz publish loop
        self._timer = self.create_timer(1.0 / 50.0, self._publish_state)

        self.get_logger().info(
            "Mock Piper driver started (50 Hz), "
            f"initial qpos={[f'{v:.4f}' for v in self._qpos]}"
        )

    def _on_command(self, msg: JointState):
        if msg.name != JOINT_NAMES:
            self.get_logger().warn(
                f"Ignoring command with wrong joint names: {msg.name}",
                throttle_duration_sec=2.0,
            )
            return
        self._qpos = list(msg.position)
        self.get_logger().debug(
            f"Updated qpos → {[f'{v:.4f}' for v in self._qpos]}"
        )

    def _publish_state(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = JOINT_NAMES
        js.position = list(self._qpos)
        self._pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = MockPiperDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
