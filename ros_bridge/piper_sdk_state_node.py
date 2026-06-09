#!/usr/bin/env python3
"""
Piper SDK read-only joint state publisher.

Publishes /piper/joint_states from the real Piper arm WITHOUT sending any
control commands.  The arm is NOT enabled — this node only opens the CAN bus
and reads the telemetry stream.

!!! SAFETY !!!
    This node NEVER calls EnableArm, JointCtrl, MotionCtrl, GripperCtrl,
    DisableArm, or any other control function.  It is a pure reader.

    The real SDK import is gated behind --allow-real-read.  Without that flag
    the node prints a notice and exits immediately.

Run (system Python, NOT the ACT conda env):
    conda deactivate
    source /opt/ros/humble/setup.bash
    source /home/huatec/piper_py_ws/install/setup.bash   # for piper_sdk_py_driver
    python3 ros_bridge/piper_sdk_state_node.py --can-port can0 --hz 30 --allow-real-read
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ros_bridge.common import JOINT_NAMES


class PiperSdkStateNode(Node):
    def __init__(self, can_port: str, hz: float):
        super().__init__("piper_sdk_state_node")
        self._can_port = can_port
        self._period = 1.0 / max(1.0, float(hz))
        self._pub = self.create_publisher(JointState, "/piper/joint_states", 10)
        self._adapter = None
        self._timer = self.create_timer(self._period, self._tick)
        self.get_logger().info(
            f"SDK state node created (can={can_port}, hz={hz:.1f}) — not connected yet"
        )

    def connect_sdk(self):
        """Open CAN port for READ-ONLY telemetry. Does NOT enable the arm."""
        # Lazy import — only runs when --allow-real-read is given.
        # The SDK package lives in the piper_py_ws ROS 2 workspace; source its
        # setup.bash before launching this node.
        from piper_sdk_py_driver.sdk_adapter import PiperSdkAdapter

        self._adapter = PiperSdkAdapter(
            can_port=self._can_port,
            gripper_exist=True,
            enable_timeout=10.0,
        )
        self._adapter.connect()  # ConnectPort only — NO EnableArm call
        self.get_logger().info(
            f"Connected to Piper SDK on {self._can_port} (read-only, arm NOT enabled)"
        )

    def _tick(self):
        if self._adapter is None:
            return
        try:
            js = self._adapter.read_joint_state()
        except Exception as exc:
            self.get_logger().error(f"SDK read failed: {exc}", throttle_duration_sec=2.0)
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        # joint state positions are already in rad (J1-J6) / metres (gripper)
        # per SDK adapter unit conversion (see sdk_adapter.py lines 218-258)
        msg.position = [float(v) for v in js.position]
        msg.velocity = [float(v) for v in js.velocity]
        msg.effort = [float(v) for v in js.effort]
        self._pub.publish(msg)

    def disconnect_sdk(self):
        if self._adapter is not None:
            # disconnect() only calls DisableArm() if _enabled is True.
            # Since we never call enable(), _enabled stays False => no arm
            # control command is ever sent.
            self._adapter.disconnect()
            self._adapter = None
            self.get_logger().info("SDK disconnected (no arm control was sent)")

    def destroy_node(self):
        self.disconnect_sdk()
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Piper SDK read-only joint state → /piper/joint_states"
    )
    parser.add_argument("--can-port", default="can0", help="CAN port name (default: can0)")
    parser.add_argument("--hz", type=float, default=30.0, help="Publish rate in Hz (default: 30)")
    parser.add_argument(
        "--allow-real-read",
        action="store_true",
        help="Actually connect to the REAL Piper arm via CAN/SDK (read-only). "
             "Without this flag the node prints a safety notice and exits.",
    )
    parsed = parser.parse_args(args)  # args=None → parses sys.argv[1:]

    if not parsed.allow_real_read:
        print(
            "\n==============================================================\n"
            "  SAFETY: --allow-real-read NOT set.\n"
            "  This node will NOT connect to the real Piper arm.\n"
            "  To read real joint states, re-run with:\n"
            f"    python3 ros_bridge/piper_sdk_state_node.py --can-port {parsed.can_port} "
            f"--hz {parsed.hz} --allow-real-read\n"
            "==============================================================\n"
        )
        return 0

    print(
        "\n==============================================================\n"
        f"  Connecting to Piper SDK on {parsed.can_port} (READ-ONLY).\n"
        "  The arm will NOT be enabled. No control commands will be sent.\n"
        "  Press Ctrl-C to stop.\n"
        "==============================================================\n"
    )

    rclpy.init(args=sys.argv)
    node = PiperSdkStateNode(parsed.can_port, parsed.hz)
    try:
        node.connect_sdk()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
