#!/usr/bin/env python3
"""
ROS 2 safety gate node.

Subscribes to /piper/joint_states and /policy/target_joint_raw, validates every
incoming raw action against joint limits and per-step delta limits, and publishes
the safe action to /piper/command_joint_safe.

Run with system Python (NOT the ACT conda environment):
    conda deactivate
    source /opt/ros/humble/setup.bash
    python3 ros_bridge/safety_gate_node.py

!!! SAFETY NOTE !!!
    In mock mode `enabled` defaults to True so the pipeline can be tested.
    On a REAL robot `enabled` MUST default to False and only be flipped True
    through an explicit ROS service call or a parameter read at startup.
    Never ship this file with `enabled = True` in production.
"""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ros_bridge.common import JOINT_LIMITS, JOINT_NAMES, MAX_DELTA


class SafetyGateNode(Node):
    def __init__(self):
        super().__init__("piper_safety_gate_node")

        # ── Safety-critical flag ──
        # Mock mode: enabled by default so the test pipeline runs.
        # Real robot: this MUST be False; enable through a service or parameter.
        self.enabled = True  # <-- SET TO False FOR REAL ROBOT DEPLOYMENT

        self._current_qpos = None
        self._current_qpos_time = 0.0
        self._stale_timeout = 0.5  # seconds — reject if joint_states is too old

        self._pub = self.create_publisher(
            JointState, "/piper/command_joint_safe", 10
        )

        self.create_subscription(
            JointState, "/piper/joint_states", self._on_joint_states, 10
        )
        self.create_subscription(
            JointState, "/policy/target_joint_raw", self._on_target_raw, 10
        )

        self.get_logger().info(
            "Safety gate started (mock mode — enabled=True). "
            "For real robot, set enabled=False and enable via service."
        )

    def _on_joint_states(self, msg: JointState):
        self._current_qpos = list(msg.position)
        self._current_qpos_time = time.time()

    def _on_target_raw(self, msg: JointState):
        if not self.enabled:
            self.get_logger().warn("Safety gate disabled — dropping raw action.")
            return

        now = time.time()
        raw = msg.position

        # ── Pre-conditions ──
        if self._current_qpos is None:
            self.get_logger().warn(
                "No current qpos yet — drop raw action.", throttle_duration_sec=2.0
            )
            return

        if now - self._current_qpos_time > self._stale_timeout:
            self.get_logger().warn(
                "Joint state stale (>0.5 s) — drop raw action.", throttle_duration_sec=2.0
            )
            return

        if msg.name != JOINT_NAMES:
            self.get_logger().warn(
                f"Joint name mismatch: got {msg.name}, expected {JOINT_NAMES}",
                throttle_duration_sec=2.0,
            )
            return

        if len(raw) != 7:
            self.get_logger().warn(f"Wrong action dim: {len(raw)}")
            return

        if any(math.isnan(v) or math.isinf(v) for v in raw):
            self.get_logger().warn("NaN/Inf in raw action — drop.")
            return

        # ── Safety clamping ──
        # Step 1: clip to absolute joint limits
        limited = []
        for name, val in zip(JOINT_NAMES, raw):
            lo, hi = JOINT_LIMITS.get(name, (-math.inf, math.inf))
            limited.append(max(lo, min(hi, float(val))))

        # Step 2: clip per-step delta relative to current_qpos
        safe = []
        for i, name in enumerate(JOINT_NAMES):
            lo, hi = JOINT_LIMITS.get(name, (-math.inf, math.inf))
            max_d = MAX_DELTA.get(name, 0.02)
            cur = float(self._current_qpos[i])
            val = limited[i]
            val = max(lo, min(hi, val))
            val = max(cur - max_d, min(cur + max_d, val))
            safe.append(val)

        # ── Publish safe command ──
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = JOINT_NAMES
        out.position = safe
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
