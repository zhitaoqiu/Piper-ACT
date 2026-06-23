"""
Shared constants and safety functions for the ROS mock bridge.

This file must NOT import rclpy or any Piper SDK.
"""

import math

JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper",
]

JOINT_LIMITS = {
    "joint1":   (-2.5, 2.5),
    "joint2":   (-0.05, 2.3),
    "joint3":   (-2.8, 0.2),
    "joint4":   (-2.5, 2.5),
    "joint5":   (-1.5, 1.5),
    "joint6":   (-2.5, 2.5),
    "gripper":  (0.043, 0.100),
}

MAX_DELTA = {
    "joint1":   0.030,
    "joint2":   0.030,
    "joint3":   0.030,
    "joint4":   0.020,
    "joint5":   0.020,
    "joint6":   0.020,
    "gripper":  0.004,
}

MOCK_INITIAL_QPOS = [
    0.020,
    0.004,
    -0.005,
    -0.019,
    0.328,
    0.026,
    0.099,
]

UDP_HOST = "127.0.0.1"
UDP_PORT = 50051


def clip_value(x, lo, hi):
    return max(lo, min(hi, x))


def clip_joint_positions(names, positions):
    """Clamp each joint position to its JOINT_LIMITS."""
    clipped = []
    for name, pos in zip(names, positions):
        lo, hi = JOINT_LIMITS.get(name, (-math.inf, math.inf))
        clipped.append(clip_value(pos, lo, hi))
    return clipped


def clip_delta(names, current, target):
    """Clamp target so that |target[i] - current[i]| <= MAX_DELTA[name].

    Also applies absolute joint limits.
    """
    safe = []
    for name, cur, tgt in zip(names, current, target):
        max_d = MAX_DELTA.get(name, 0.02)
        lo, hi = JOINT_LIMITS.get(name, (-math.inf, math.inf))
        # First clip to joint limits
        tgt = clip_value(tgt, lo, hi)
        # Then clip delta relative to current
        tgt = clip_value(tgt, cur - max_d, cur + max_d)
        safe.append(tgt)
    return safe
