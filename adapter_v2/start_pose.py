"""Manual start-pose checks for adapter v2."""

from __future__ import annotations

import numpy as np

from .schema import QposTolerance, STANDARD_START_QPOS, as_qpos


def qpos_diff(current, target) -> np.ndarray:
    return np.abs(as_qpos(current, label="current qpos") - as_qpos(target, label="target qpos"))


def start_pose_guard(
    current,
    target=STANDARD_START_QPOS,
    tolerance: QposTolerance = QposTolerance(),
) -> bool:
    diff = qpos_diff(current, target)
    return bool(np.all(diff[:6] <= tolerance.arm_rad) and diff[6] <= tolerance.gripper_m)
