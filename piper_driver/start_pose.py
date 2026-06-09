"""Manual start-pose checks for Piper driver."""

from __future__ import annotations

import numpy as np

from .schema import (
    QposTolerance,
    STANDARD_START_QPOS,
    ZONE_ARM_TOLERANCE_RAD,
    ZONE_GRIPPER_OPEN_MIN_M,
    StartGuardMode,
    as_qpos,
)


def qpos_diff(current, target) -> np.ndarray:
    return np.abs(as_qpos(current, label="current qpos") - as_qpos(target, label="target qpos"))


def _check_strict(
    diff: np.ndarray,
    tolerance: QposTolerance,
) -> bool:
    return bool(np.all(diff[:6] <= tolerance.arm_rad) and diff[6] <= tolerance.gripper_m)


def _check_zone(
    current: np.ndarray,
    diff: np.ndarray,
) -> bool:
    arm_ok = bool(np.all(diff[:6] <= np.asarray(ZONE_ARM_TOLERANCE_RAD, dtype=np.float32)))
    gripper_ok = bool(float(current[6]) >= ZONE_GRIPPER_OPEN_MIN_M)
    return arm_ok and gripper_ok


def start_pose_guard(
    current,
    target=STANDARD_START_QPOS,
    *,
    mode: StartGuardMode = "zone",
    tolerance: QposTolerance = QposTolerance(),
) -> bool:
    """Check whether *current* is within tolerance of *target*.

    ``strict`` uses a single scalar arm tolerance and absolute gripper
    difference.  ``zone`` uses per-joint tolerances for J1–J3 / J4–J6 and
    requires the gripper to be open above ``ZONE_GRIPPER_OPEN_MIN_M``.
    """
    diff = qpos_diff(current, target)
    if mode == "strict":
        return _check_strict(diff, tolerance)
    if mode == "zone":
        return _check_zone(current, diff)
    raise ValueError(f"Unknown start guard mode: {mode!r}")


def describe_guard_result(
    current,
    target=STANDARD_START_QPOS,
    *,
    mode: StartGuardMode = "zone",
    tolerance: QposTolerance = QposTolerance(),
) -> str:
    diff = qpos_diff(current, target)
    if mode == "strict":
        arm_max = float(np.max(diff[:6]))
        gripper = float(diff[6])
        details = f"arm max diff={arm_max:.5f} vs tol={tolerance.arm_rad:.5f}, gripper diff={gripper:.5f} vs tol={tolerance.gripper_m:.5f}"
    else:
        per_joint_ok = diff[:6] <= np.asarray(ZONE_ARM_TOLERANCE_RAD, dtype=np.float32)
        bad_joints = [f"j{i+1}" for i, ok in enumerate(per_joint_ok) if not ok]
        arm_detail = f"zone arm per-joint: {' '.join(f'j{i+1}={diff[i]:.4f}/{ZONE_ARM_TOLERANCE_RAD[i]:.4f}' for i in range(6))}"
        gripper_detail = f"gripper={float(current[6]):.5f} m (need >= {ZONE_GRIPPER_OPEN_MIN_M})"
        details = f"{arm_detail}, {gripper_detail}"
        if bad_joints:
            details += f" | exceeded: {', '.join(bad_joints)}"
    return details
