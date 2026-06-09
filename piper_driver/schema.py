"""Schema and measured constants for the Piper LeRobot Piper driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

StartGuardMode = Literal["strict", "zone"]

JOINT_NAMES = tuple(f"j{i}" for i in range(1, 7))
MOTOR_NAMES = JOINT_NAMES + ("gripper",)
MOTOR_POS_KEYS = tuple(f"{name}.pos" for name in MOTOR_NAMES)
STATE_NAMES = list(MOTOR_NAMES)

STATE_DIM = 7
GLOBAL_CAMERA_NAME = "global_rgb"
GLOBAL_CAMERA_KEY = f"observation.images.{GLOBAL_CAMERA_NAME}"
WRIST_CAMERA_NAME = "wrist_rgb"
WRIST_CAMERA_KEY = f"observation.images.{WRIST_CAMERA_NAME}"
DUAL_CAMERA_KEYS = (GLOBAL_CAMERA_KEY, WRIST_CAMERA_KEY)

GRIPPER_OPEN_M = 0.0995
GRIPPER_STRONG_CLOSE_MIN_M = 0.045
GRIPPER_STRONG_CLOSE_MAX_M = 0.055
PIPER_GRIPPER_MAX_M = 0.101

# Measured by the successful old single-camera 10-demo baseline. Keep this as
# the manual start-pose comparison target until piper-driver data validates it locally.
STANDARD_START_QPOS = np.asarray(
    [0.06292, 0.00750, -0.00396, 0.02732, 0.30946, -0.09826, GRIPPER_OPEN_M],
    dtype=np.float32,
)

# These are deployment safety values proven by the baseline, not mechanical
# Piper limits. Validate and tune them before broad piper-driver use.
DEFAULT_MAX_DELTA_RAD = np.asarray([0.030, 0.030, 0.030, 0.012, 0.012, 0.012], dtype=np.float32)


@dataclass(frozen=True)
class QposTolerance:
    arm_rad: float = 0.05
    gripper_m: float = 0.01


# Zone mode: per-joint tolerances for interactive data collection.
# J1–J3 are 0.08 rad, J4–J6 are 0.12 rad, gripper must be open ≥ 0.09 m.
ZONE_ARM_TOLERANCE_RAD = [0.10, 0.10, 0.10, 0.12, 0.12, 0.12]
ZONE_GRIPPER_OPEN_MIN_M = 0.09


def as_qpos(values, *, label: str = "qpos") -> np.ndarray:
    qpos = np.asarray(values, dtype=np.float32).reshape(-1)
    if qpos.shape != (STATE_DIM,):
        raise ValueError(f"{label} must have shape ({STATE_DIM},), got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{label} contains NaN or Inf: {qpos}")
    return qpos


def qpos_to_action(qpos) -> dict[str, float]:
    qpos_arr = as_qpos(qpos, label="action qpos")
    return {key: float(value) for key, value in zip(MOTOR_POS_KEYS, qpos_arr, strict=True)}


def action_to_qpos(action: dict[str, float]) -> np.ndarray:
    missing_dot = [key for key in MOTOR_POS_KEYS if key not in action]
    if not missing_dot:
        return as_qpos([action[key] for key in MOTOR_POS_KEYS], label="action")
    # Fall back to bare names (e.g. "j1" instead of "j1.pos") from dataset features.
    bare_keys = {name: idx for idx, name in enumerate(MOTOR_NAMES)}
    missing_bare = [name for name in MOTOR_NAMES if name not in action]
    if missing_bare:
        raise KeyError(
            f"action missing Piper piper-driver keys: {missing_dot} "
            f"(also tried bare names, missing: {missing_bare})"
        )
    return as_qpos([action[name] for name in MOTOR_NAMES], label="action")
