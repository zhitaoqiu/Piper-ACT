"""Start-pose and gripper safety helpers for adapter v2."""

from __future__ import annotations

import time

import numpy as np

from .schema import (
    DEFAULT_MAX_DELTA_RAD,
    GRIPPER_OPEN_M,
    PIPER_GRIPPER_MAX_M,
    QposTolerance,
    STANDARD_START_QPOS,
    as_qpos,
)


def qpos_diff(current, target) -> np.ndarray:
    return np.abs(as_qpos(current, label="current qpos") - as_qpos(target, label="target qpos"))


def reset_guard(current, target=STANDARD_START_QPOS, tolerance: QposTolerance = QposTolerance()) -> bool:
    diff = qpos_diff(current, target)
    return bool(np.all(diff[:6] <= tolerance.arm_rad) and diff[6] <= tolerance.gripper_m)


def step_toward(current, target, max_delta_rad=DEFAULT_MAX_DELTA_RAD, max_gripper_delta_m: float = 0.004):
    cur = as_qpos(current, label="current qpos")
    tgt = as_qpos(target, label="target qpos")
    max_delta = np.asarray(max_delta_rad, dtype=np.float32).reshape(-1)
    if max_delta.shape != (6,):
        raise ValueError(f"max_delta_rad must have shape (6,), got {max_delta.shape}")
    delta = tgt - cur
    delta[:6] = np.clip(delta[:6], -max_delta, max_delta)
    delta[6] = np.clip(delta[6], -max_gripper_delta_m, max_gripper_delta_m)
    nxt = cur + delta
    nxt[6] = np.clip(nxt[6], 0.0, PIPER_GRIPPER_MAX_M)
    return nxt


def move_to_qpos(
    bus,
    target,
    *,
    hz: float = 30.0,
    max_steps: int = 300,
    tolerance: QposTolerance = QposTolerance(),
    velocity_pct: int | None = None,
):
    target_qpos = as_qpos(target, label="reset target")
    current = bus.read_qpos()
    for step in range(max_steps):
        if reset_guard(current, target_qpos, tolerance):
            return True, current, step
        sent = step_toward(current, target_qpos)
        bus.write_qpos(sent, velocity_pct=velocity_pct)
        time.sleep(1.0 / hz)
        current = bus.read_qpos()
    return reset_guard(current, target_qpos, tolerance), current, max_steps


def open_gripper(bus, target_m: float = GRIPPER_OPEN_M, *, settle_s: float = 0.3, velocity_pct: int | None = None):
    current = bus.read_qpos()
    target = current.copy()
    target[6] = float(np.clip(target_m, 0.0, PIPER_GRIPPER_MAX_M))
    bus.write_qpos(target, velocity_pct=velocity_pct)
    time.sleep(settle_s)
    return bus.read_qpos()


def reset_to_standard_start(bus, *, q_start=STANDARD_START_QPOS, hz: float = 30.0, velocity_pct: int | None = None):
    ok, final_qpos, steps = move_to_qpos(bus, q_start, hz=hz, velocity_pct=velocity_pct)
    if not ok:
        return ok, final_qpos, steps
    opened = open_gripper(bus, velocity_pct=velocity_pct)
    return reset_guard(opened, q_start), opened, steps
