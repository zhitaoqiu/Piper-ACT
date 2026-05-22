#!/usr/bin/env python3
"""Check whether Piper is manually near the adapter-v2 start pose."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapter_v2.piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from adapter_v2.schema import (
    QposTolerance,
    STANDARD_START_QPOS,
    StartGuardMode,
    as_qpos,
)
from adapter_v2.start_pose import describe_guard_result, qpos_diff, start_pose_guard


def parse_q_start(text: str):
    if not text:
        return STANDARD_START_QPOS.copy()
    return as_qpos([float(value.strip()) for value in text.split(",")], label="--q-start")


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter v2 read-only manual start-pose check.")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument(
        "--mode",
        choices=("strict", "zone"),
        default="strict",
        help="strict: scalar arm/gripper tolerance. zone: per-joint tolerances.",
    )
    parser.add_argument("--arm-tol", type=float, default=QposTolerance.arm_rad)
    parser.add_argument("--gripper-tol", type=float, default=QposTolerance.gripper_m)
    parser.add_argument(
        "--q-start",
        default="",
        help="Optional comma-separated [j1,j2,j3,j4,j5,j6,gripper] target.",
    )
    args = parser.parse_args()
    mode: StartGuardMode = args.mode
    q_start = parse_q_start(args.q_start)
    tolerance = QposTolerance(arm_rad=args.arm_tol, gripper_m=args.gripper_tol)

    bus = PiperMotorsBusV2(PiperMotorsBusV2Config(can_port=args.can_port))
    try:
        bus.connect()
        current = bus.read_qpos()
    finally:
        bus.disconnect()

    diff = qpos_diff(current, q_start)
    ok = start_pose_guard(current, q_start, mode=mode, tolerance=tolerance)
    print("Adapter v2 manual start-pose check")
    print("  motion   : none")
    print(f"  mode     : {mode}")
    print(f"  can_port : {args.can_port}")
    print(f"  target   : {[round(float(value), 6) for value in q_start]}")
    print(f"  current  : {[round(float(value), 6) for value in current]}")
    print(f"  abs diff : {[round(float(value), 6) for value in diff]}")
    print(f"  details  : {describe_guard_result(current, q_start, mode=mode, tolerance=tolerance)}")
    print(f"RESULT: {'PASS' if ok else 'FAIL - adjust the teaching pose manually'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
