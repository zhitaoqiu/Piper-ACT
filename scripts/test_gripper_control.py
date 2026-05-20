#!/usr/bin/env python3
"""Test gripper control: verify dim 7 responds to commands.

Keeps joints 1-6 locked to current position.
Only changes gripper value to test open/close direction.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def fmt_vec(v):
    return "[" + ", ".join(f"{float(x):.4f}" for x in v) + "]"


def main():
    from hardware.piper_wrapper import PiperRobot

    print("Connecting to Piper arm...")
    robot = PiperRobot()
    robot.connect()  # connect + enable in one call

    current = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    print(f"Initial joint state (7D): {fmt_vec(current)}")
    print(f"  Joints 1-6: {fmt_vec(current[:6])}")
    print(f"  Gripper (dim 7): {current[6]:.6f} m")
    print()

    base_joints = current[:6].copy()

    test_values = [
        0.0,
        0.001,
        0.005,
        0.01,
        0.03,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
    ]
    seen = set()
    test_values = [v for v in test_values if not (v in seen or seen.add(v))]

    print("=" * 70)
    print("Testing gripper values (joints 1-6 locked to current)")
    print(f"Base joints: {fmt_vec(base_joints)}")
    print("-" * 70)
    print(f"{'Cmd [m]':>12}  {'Feedback [m]':>12}  {'Error [m]':>12}  "
          f"{'Arm drift':>10}  {'Observed':>10}")
    print("-" * 70)

    for gv in test_values:
        target = np.append(base_joints, gv).astype(np.float32)
        success = robot.set_joint_positions(target.tolist(), velocity_pct=25)
        time.sleep(2.0)

        observed = np.asarray(robot.get_joint_positions(), dtype=np.float32)
        grip_actual = observed[6]
        arm_drift = np.max(np.abs(observed[:6] - base_joints))
        error = abs(grip_actual - gv)

        state_str = "CLOSED ~0" if gv < 0.005 else ("OPEN" if gv > 0.05 else "MID")
        print(f"  {gv:10.6f}    {grip_actual:10.6f}    {error:10.6f}    "
              f"{arm_drift:8.5f}    {state_str:>10}")

    print("-" * 70)
    print(f"\nFinal joint state: {fmt_vec(robot.get_joint_positions())}")
    print(f"Base joints drift: {fmt_vec(np.asarray(robot.get_joint_positions(), dtype=np.float32)[:6] - base_joints)}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  - If feedback ≈ command and you observe physical open/close:")
    print("    → Gripper unit is METERS. 0.00=closed, 0.08~0.10=fully open.")
    print("  - If feedback >> command (e.g., cmd=0.001 gives 0.08):")
    print("    → Gripper unit is RAW (1e6× meters). Need to divide by 1e6.")
    print("  - If feedback << command (e.g., cmd=0.08 gives 0.00008):")
    print("    → SDK expects meters but hardware returns raw. Unit mismatch.")
    print("=" * 70)

    robot.disable()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
