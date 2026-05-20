#!/usr/bin/env python3
"""Check Piper gripper alignment with LeRobot dataset distribution.

USAGE:
  # Offline: just show dataset stats
  python3 scripts/check_piper_lerobot_alignment.py \
      --dataset-root data/lerobot_dataset_v2_delta

  # Online: compare with live robot
  python3 scripts/check_piper_lerobot_alignment.py \
      --dataset-root data/lerobot_dataset_v2_delta \
      --can-port can0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset_stats(root: Path):
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"No stats.json at {stats_path}")
    return json.loads(stats_path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Check Piper ↔ LeRobot gripper alignment")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--can-port", type=str, default=None,
                        help="If set, connect to Piper and read live gripper state.")
    parser.add_argument("--expected-gripper-min", type=float, default=None)
    parser.add_argument("--expected-gripper-max", type=float, default=None)
    args = parser.parse_args()

    # 1. Dataset stats
    print("=" * 60)
    print("  Piper ↔ LeRobot Gripper Alignment Check")
    print("=" * 60)
    stats = load_dataset_stats(args.dataset_root)

    obs_state = stats.get("observation.state", {})
    action = stats.get("action", {})
    grip_idx = 6  # dim 7 is gripper

    ds_min = obs_state["min"][grip_idx]
    ds_max = obs_state["max"][grip_idx]
    ds_mean = obs_state["mean"][grip_idx]
    ds_std = obs_state["std"][grip_idx]

    print(f"\n  Dataset: {args.dataset_root}")
    print(f"  observation.state gripper:")
    print(f"    min={ds_min:.6f}  max={ds_max:.6f}  mean={ds_mean:.6f}  std={ds_std:.6f}")
    if action:
        print(f"  action gripper:")
        print(f"    min={action['min'][grip_idx]:.6f}  max={action['max'][grip_idx]:.6f}  "
              f"mean={action['mean'][grip_idx]:.6f}  std={action['std'][grip_idx]:.6f}")

    expected_min = args.expected_gripper_min if args.expected_gripper_min is not None else ds_min
    expected_max = args.expected_gripper_max if args.expected_gripper_max is not None else ds_max

    print(f"\n  Expected gripper range (for comparison): [{expected_min:.4f}, {expected_max:.4f}]")

    # 2. Robot state
    if args.can_port:
        print(f"\n  Connecting to Piper ({args.can_port}) ...")
        try:
            from hardware.piper_wrapper import PiperRobot
            robot = PiperRobot(can_port=args.can_port)
            robot.connect()  # connect + enable in one call
            state = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            robot_grip = state[6]
            robot.disable()
            robot.disconnect()
            print(f"  Robot gripper state: {robot_grip:.6f}")
            print(f"  Full robot state:    {[round(float(x), 4) for x in state]}")
            print()
            print("-" * 60)

            in_range = expected_min <= robot_grip <= expected_max
            if in_range:
                print("  [PASS] Robot gripper IS within dataset distribution range.")
                print(f"  Robot: {robot_grip:.4f}  ∈  [{expected_min:.4f}, {expected_max:.4f}]")
            else:
                print("  [FAIL] Robot gripper is OUTSIDE dataset distribution range.")
                print(f"  Robot: {robot_grip:.6f}  ∉  [{expected_min:.4f}, {expected_max:.4f}]")
                ratio = ds_mean / robot_grip if abs(robot_grip) > 1e-9 else float("inf")
                print(f"  Dataset mean / robot ≈ {ratio:.1f}x")
                print()
                print("  Possible causes:")
                print("    1. Physical gripper is in different position (closed vs open)")
                print("    2. Gripper unit mismatch (wrapper bug)")
                print("    3. SDK version difference")
                print()
                print("  Recommended:")
                print("    - Run test_gripper_control.py to verify gripper range")
                print("    - Check PiperRobot wrapper for unit conversions")
                print("    - Manually open gripper before deployment")
        except Exception as e:
            print(f"  [ERROR] Could not connect to Piper: {e}")
            print("  Skipping robot comparison.")
    else:
        print(f"\n  (No --can-port provided; dataset-only mode)")
        print(f"  To compare with live robot, run with --can-port can0")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
