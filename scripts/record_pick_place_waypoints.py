#!/usr/bin/env python3
"""Record full pick-and-place waypoints manually — no cameras, no dataset.

The user moves the arm to each pose, presses ENTER to capture.
All 10 poses are saved to a single JSON for use with quick_bottle_grasp.py.

USAGE:
  python3 scripts/record_pick_place_waypoints.py \
      --output configs/bottle_pick_place_waypoints_today.json

Pose order:
  1. start_pose       — arm at rest, gripper closed
  2. pre_grasp_pose   — above bottle, gripper closed
  3. approach_pose    — at bottle, gripper OPEN
  4. close_gripper_pose — gripper CLOSED on bottle
  5. lift_pose        — lifted up, gripper still closed
  6. place_pre_pose   — above place point, gripper closed
  7. place_pose       — at place point, gripper closed
  8. release_pose     — gripper OPEN to release
  9. retreat_pose     — arm backed away, gripper open
 10. home_pose        — back to start/safe position, gripper open
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def fmt_vec(v):
    return "[" + ", ".join(f"{float(x):.4f}" for x in v) + "]"


def interpolate_joint_path(start, target, max_step_rad=0.03, max_step_gripper=0.004):
    """Generate intermediate joint targets from start to target (excluding start, including target)."""
    diff = np.asarray(target, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    arm_steps = int(np.ceil(np.max(np.abs(diff[:6])) / max_step_rad)) if max_step_rad > 0 else 1
    grip_steps = int(np.ceil(abs(diff[6]) / max_step_gripper)) if max_step_gripper > 0 else 1
    n_steps = max(arm_steps, grip_steps, 1)
    waypoints = []
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp = np.asarray(start, dtype=np.float64) + diff * alpha
        waypoints.append(interp)
    return waypoints


POSE_ORDER = [
    ("start_pose",         "Start position (arm at rest, gripper CLOSED)"),
    ("pre_grasp_pose",     "Pre-grasp: above bottle, gripper CLOSED"),
    ("approach_pose",      "Approach: at bottle, gripper OPEN — bottle between fingers"),
    ("close_gripper_pose", "Close gripper: CLOSED onto bottle, arm stays still"),
    ("lift_pose",          "Lift: raised up, gripper stays CLOSED"),
    ("place_pre_pose",     "Place-pre: above drop point, gripper stays CLOSED"),
    ("place_pose",         "Place: at drop point, gripper stays CLOSED"),
    ("release_pose",       "Release: OPEN gripper to drop bottle"),
    ("retreat_pose",       "Retreat: back away from bottle, gripper OPEN"),
    ("home_pose",          "Home: return to safe position, gripper OPEN"),
]


def main():
    parser = argparse.ArgumentParser(description="Manual pick-and-place waypoint recorder")
    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "configs" / "bottle_pick_place_waypoints_today.json")
    parser.add_argument("--open-gripper", type=float, default=0.10)
    parser.add_argument("--close-gripper", type=float, default=0.0)
    args = parser.parse_args()

    print("=" * 60)
    print("  Pick-and-Place Waypoint Recorder")
    print("=" * 60)
    print(f"  Output: {args.output}")
    print(f"  open_gripper={args.open_gripper}  close_gripper={args.close_gripper}")
    print()
    print("  For each pose:")
    print("    1. Manually move the arm to the described pose.")
    print("    2. Press ENTER to capture the current joint state.")
    print("    3. Optionally press R to re-read position before ENTER.")
    print("  Ctrl+C to abort at any time.")
    print()

    from hardware.piper_wrapper import PiperRobot
    robot = PiperRobot(can_port=args.can_port)
    robot.connect()  # connect + enable in one call
    print("  Arm ENABLED.\n")

    waypoints = {}
    try:
        for pose_key, description in POSE_ORDER:
            print("-" * 60)
            print(f"\n  >>> {pose_key}")
            print(f"      {description}")
            print()

            while True:
                state = [float(x) for x in robot.get_joint_positions()]
                sys.stdout.write(f"\r  Current: {fmt_vec(state)}  [ENTER=capture, R=refresh]  ")
                sys.stdout.flush()

                inp = input().strip().lower()
                if inp == 'r':
                    continue
                if inp == '':
                    waypoints[pose_key] = state
                    print(f"\n  Captured {pose_key}: {fmt_vec(state)}")
                    break

        # ── Build output ──
        out = {
            "source": "manual_pick_place_record",
            "created_at": datetime.now().isoformat(),
            "open_gripper": args.open_gripper,
            "close_gripper": args.close_gripper,
            "notes": "Manual pick-and-place waypoint recording",
        }
        for pk, _ in POSE_ORDER:
            out[pk] = waypoints[pk]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\n  Saved → {args.output}")

        # ── Return to start pose before disabling ──
        print("\n" + "=" * 60)
        print("  Returning to start pose ...")
        start_pose = np.array(waypoints["start_pose"], dtype=np.float64)
        cur = np.array(robot.get_joint_positions(), dtype=np.float64)
        print(f"  Current: {fmt_vec(cur)}")
        print(f"  Target:  {fmt_vec(start_pose)}")
        path = interpolate_joint_path(cur, start_pose, max_step_rad=0.03, max_step_gripper=0.004)
        for i, pt in enumerate(path):
            robot.set_joint_positions(pt.tolist(), velocity_pct=50)
            if i == 0 or i == len(path) - 1 or (i + 1) % 10 == 0:
                print(f"  return step {i+1:3d}/{len(path):3d}  {fmt_vec(pt)}")
            time.sleep(0.05)
        print("  Returned to start pose.")

        print(f"\n  Next step (dry-run):")
        print(f"  python3 scripts/quick_bottle_grasp.py \\")
        print(f"      --waypoints {args.output} \\")
        print(f"      --mode pick_place --dry-run --velocity-pct 50")

        print(f"\n  Next step (step-confirm):")
        print(f"  python3 scripts/quick_bottle_grasp.py \\")
        print(f"      --waypoints {args.output} \\")
        print(f"      --mode pick_place --step-confirm --velocity-pct 50 --log-result")

    except KeyboardInterrupt:
        print("\n\n  Interrupted — partial poses not saved.")
        return 1
    finally:
        print()
        print("  Done. Arm stays ENABLED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
