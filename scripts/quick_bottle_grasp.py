#!/usr/bin/env python3
"""Play back pre-recorded waypoints on Piper arm — no model, no training.

Usage:
  # Safety: always start with dry-run
  python3 scripts/quick_bottle_grasp.py --dry-run

  # Step-by-step confirmation
  python3 scripts/quick_bottle_grasp.py --step-confirm

  # Full auto run
  python3 scripts/quick_bottle_grasp.py --hold-seconds 3

Controls:
  ENTER = proceed to next phase (step-confirm mode)
  Q then ENTER = abort and shutdown
  Ctrl+C = emergency stop at any time
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import select
import shutil
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PIPER_GRIPPER_MAX_M = 0.101
PIPER_JOINT_LIMIT_RAD = 3.14

WAYPOINT_KEYS = ["start_pose", "pre_grasp_pose", "approach_pose",
                   "close_gripper_pose", "lift_pose"]
PHASE_NAMES   = ["start", "pre_grasp", "approach", "close_gripper", "lift"]

# Pick-and-place phase definitions: (pose_key, phase_name, gripper_behavior)
# gripper_behavior: "keep" = use waypoint value | "open" = force open_grip | "close" = force close_grip
PICK_PLACE_PHASES = [
    ("start_pose",         "start",         "keep"),
    ("pre_grasp_pose",     "pre_grasp",     "keep"),
    ("approach_pose",      "approach",      "open"),
    ("close_gripper_pose", "close_gripper", "close"),
    ("lift_pose",          "lift",          "keep"),
    ("place_pre_pose",     "place_pre",     "keep"),
    ("place_pose",         "place",         "keep"),
    ("release_pose",       "release",       "open"),
    ("retreat_pose",       "retreat",       "keep"),
    ("home_pose",          "home",          "keep"),
]

# Phases that need extra dwell after execution
DWELL_PHASES = {
    "close_gripper": "post_close_dwell",
    "release":       "post_release_dwell",
}

def detect_mode(waypoints: dict, requested: str) -> str:
    """Resolve mode: 'auto' → inspect waypoints keys; otherwise use requested."""
    if requested != "auto":
        return requested
    has_place = all(k in waypoints for k in ["place_pre_pose", "place_pose", "release_pose"])
    return "pick_place" if has_place else "grasp"


def fmt_vec(v, precision=4):
    return "[" + ", ".join(f"{float(x):.{precision}f}" for x in v) + "]"


def interpolate_joint_path(start: np.ndarray, target: np.ndarray,
                           max_step_rad: float, max_step_gripper: float):
    """Generate intermediate joint targets from start to target.

    Returns list of (7,) numpy arrays, NOT including start, including target.
    """
    diff = np.asarray(target, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    arm_steps = int(np.ceil(np.max(np.abs(diff[:6])) / max_step_rad)) if max_step_rad > 0 else 1
    grip_steps = int(np.ceil(abs(diff[6]) / max_step_gripper)) if max_step_gripper > 0 else 1
    n_steps = max(arm_steps, grip_steps, 1)

    waypoints = []
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp = np.asarray(start, dtype=np.float32) + diff * alpha
        waypoints.append(interp)
    return waypoints


def safety_check_joints(positions: np.ndarray):
    """Returns True if positions are within safe limits."""
    for j in range(6):
        if abs(positions[j]) > PIPER_JOINT_LIMIT_RAD:
            print(f"  [ERROR] Joint {j+1} = {positions[j]:.3f} exceeds limit ±{PIPER_JOINT_LIMIT_RAD}")
            return False
    if positions[6] < 0.0 or positions[6] > PIPER_GRIPPER_MAX_M:
        print(f"  [ERROR] Gripper = {positions[6]:.4f} outside [0, {PIPER_GRIPPER_MAX_M}]")
        return False
    return True


def wait_for_input(prompt="Press ENTER to continue, Q then ENTER to quit: "):
    """Terminal-based input — works over SSH."""
    s = input(prompt).strip()
    if s.lower() == 'q':
        return 'quit'
    return 'go'

def build_phase_list(waypoints: dict, open_grip: float, close_grip: float, mode: str):
    """Build ordered list of (phase_name, target_7d) from waypoint dict.

    Gripper overrides are applied based on the phase definition's gripper_behavior.
    """
    phase_defs = PICK_PLACE_PHASES if mode == "pick_place" else list(zip(WAYPOINT_KEYS, PHASE_NAMES, ["keep"] * len(WAYPOINT_KEYS)))
    # For grasp mode, override approach to open_grip
    grasp_overrides = {"approach": "open"} if mode == "grasp" else {}
    phases = []
    for pose_key, phase_name, grip_behavior in phase_defs:
        # Merge mode-specific overrides with phase definition
        behavior = grasp_overrides.get(phase_name, grip_behavior)
        target = np.array(waypoints[pose_key], dtype=np.float32)
        if behavior == "open":
            target[6] = float(open_grip)
        elif behavior == "close":
            target[6] = float(close_grip)
        phases.append((phase_name, target))
    return phases


# ── Raw terminal (for --tune-grasp jog) ──

class RawTerm:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    @staticmethod
    def read_key(timeout: float = 0.1) -> str:
        if select.select([sys.stdin], [], [], timeout)[0]:
            b = sys.stdin.read(1)
            if b == '\x1b':
                return 'ESC'
            if b == '\x03':
                raise KeyboardInterrupt
            return b if ord(b) >= 32 else ''
        return ''


def tune_grasp_jog(robot, waypoints, args):
    """Enter jog mode to micro-adjust approach / close_gripper / lift poses.

    Returns updated waypoints dict, or None if user exits without saving.
    """
    open_grip = waypoints.get("open_gripper", args.open_gripper if hasattr(args, 'open_gripper') and args.open_gripper else 0.10)
    close_grip = waypoints.get("close_gripper", 0.0)

    print("\n" + "=" * 60)
    print("  JOG MODE — micro-adjust grasp point")
    print("=" * 60)
    print(f"  open_gripper={open_grip:.4f}  close_gripper={close_grip:.4f}")
    print()
    print("  Keys:")
    print("    q/a  J1 +/-    w/s  J2 +/-    e/d  J3 +/-")
    print("    r/f  J4 +/-    t/g  J5 +/-    y/h  J6 +/-")
    print("    o=open gripper  c=close gripper  p=print pose")
    print("    l=test lift  A=save approach  C=save close_gripper  L=save lift")
    print("    x=exit")
    print("    (Hold Shift for fine step)")
    print()
    print(f"  Step: {args.jog_step:.4f} rad | Fine: {args.jog_step_fine:.4f} rad | Grip: {args.gripper_jog_step:.4f} m")
    print("-" * 60)

    cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    modified = False

    with RawTerm():
        while True:
            sys.stdout.write(
                f"\r  Pose: {fmt_vec(cur)}  "
                f"[q/a J1 w/s J2 e/d J3 r/f J4 t/g J5 y/h J6 o/c grip l=lift x=exit]  "
            )
            sys.stdout.flush()

            key = RawTerm.read_key(0.15)
            if not key:
                continue

            step = args.jog_step_fine if key.isupper() else args.jog_step
            modified_this = False

            if key == 'q':
                cur[0] += step
                modified_this = True
            elif key == 'a':
                cur[0] -= step
                modified_this = True
            elif key == 'w':
                cur[1] += step
                modified_this = True
            elif key == 's':
                cur[1] -= step
                modified_this = True
            elif key == 'e':
                cur[2] += step
                modified_this = True
            elif key == 'd':
                cur[2] -= step
                modified_this = True
            elif key == 'r':
                cur[3] += step
                modified_this = True
            elif key == 'f':
                cur[3] -= step
                modified_this = True
            elif key == 't':
                cur[4] += step
                modified_this = True
            elif key == 'g':
                cur[4] -= step
                modified_this = True
            elif key == 'y':
                cur[5] += step
                modified_this = True
            elif key == 'h':
                cur[5] -= step
                modified_this = True
            elif key == 'o':
                cur[6] = open_grip
                modified_this = True
            elif key == 'c':
                cur[6] = close_grip
                modified_this = True
            elif key == 'p':
                pass  # pose is printed on next loop
            elif key == 'l':
                # Test lift: read lift_pose from waypoints, offset to current position
                sys.stdout.write("\n")
                sys.stdout.flush()
                lift_wp = np.array(waypoints.get("lift_pose", cur.tolist()), dtype=np.float32)
                cur_hold = cur.copy()
                print(f"  Testing lift → {fmt_vec(lift_wp)}")
                from scripts.quick_bottle_grasp import interpolate_joint_path as _interp_lift
                lift_path = interpolate_joint_path(cur, lift_wp, 0.03, 0.002)
                for lt in lift_path:
                    lt[:6] = np.clip(lt[:6], -3.14, 3.14)
                    lt[6] = np.clip(lt[6], 0.0, PIPER_GRIPPER_MAX_M)
                    robot.set_joint_positions(lt.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                time.sleep(1.0)
                # Return to pre-lift position
                ret_path = interpolate_joint_path(lift_path[-1] if lift_path else cur_hold,
                                                   cur_hold, 0.03, 0.002)
                for rt in ret_path:
                    rt[:6] = np.clip(rt[:6], -3.14, 3.14)
                    rt[6] = np.clip(rt[6], 0.0, PIPER_GRIPPER_MAX_M)
                    robot.set_joint_positions(rt.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                cur = cur_hold.copy()
            elif key == 'A':
                waypoints["approach_pose"] = cur.tolist().copy()
                waypoints["approach_pose"][6] = open_grip
                print(f"\n  Saved approach_pose: {fmt_vec(waypoints['approach_pose'])}")
                modified = True
            elif key == 'C':
                waypoints["close_gripper_pose"] = cur.tolist().copy()
                waypoints["close_gripper_pose"][6] = close_grip
                print(f"\n  Saved close_gripper_pose: {fmt_vec(waypoints['close_gripper_pose'])}")
                modified = True
            elif key == 'L':
                waypoints["lift_pose"] = cur.tolist().copy()
                waypoints["lift_pose"][6] = close_grip
                print(f"\n  Saved lift_pose: {fmt_vec(waypoints['lift_pose'])}")
                modified = True
            elif key == 'x' or key == 'ESC':
                break

            if modified_this:
                cur[:6] = np.clip(cur[:6], -3.14, 3.14)
                cur[6] = np.clip(cur[6], 0.0, PIPER_GRIPPER_MAX_M)
                robot.set_joint_positions(cur.tolist(), velocity_pct=args.velocity_pct)

    if not modified:
        print("\n  No changes made.")
        return None

    return waypoints


def backup_path(filepath: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return filepath.with_name(f"{filepath.stem}.backup_{ts}{filepath.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Waypoint replay bottle grasp (no model)")
    parser.add_argument("--waypoints", type=Path,
                        default=PROJECT_ROOT / "configs" / "bottle_grasp_waypoints.json")
    parser.add_argument("--mode", type=str, choices=("grasp", "pick_place", "auto"), default="auto",
                        help="Replay mode: grasp (5-phase), pick_place (10-phase), or auto-detect.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned trajectory without sending robot commands.")
    parser.add_argument("--step-confirm", action="store_true",
                        help="Wait for SPACE before each phase.")
    parser.add_argument("--hz", type=float, default=30.0,
                        help="Control loop frequency.")
    parser.add_argument("--max-step-rad", type=float, default=0.03,
                        help="Max joint change per step (J1-J6).")
    parser.add_argument("--max-step-gripper", type=float, default=0.004,
                        help="Max gripper change per step.")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="Multiplier for step sizes (1.0=normal, 0.5=slow, 2.0=fast).")
    parser.add_argument("--velocity-pct", type=int, default=40,
                        help="Piper velocity percent (1-100). Higher = faster but more overshoot.")
    parser.add_argument("--settle-seconds", type=float, default=0.3,
                        help="Wait time after each phase for the arm to reach target.")
    parser.add_argument("--gripper-close-wait", type=float, default=1.0,
                        help="Extra wait time after close_gripper phase for gripper to engage.")
    parser.add_argument("--hold-seconds", type=float, default=1.0,
                        help="Hold time at final pose before disconnecting.")
    parser.add_argument("--post-close-dwell", type=float, default=1.0,
                        help="Extra dwell after close_gripper phase for gripper to engage.")
    parser.add_argument("--post-place-dwell", type=float, default=0.3,
                        help="Extra dwell after place phase for stability.")
    parser.add_argument("--post-release-dwell", type=float, default=0.7,
                        help="Extra dwell after release phase for gripper to open.")
    parser.add_argument("--open-gripper", type=float, default=None,
                        help="Override open_gripper value from waypoints JSON.")
    parser.add_argument("--disable", action="store_true",
                        help="Disable arm after trajectory completes (default: keep enabled).")
    parser.add_argument("--open-gripper-before", action="store_true",
                        help="Open gripper to waypoints[open_gripper] before moving arm.")
    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--log-result", action="store_true",
                        help="Log replay result to CSV.")
    parser.add_argument("--result-log", type=Path,
                        default=PROJECT_ROOT / "logs" / "replay_results.csv",
                        help="CSV path for result logging (default: logs/replay_results.csv).")
    parser.add_argument("--tune-grasp", action="store_true",
                        help="After approach, enter jog mode to micro-adjust grasp point.")
    parser.add_argument("--jog-step", type=float, default=0.01,
                        help="Joint jog step size in rad (default: 0.01).")
    parser.add_argument("--jog-step-fine", type=float, default=0.003,
                        help="Fine jog step size with Shift held (default: 0.003).")
    parser.add_argument("--gripper-jog-step", type=float, default=0.004,
                        help="Gripper jog step size in m (default: 0.004).")
    args = parser.parse_args()

    waypoints = json.loads(args.waypoints.read_text())
    src = waypoints.get("source", waypoints.get("source_episode", "?"))
    created = waypoints.get("created_at", "")
    open_grip = args.open_gripper if args.open_gripper is not None else waypoints.get("open_gripper", 0.08)
    close_grip = waypoints.get("close_gripper", 0.0)
    mode = detect_mode(waypoints, args.mode)

    # Apply speed scale
    max_step_rad = args.max_step_rad * args.speed_scale
    max_step_gripper = args.max_step_gripper * args.speed_scale

    print("=" * 60)
    if mode == "pick_place":
        print("  Quick Pick-and-Place — Waypoint Replay")
    else:
        print("  Quick Bottle Grasp — Waypoint Replay")
    print(f"  Source:    {src}")
    if created:
        print(f"  Created:   {created}")
    print(f"  Mode:      {mode}")
    print("=" * 60)
    print(f"  Waypoints: {args.waypoints}")
    print(f"  Max step rad: {max_step_rad:.4f}  Max step grip: {max_step_gripper:.4f}")
    print(f"  Open gripper: {open_grip:.4f}  Close gripper: {close_grip:.4f}")
    if args.dry_run:
        print("  DRY RUN — no robot commands will be sent.")
    if args.step_confirm:
        print("  STEP CONFIRM — press SPACE before each phase.")

    # Show waypoint preview
    if mode == "pick_place":
        preview_keys = [pp[0] for pp in PICK_PLACE_PHASES]
        preview_names = [pp[1] for pp in PICK_PLACE_PHASES]
    else:
        preview_keys = WAYPOINT_KEYS
        preview_names = PHASE_NAMES
    print("\n  Planned phases:")
    for pose_key, phase_name in zip(preview_keys, preview_names):
        if pose_key not in waypoints:
            print(f"    {phase_name:>16}: [MISSING]")
            continue
        wp = waypoints[pose_key]
        print(f"    {phase_name:>16}: {fmt_vec(wp)}")

    # --- Connect robot ---
    robot = None
    if not args.dry_run:
        print(f"\n[1/2] Connecting Piper ({args.can_port}) ...")
        from hardware.piper_wrapper import PiperRobot
        robot = PiperRobot(can_port=args.can_port)
        robot.connect()  # connect + enable in one call
        print("  Robot connected and enabled.")
    else:
        print("\n[1/2] DRY RUN — skipping robot connection.")

    try:
        # --- Build phase list ---
        if args.open_gripper_before and mode == "grasp":
            # Legacy: force open at start for grasp mode
            open_grip_override = open_grip
        phases = build_phase_list(waypoints, open_grip, close_grip, mode)

        current = None
        if robot is not None:
            current = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            saved_home = current.copy()  # actual starting position
        else:
            current = np.array(phases[0][1], dtype=np.float32)
            saved_home = current.copy()

        # For pick_place, use home_pose from waypoints as return target
        if mode == "pick_place" and "home_pose" in waypoints:
            home_pose = np.array(waypoints["home_pose"], dtype=np.float32)
        else:
            home_pose = saved_home.copy()

        print(f"\n[2/2] Current robot state: {fmt_vec(current)}")
        print(f"      (Will return to home: {fmt_vec(home_pose)} before disabling)")
        print("-" * 60)
        if args.step_confirm:
            print("  Press ENTER to start each phase, Q then ENTER to quit.")
        print("  Ctrl+C to emergency stop at any time.")
        print()

        phase_errors = {}
        final_gripper = None
        for phase_name, target in phases:
            if not safety_check_joints(target):
                print(f"  [ABORT] Safety check failed for '{phase_name}'")
                break

            n_interp = len(interpolate_joint_path(current, target,
                                                  max_step_rad, max_step_gripper))
            print(f"\n  >>> Phase: {phase_name}")
            print(f"      Target: {fmt_vec(target)}  "
                  f"(~{n_interp} steps at ~{1.0/args.hz:.3f}s each → ~{n_interp/args.hz:.1f}s)")

            skip_arm = args.dry_run
            if args.dry_run:
                pass  # already printing
            else:
                pass  # execute below

            if args.step_confirm:
                result = wait_for_input("      Press ENTER to execute, Q then ENTER to quit: ")
                if result == 'quit':
                    print("\n  User quit.")
                    return 0

            # --- Execute interpolated path ---
            interp_path = interpolate_joint_path(current, target,
                                                 max_step_rad, max_step_gripper)
            for step_i, interp_target in enumerate(interp_path):
                loop_start = time.time()

                if not args.dry_run:
                    robot.set_joint_positions(interp_target.tolist(), velocity_pct=args.velocity_pct)

                # Print progress every 10 steps or at boundaries
                if step_i == 0 or step_i == len(interp_path) - 1 or (step_i + 1) % 10 == 0:
                    progress = (step_i + 1) / len(interp_path) * 100
                    print(f"      [{step_i+1:4d}/{len(interp_path):4d}  {progress:5.1f}%]  "
                          f"arm={fmt_vec(interp_target[:6], 3)}  grip={interp_target[6]:.4f}")

                elapsed = time.time() - loop_start
                step_time = 1.0 / args.hz
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)

            # Read back actual position (non-dry-run)
            if not args.dry_run:
                time.sleep(args.settle_seconds)
                current = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                # Dwell after certain phases (close_gripper, release, place)
                dwell_name = DWELL_PHASES.get(phase_name)
                dwell_sec = getattr(args, dwell_name, None) if dwell_name else None
                if dwell_sec is None and phase_name == "close_gripper":
                    dwell_sec = args.gripper_close_wait  # backward compat
                if dwell_sec and dwell_sec > 0:
                    print(f"      Dwelling {dwell_sec:.1f}s after {phase_name} ...")
                    time.sleep(dwell_sec)
                    hold_target = current.copy()
                    hold_target[6] = target[6]
                    for _ in range(int(dwell_sec * args.hz)):
                        robot.set_joint_positions(hold_target.tolist(), velocity_pct=args.velocity_pct)
                        time.sleep(1.0 / args.hz)
                    current = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            else:
                current = target.copy()

            # Show error
            arm_err = np.max(np.abs(current[:6] - target[:6]))
            grip_err = abs(current[6] - target[6])
            print(f"      Actual: {fmt_vec(current)}")
            print(f"      Error:  arm_max={arm_err:.4f} rad  grip={grip_err:.4f} m")
            phase_errors[phase_name] = {"arm_err": float(arm_err), "grip_err": float(grip_err)}
            if phase_name == "lift":
                final_gripper = float(current[6])

            # ── Tune-grasp: enter jog mode after approach ──
            if args.tune_grasp and phase_name == "approach":
                updated = tune_grasp_jog(robot, waypoints, args)
                if updated is not None:
                    # Backup and save
                    if args.waypoints.exists():
                        bu = backup_path(args.waypoints)
                        shutil.copy2(args.waypoints, bu)
                        print(f"\n  Backup: {bu}")
                    args.waypoints.write_text(json.dumps(updated, indent=2) + "\n")
                    print(f"  Updated waypoints saved → {args.waypoints}")
                    # Rebuild phases with new waypoints
                    phases = build_phase_list(updated, open_grip, close_grip, mode)
                    # Skip remaining old phases; re-execute from close_gripper
                    ans = input("  Continue with close_gripper + lift? [Y/n]: ").strip().lower()
                    if ans == 'n':
                        print("  Stopping after tune-grasp.")
                        break
                    # Skip start, pre_grasp, approach (already done) — run only close + lift
                    for phase_name2, target2 in phases:
                        if phase_name2 in ("start", "pre_grasp", "approach"):
                            continue
                        # Re-execute close_gripper and lift
                        if not safety_check_joints(target2):
                            print(f"  [ABORT] Safety check failed for '{phase_name2}'")
                            break
                        n_interp2 = len(interpolate_joint_path(current, target2,
                                                               max_step_rad, max_step_gripper))
                        print(f"\n  >>> Phase: {phase_name2}")
                        print(f"      Target: {fmt_vec(target2)}  "
                              f"(~{n_interp2} steps → ~{n_interp2/args.hz:.1f}s)")
                        if args.step_confirm:
                            result = wait_for_input("      Press ENTER to execute, Q then ENTER to quit: ")
                            if result == 'quit':
                                print("\n  User quit.")
                                break
                        interp_path2 = interpolate_joint_path(current, target2,
                                                               max_step_rad, max_step_gripper)
                        for step_i2, interp_t2 in enumerate(interp_path2):
                            loop_start2 = time.time()
                            if not args.dry_run:
                                robot.set_joint_positions(interp_t2.tolist(),
                                                         velocity_pct=args.velocity_pct)
                            if step_i2 == 0 or step_i2 == len(interp_path2) - 1 or (step_i2 + 1) % 10 == 0:
                                progress2 = (step_i2 + 1) / len(interp_path2) * 100
                                print(f"      [{step_i2+1:4d}/{len(interp_path2):4d}  {progress2:5.1f}%]  "
                                      f"arm={fmt_vec(interp_t2[:6], 3)}  grip={interp_t2[6]:.4f}")
                            elapsed2 = time.time() - loop_start2
                            step_t2 = 1.0 / args.hz
                            if elapsed2 < step_t2:
                                time.sleep(step_t2 - elapsed2)
                        if not args.dry_run:
                            time.sleep(args.settle_seconds)
                            current = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                            if phase_name2 == "close_gripper":
                                print(f"      Waiting {args.gripper_close_wait:.1f}s for gripper to close ...")
                                time.sleep(args.gripper_close_wait)
                                hold_target = current.copy()
                                hold_target[6] = target2[6]
                                for _ in range(int(args.gripper_close_wait * args.hz)):
                                    robot.set_joint_positions(hold_target.tolist(),
                                                             velocity_pct=args.velocity_pct)
                                    time.sleep(1.0 / args.hz)
                                current = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                        else:
                            current = target2.copy()
                        arm_err2 = np.max(np.abs(current[:6] - target2[:6]))
                        grip_err2 = abs(current[6] - target2[6])
                        print(f"      Actual: {fmt_vec(current)}")
                        print(f"      Error:  arm_max={arm_err2:.4f} rad  grip={grip_err2:.4f} m")
                        phase_errors[phase_name2] = {"arm_err": float(arm_err2), "grip_err": float(grip_err2)}
                        if phase_name2 == "lift":
                            final_gripper = float(current[6])
                break  # Exit original phase loop

        # --- Hold ---
        print(f"\n  >>> Holding for {args.hold_seconds:.1f}s ..."
              f"  (Ctrl+C to stop)")
        hold_start = time.time()
        while time.time() - hold_start < args.hold_seconds:
            if not args.dry_run and robot:
                robot.set_joint_positions(current.tolist(), velocity_pct=25)
            time.sleep(0.1)

        # --- Return to home ---
        skip_return = False
        print(f"\n  >>> Return to home: {fmt_vec(home_pose)}")
        if args.step_confirm:
            result = wait_for_input("      Press ENTER to return home, Q then ENTER to quit: ")
            if result == 'quit':
                print("  Skipping return, going to shutdown.")
                skip_return = True

        if not skip_return:
            return_path = interpolate_joint_path(current, home_pose,
                                                  max_step_rad, max_step_gripper)
            for ri, rt in enumerate(return_path):
                t_start = time.time()
                rt[:6] = np.clip(rt[:6], -3.14, 3.14)
                rt[6] = np.clip(rt[6], 0.0, PIPER_GRIPPER_MAX_M)
                if not args.dry_run:
                    robot.set_joint_positions(rt.tolist(), velocity_pct=args.velocity_pct)
                if ri == 0 or ri == len(return_path) - 1 or (ri + 1) % 10 == 0:
                    print(f"      [{ri+1:4d}/{len(return_path):4d}]  "
                          f"arm={fmt_vec(rt[:6], 3)}  grip={rt[6]:.4f}")
                elapsed = time.time() - t_start
                s_time = 1.0 / args.hz
                if elapsed < s_time:
                    time.sleep(s_time - elapsed)

            if not args.dry_run:
                time.sleep(args.settle_seconds)
                current = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                err = np.max(np.abs(current - home_pose))
                print(f"      Home position error: {err:.4f}")
        print("\n  Trajectory complete.")

        # ── Result logging ──
        if args.log_result:
            outcome = "unknown"
            if not args.dry_run:
                q = "  Did pick-and-place succeed? [y/n/u]: " if mode == "pick_place" else "  Did the grasp succeed? [y/n/u]: "
                ans = input(q).strip().lower()
                outcome = {"y": "success", "n": "fail", "u": "unknown"}.get(ans, "unknown")
            else:
                print("  [dry-run] Result: unknown")

            args.result_log.parent.mkdir(parents=True, exist_ok=True)
            file_exists = args.result_log.exists()
            with open(args.result_log, "a", newline="") as f:
                writer = csv.writer(f)
                final_state_str = fmt_vec(current) if final_gripper is not None else ""
                if not file_exists:
                    if mode == "pick_place":
                        writer.writerow([
                            "timestamp", "waypoints_file", "mode", "velocity_pct",
                            "outcome", "user_note", "final_state",
                            "approach_arm_err", "approach_grip_err",
                            "close_arm_err", "close_grip_err",
                            "lift_arm_err", "lift_grip_err",
                            "place_arm_err", "place_grip_err",
                            "release_arm_err", "release_grip_err",
                            "final_gripper",
                        ])
                    else:
                        writer.writerow([
                            "timestamp", "waypoints_file", "velocity_pct",
                            "outcome", "user_note",
                            "approach_arm_err", "approach_grip_err",
                            "close_arm_err", "close_grip_err",
                            "lift_arm_err", "lift_grip_err",
                            "final_gripper",
                        ])
                note = input("  Note (optional): ").strip()
                if mode == "pick_place":
                    writer.writerow([
                        datetime.now().isoformat(),
                        str(args.waypoints), mode, args.velocity_pct,
                        outcome, note, final_state_str,
                        f"{phase_errors.get('approach', {}).get('arm_err', float('nan')):.4f}",
                        f"{phase_errors.get('approach', {}).get('grip_err', float('nan')):.4f}",
                        f"{phase_errors.get('close_gripper', {}).get('arm_err', float('nan')):.4f}",
                        f"{phase_errors.get('close_gripper', {}).get('grip_err', float('nan')):.4f}",
                        f"{phase_errors.get('lift', {}).get('arm_err', float('nan')):.4f}",
                        f"{phase_errors.get('lift', {}).get('grip_err', float('nan')):.4f}",
                        f"{phase_errors.get('place', {}).get('arm_err', float('nan')):.4f}",
                        f"{phase_errors.get('place', {}).get('grip_err', float('nan')):.4f}",
                        f"{phase_errors.get('release', {}).get('arm_err', float('nan')):.4f}",
                        f"{phase_errors.get('release', {}).get('grip_err', float('nan')):.4f}",
                        f"{final_gripper:.4f}" if final_gripper is not None else "",
                    ])
                else:
                    writer.writerow([
                        datetime.now().isoformat(),
                        str(args.waypoints),
                        args.velocity_pct,
                        outcome,
                        note,
                        f"{phase_errors.get('approach', {}).get('arm_err', float('nan')):.4f}",
                        f"{phase_errors.get('approach', {}).get('grip_err', float('nan')):.4f}",
                        f"{phase_errors.get('close_gripper', {}).get('arm_err', float('nan')):.4f}",
                        f"{phase_errors.get('close_gripper', {}).get('grip_err', float('nan')):.4f}",
                        f"{phase_errors.get('lift', {}).get('arm_err', float('nan')):.4f}",
                        f"{phase_errors.get('lift', {}).get('grip_err', float('nan')):.4f}",
                        f"{final_gripper:.4f}" if final_gripper is not None else "",
                    ])
            print(f"  Result logged → {args.result_log}")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        if args.disable:
            print("  Shutting down ...")
            if robot is not None:
                robot.disable()
                robot.disconnect()
            print("  Done.")
        else:
            print("  Trajectory complete. Arm stays ENABLED.")


if __name__ == "__main__":
    main()
