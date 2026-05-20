#!/usr/bin/env python3
"""Record a continuous joint trajectory and auto-extract waypoints.

Like data collection, but no cameras, no dataset — just joints.
Move the arm through the full trajectory in one continuous motion.

USAGE:
  # Grasp only (5 poses)
  python3 scripts/record_waypoint_trajectory.py --mode grasp

  # Full pick-and-place (10 poses)
  python3 scripts/record_waypoint_trajectory.py --mode pick_place

Flow:
  1. Arm is enabled.
  2. Move to START pose, press SPACE to begin recording.
  3. Move through the full trajectory.
  4. Press SPACE again to stop (or arm auto-stops after 5s of no movement).
  5. Script auto-extracts waypoints and saves JSON.
  6. Arm returns to start pose before disabling.

Then replay:
  python3 scripts/quick_bottle_grasp.py \\
      --waypoints configs/bottle_grasp_waypoints_today.json \\
      --step-confirm --velocity-pct 50
"""
from __future__ import annotations

import argparse
import json
import os
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


def fmt_vec(v, precision=3):
    return "[" + ", ".join(f"{float(x):.{precision}f}" for x in v) + "]"


# ── raw terminal ──

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
        import select
        if select.select([sys.stdin], [], [], timeout)[0]:
            b = sys.stdin.read(1)
            if b == '\x1b':
                return 'ESC'
            if b == '\x03':
                raise KeyboardInterrupt
            return b if ord(b) >= 32 else ''
        return ''


def backup_path(filepath: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return filepath.with_name(f"{filepath.stem}.backup_{ts}{filepath.suffix}")


def max_movement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def extract_waypoints(states: np.ndarray):
    """Auto-detect 5 waypoints using BOTH gripper state and arm velocity.

    Key idea: don't just look at gripper — also check where the arm is actually
    moving between phases (descent, grasp, lift).

    states: (N, 7) — [j1..j6, gripper]
    """
    n = len(states)
    grip = states[:, 6]
    arm = states[:, :6]

    # Per-frame arm velocity (max joint delta)
    arm_vel = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        arm_vel[i] = float(np.max(np.abs(arm[i] - arm[i - 1])))

    # Smooth with 5-frame box filter
    kernel = np.ones(5) / 5.0
    arm_vel_smooth = np.convolve(arm_vel, kernel, mode="same")

    # ── start ──
    start_idx = 0

    # ── gripper events (raw, for reference) ──
    open_start = n - 1
    for i in range(n):
        if grip[i] > 0.01:
            open_start = i
            break

    max_grip_idx = int(np.argmax(grip))

    close_start = max_grip_idx
    for i in range(max_grip_idx, min(max_grip_idx + 50, n - 1)):
        if grip[i] - grip[i + 1] > 0.003:
            close_start = i
            break

    # close_end: grip stops changing (plateau) AFTER dropping from max = bottle contact.
    # Don't use absolute threshold — bottle prevents closing to 0.
    close_end = n - 1
    grip_drop_threshold = 0.015  # grip must drop at least this much from max
    for i in range(close_start + 5, n - 3):
        grip_dropped = (grip[max_grip_idx] - grip[i]) > grip_drop_threshold
        grip_plateau = abs(grip[i + 3] - grip[i - 3]) < 0.0008
        if grip_dropped and grip_plateau:
            close_end = i
            break

    # ── pre_grasp: stable arm pose BEFORE gripper opens ──
    # Look backwards from open_start to find a distinct arm configuration
    # where the arm was holding still (low velocity).
    pre_grasp_idx = max(0, open_start - 10)  # fallback
    for i in range(open_start - 5, max(0, open_start - 200), -1):
        if arm_vel_smooth[i] < 0.0015:
            # Make sure arm at this frame is different from start
            if max_movement(arm[i], arm[start_idx]) > 0.5:
                pre_grasp_idx = i
                break

    # ── approach: arm at bottle level, gripper open ──
    # Look in [open_start, close_start] for the frame with max J2 (arm most
    # extended forward, i.e. descended to bottle), with low velocity (stable).
    approach_idx = max_grip_idx  # fallback
    search_a = max(open_start, pre_grasp_idx + 5)
    search_b = min(close_start + 1, n)
    if search_b > search_a:
        # Prefer stable frame with high J2 (forward reach) and open gripper
        best_score = -1.0
        for i in range(search_a, search_b):
            if grip[i] < 0.01 or arm_vel_smooth[i] > 0.005:
                continue
            score = arm[i, 1] - arm_vel_smooth[i] * 50  # high J2, low velocity
            if score > best_score:
                best_score = score
                approach_idx = i

    # ── close_gripper: where gripper actually stopped closing (bottle contact) ──
    close_gripper_idx = close_end

    # ── lift: arm has moved UP from grasp position ──
    # Look after close_end for a velocity surge (arm lifting), take the
    # frame where the surge peaks.
    lift_idx = min(close_end + 40, n - 1)  # fallback
    search_l = close_end
    search_r = n
    if search_r > search_l + 10:
        vel_seg = arm_vel_smooth[search_l:search_r]
        peak = int(np.argmax(vel_seg))
        if vel_seg[peak] > 0.002:
            # Take a frame slightly after the velocity peak (arm at top)
            lift_idx = min(search_l + peak + 5, n - 1)
        else:
            # No clear velocity peak — look for most changed arm position
            lift_idx = min(close_end + 40, n - 1)

    # ── sanity: ensure monotonic ordering ──
    pre_grasp_idx = max(0, min(pre_grasp_idx, open_start - 2))
    approach_idx = max(pre_grasp_idx + 1, min(approach_idx, close_start))
    close_gripper_idx = max(approach_idx + 1, min(close_gripper_idx, close_end))
    lift_idx = max(close_gripper_idx + 1, min(lift_idx, n - 1))

    warnings = []
    # Check if gripper actually plateaued (hit bottle) vs still moving at end
    if close_end >= n - 5:
        tail_change = abs(grip[-1] - grip[max(0, n - 10)])
        if tail_change < 0.001:
            # Gripper plateaued — likely hit bottle, OK
            pass
        else:
            warnings.append(
                "gripper still moving at end of recording; hold close + lift longer"
            )
    if grip[close_gripper_idx] < 0.005:
        warnings.append(
            f"close_gripper grip={grip[close_gripper_idx]:.4f} is nearly zero — "
            "bottle may not be in gripper"
        )
    lift_arm_delta = max_movement(arm[lift_idx], arm[close_gripper_idx])
    if lift_arm_delta < 0.15:
        warnings.append(
            f"lift pose is too close to close_gripper pose (max arm delta={lift_arm_delta:.3f} rad)"
        )

    result = {
        "start_pose":            [float(x) for x in states[start_idx]],
        "pre_grasp_pose":        [float(x) for x in states[pre_grasp_idx]],
        "approach_pose":         [float(x) for x in states[approach_idx]],
        "close_gripper_pose":    [float(x) for x in states[close_gripper_idx]],
        "lift_pose":             [float(x) for x in states[lift_idx]],
        "_frames": {
            "start": start_idx,
            "pre_grasp": pre_grasp_idx,
            "approach": approach_idx,
            "close_gripper": close_gripper_idx,
            "lift": lift_idx,
            "gripper_open_start": int(open_start),
            "gripper_close_start": int(close_start),
            "gripper_close_end": int(close_end),
            "total_frames": n,
        },
        "_warnings": warnings,
    }
    return result


def extract_pick_place_waypoints(states: np.ndarray):
    """Auto-detect 10 waypoints for full pick-and-place.

    Phases: start → pre_grasp → approach(open) → close_gripper(close) → lift
            → place_pre → place → release(open) → retreat → home

    Uses gripper state and arm velocity to detect transitions.
    """
    n = len(states)
    grip = states[:, 6]
    arm = states[:, :6]

    arm_vel = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        arm_vel[i] = float(np.max(np.abs(arm[i] - arm[i - 1])))
    kernel = np.ones(5) / 5.0
    arm_vel_smooth = np.convolve(arm_vel, kernel, mode="same")

    # ── start ──
    start_idx = 0

    # ── Find gripper events ──
    # First time gripper opens (for approach)
    open_start = n - 1
    for i in range(n):
        if grip[i] > 0.01:
            open_start = i
            break
    max_grip_idx = int(np.argmax(grip[:min(open_start + 300, n)])) if open_start < n else 0

    # First close start: grip begins dropping from max
    # Wider window for pick-and-place (gripper open during full descent)
    close_start = max_grip_idx
    for i in range(max_grip_idx, min(max_grip_idx + 300, n - 1)):
        if grip[i] - grip[i + 1] > 0.003:
            close_start = i
            break

    # First close end: grip plateaus after dropping (bottle contact prevents full close)
    # Limit search to close_start window — don't search into lift phase
    close_end = min(close_start + 200, n - 1)
    grip_drop_threshold = 0.015
    for i in range(close_start + 5, min(close_start + 200, n - 3)):
        grip_dropped = (grip[max_grip_idx] - grip[i]) > grip_drop_threshold
        grip_plateau = abs(grip[i + 3] - grip[i - 3]) < 0.0008
        if grip_dropped and grip_plateau:
            close_end = i
            break

    # ── pre_grasp: stable arm BEFORE gripper opens ──
    pre_grasp_idx = max(0, open_start - 10)
    for i in range(open_start - 5, max(0, open_start - 200), -1):
        if arm_vel_smooth[i] < 0.0015:
            if max_movement(arm[i], arm[start_idx]) > 0.5:
                pre_grasp_idx = i
                break

    # ── approach: max J2 while gripper open, before close ──
    approach_idx = max_grip_idx
    search_a = max(open_start, pre_grasp_idx + 5)
    search_b = min(close_start + 1, n)
    if search_b > search_a:
        best_score = -1.0
        for i in range(search_a, search_b):
            if grip[i] < 0.01 or arm_vel_smooth[i] > 0.005:
                continue
            score = arm[i, 1] - arm_vel_smooth[i] * 50
            if score > best_score:
                best_score = score
                approach_idx = i

    # ── close_gripper: gripper plateau ──
    close_gripper_idx = close_end

    # ── lift: FIRST significant arm velocity peak after close_end ──
    # Use first peak (not global max) because pick-and-place has multiple
    # velocity surges (lift, place descent, retreat).
    lift_idx = min(close_end + 40, n - 1)
    for i in range(close_end + 10, n - 5):
        if arm_vel_smooth[i] > 0.003:
            # Found first velocity surge — take frame slightly after peak
            peak = i
            for j in range(i, min(i + 60, n - 1)):
                if arm_vel_smooth[j] > arm_vel_smooth[peak]:
                    peak = j
                if arm_vel_smooth[j] < 0.002 and j > peak + 10:
                    break
            lift_idx = min(peak + 5, n - 1)
            break

    # ── Second gripper open (release) ──
    # After close, gripper sits at ~0.05 (bottle width). Release opens it to ~0.10.
    # Detect significant RISE in grip (not absolute threshold).
    release_start = n - 1
    for i in range(lift_idx + 20, n - 3):
        # Grip rising: current is notably higher than a few frames ago
        grip_rising = grip[i] - grip[max(0, i - 5)] > 0.008
        grip_recently_closed = grip[max(0, i - 15)] < 0.07
        if grip_rising and grip_recently_closed:
            release_start = i
            break

    # release stable: after release_start, find first stable open grip
    release_idx = min(release_start + 10, n - 1)
    for i in range(release_start, min(release_start + 100, n)):
        if grip[i] > 0.05 and arm_vel_smooth[i] < 0.003:
            release_idx = i
            break

    # ── place: arm J2 peak between lift and release, gripper still closed (~0.05 with bottle) ──
    place_idx = (lift_idx + release_start) // 2
    search_p_start = lift_idx + 10
    search_p_end = release_start - 5
    if search_p_end > search_p_start:
        best_score = -1.0
        for i in range(search_p_start, search_p_end):
            # Gripper should still be closed (bottle in gripper → ~0.05), not yet released
            if grip[i] > 0.07 or arm_vel_smooth[i] > 0.005:
                continue
            score = arm[i, 1] - arm_vel_smooth[i] * 50
            if score > best_score:
                best_score = score
                place_idx = i

    # ── place_pre: stable arm before place descent, gripper still closed ──
    place_pre_idx = lift_idx + 10
    for i in range(lift_idx + 5, place_idx - 3):
        if arm_vel_smooth[i] < 0.002 and grip[i] < 0.07:
            place_pre_idx = i
            break

    # ── retreat: arm stable after release, away from place ──
    retreat_idx = min(release_idx + 20, n - 1)
    for i in range(release_idx + 5, min(release_idx + 150, n)):
        if arm_vel_smooth[i] < 0.002:
            retreat_idx = i
            break

    # ── home: final stable pose ──
    home_idx = n - 1
    for i in range(n - 5, retreat_idx + 10, -1):
        if arm_vel_smooth[i] < 0.0015:
            home_idx = i
            break

    # ── sanity ordering ──
    pre_grasp_idx = max(0, min(pre_grasp_idx, open_start - 2))
    approach_idx = max(pre_grasp_idx + 1, min(approach_idx, close_start))
    close_gripper_idx = max(approach_idx + 1, min(close_gripper_idx, close_end))
    lift_idx = max(close_gripper_idx + 1, min(lift_idx, place_pre_idx - 2))
    place_pre_idx = max(lift_idx + 1, min(place_pre_idx, place_idx - 2))
    place_idx = max(place_pre_idx + 1, min(place_idx, release_start - 2))
    release_idx = max(place_idx + 1, min(release_idx, retreat_idx - 2))
    retreat_idx = max(release_idx + 1, min(retreat_idx, home_idx - 2))
    home_idx = max(retreat_idx + 1, home_idx)

    warnings = []
    if close_end >= n - 5:
        tail_change = abs(grip[-1] - grip[max(0, n - 10)])
        if tail_change > 0.001:
            warnings.append("gripper still moving at end; hold final pose longer")
    if grip[close_gripper_idx] > 0.08:
        warnings.append(f"close_gripper grip={grip[close_gripper_idx]:.4f} seems high; bottle may not be grasped (expected ~0.05)")
    if release_idx >= n - 10:
        warnings.append("release detected very late; make sure to open gripper during recording")
    lift_arm_delta = max_movement(arm[lift_idx], arm[close_gripper_idx])
    if lift_arm_delta < 0.15:
        warnings.append(f"lift too close to close_gripper (delta={lift_arm_delta:.3f} rad)")

    result = {
        "start_pose":            [float(x) for x in states[start_idx]],
        "pre_grasp_pose":        [float(x) for x in states[pre_grasp_idx]],
        "approach_pose":         [float(x) for x in states[approach_idx]],
        "close_gripper_pose":    [float(x) for x in states[close_gripper_idx]],
        "lift_pose":             [float(x) for x in states[lift_idx]],
        "place_pre_pose":        [float(x) for x in states[place_pre_idx]],
        "place_pose":            [float(x) for x in states[place_idx]],
        "release_pose":          [float(x) for x in states[release_idx]],
        "retreat_pose":          [float(x) for x in states[retreat_idx]],
        "home_pose":             [float(x) for x in states[home_idx]],
        "_frames": {
            "start": start_idx,
            "pre_grasp": pre_grasp_idx,
            "approach": approach_idx,
            "close_gripper": close_gripper_idx,
            "lift": lift_idx,
            "place_pre": place_pre_idx,
            "place": place_idx,
            "release": release_idx,
            "retreat": retreat_idx,
            "home": home_idx,
            "gripper_open_start": int(open_start),
            "gripper_close_start": int(close_start),
            "gripper_close_end": int(close_end),
            "gripper_release_start": int(release_start),
            "total_frames": n,
        },
        "_warnings": warnings,
    }
    return result


def interpolate_joint_path(start, target, max_step_rad=0.03, max_step_gripper=0.004):
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


def main():
    parser = argparse.ArgumentParser(description="Record continuous trajectory, extract waypoints")
    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "configs" / "bottle_grasp_waypoints_today.json")
    parser.add_argument("--trajectory-output", type=Path, default=None,
                        help="Also save full recorded trajectory as CSV")
    parser.add_argument("--open-gripper", type=float, default=0.10)
    parser.add_argument("--close-gripper", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--auto-stop-sec", type=float, default=5.0,
                        help="Auto-stop if no significant arm movement for this many seconds")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manual-waypoints", action="store_true",
                        help="Skip auto-extraction, manually select waypoints after recording")
    parser.add_argument("--mode", choices=("grasp", "pick_place"), default="grasp",
                        help="Recording mode: grasp (5 poses) or pick_place (10 poses)")
    parser.add_argument("--return-to-start", action="store_true", default=True,
                        help="Return arm to start pose before disabling (default: on)")
    parser.add_argument("--no-return", action="store_true",
                        help="Skip return-to-start")
    args = parser.parse_args()

    print("=" * 60)
    print("  Piper Continuous Trajectory Recorder")
    print("=" * 60)
    print(f"  Mode:   {args.mode}")
    print(f"  Output: {args.output}")
    print()

    from hardware.piper_wrapper import PiperRobot
    robot = PiperRobot(can_port=args.can_port)
    robot.connect()  # connect + enable in one call
    print("  Arm ENABLED.")

    if args.mode == "pick_place":
        traj_desc = (
            "start → pre_grasp → approach(open) → close_gripper(close) → lift → "
            "place_pre → place → release(open) → retreat → home"
        )
        pose_keys = [
            "start_pose", "pre_grasp_pose", "approach_pose",
            "close_gripper_pose", "lift_pose",
            "place_pre_pose", "place_pose", "release_pose",
            "retreat_pose", "home_pose",
        ]
    else:
        traj_desc = "pre_grasp → approach → open gripper → close gripper → lift"
        pose_keys = [
            "start_pose", "pre_grasp_pose", "approach_pose",
            "close_gripper_pose", "lift_pose",
        ]

    try:
        # ── Wait for SPACE to start (raw terminal for single-key) ──
        print("\n  Move arm to START pose (rest position, gripper closed).")

        with RawTerm():
            print("  Press SPACE when ready to begin recording.\n")
            while True:
                state = [float(x) for x in robot.get_joint_positions()]
                sys.stdout.write(f"\r  Current: {fmt_vec(state)}  [SPACE=start, Q=quit]  ")
                sys.stdout.flush()
                key = RawTerm.read_key(0.15)
                if key.lower() == 'q' or key == 'ESC':
                    print("\n  Quit.")
                    return 0
                if key == ' ':
                    break

        # ── Record (raw terminal for single-key stop) ──
        with RawTerm():
            print(f"\n\n  ▶ RECORDING  (Hz={args.hz:.0f})")
            print(f"    Move through the full trajectory now:")
            print(f"    {traj_desc}")
            print(f"    Press SPACE to stop, or hold still for {args.auto_stop_sec:.0f}s to auto-stop.\n")

            frames = []
            step_time = 1.0 / args.hz
            last_moving_frame = 0
            movement_threshold = 0.003  # rad
            while True:
                t_start = time.time()
                state = [float(x) for x in robot.get_joint_positions()]
                frames.append(state)

                # Check movement (arm joints only)
                if len(frames) >= 3:
                    prev = np.array(frames[-3], dtype=np.float32)
                    cur = np.array(state, dtype=np.float32)
                    if max_movement(prev[:6], cur[:6]) > movement_threshold:
                        last_moving_frame = len(frames)

                # Print every 10 frames
                n = len(frames)
                if n % 10 == 1 or n <= 3:
                    sys.stdout.write(
                        f"\r  [{n:5d} frames, {n / args.hz:5.1f}s]  "
                        f"{fmt_vec(state)}  Grip={state[6]:.4f}  \n"
                        f"  ▶ SPACE=stop  Q=quit\n"
                        f"\033[2A"
                    )
                    sys.stdout.flush()

                # Check for key
                key = RawTerm.read_key(0.01)
                if key.lower() == 'q' or key == 'ESC':
                    print("\n\n  Recording cancelled.")
                    return 0
                if key == ' ':
                    break

                # Auto-stop
                idle_frames = len(frames) - last_moving_frame
                if idle_frames > args.auto_stop_sec * args.hz:
                    print(f"\n\n  Auto-stopped after {args.auto_stop_sec:.0f}s of no arm movement.")
                    break

                elapsed = time.time() - t_start
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)

        # ── Terminal is back to normal ──
        print(f"\n  Recorded {len(frames)} frames ({len(frames)/args.hz:.1f}s).")

        if len(frames) < 10:
            print("  Too few frames — aborting.")
            return 1

        # ── Extract waypoints ──
        states = np.array(frames, dtype=np.float32)
        grip = states[:, 6]

        print(f"\n  Gripper stats: min={grip.min():.4f}  max={grip.max():.4f}  "
              f"first={grip[0]:.4f}  last={grip[-1]:.4f}")

        if args.manual_waypoints:
            print("\n  Manual waypoint mode — enter frame indices:")
            result = {}
            result["start_pose"] = [float(x) for x in states[0]]
            for pk in pose_keys:
                if pk == "start_pose":
                    continue
                name = pk.replace("_pose", "")
                inp = input(f"  {pk} frame index: ").strip()
                idx = int(inp) if inp else 0
                result[pk] = [float(x) for x in states[idx]]
        else:
            if args.mode == "pick_place":
                result = extract_pick_place_waypoints(states)
            else:
                result = extract_waypoints(states)
            # Print detected frames
            fi = result["_frames"]
            print(f"\n  Auto-detected waypoints:")
            for pk in pose_keys:
                name = pk.replace("_pose", "")
                if name in fi:
                    print(f"    {name:16s} frame {fi[name]:4d}  grip={states[fi[name],6]:.4f}")
            print(f"    gripper open start:     {fi.get('gripper_open_start', '?'):}")
            print(f"    gripper close start:    {fi.get('gripper_close_start', '?'):}")
            print(f"    gripper close end:      {fi.get('gripper_close_end', '?'):}")
            if "gripper_release_start" in fi:
                print(f"    gripper release start:  {fi['gripper_release_start']}")
            warnings = result.get("_warnings", [])
            if warnings:
                print("\n  [WARN] Waypoint quality issues:")
                for warning in warnings:
                    print(f"    - {warning}")

            prompt = "\n  Accept these waypoints? [y/N] " if warnings else "\n  Accept these waypoints? [Y/n] "
            ok = input(prompt).strip().lower()
            reject = ok != "y" if warnings else ok == "n"
            if reject:
                print("  Edit frame indices manually:")
                for pk in pose_keys:
                    if pk == "start_pose":
                        continue
                    name = pk.replace("_pose", "")
                    cur = fi.get(name, 0)
                    inp = input(f"  {pk} frame [{cur}]: ").strip()
                    if inp:
                        idx = int(inp)
                        fi[name] = idx
                        result[pk] = [float(x) for x in states[idx]]

        # ── Build output ──
        out = {
            "source": "continuous_record_today",
            "created_at": datetime.now().isoformat(),
            "mode": args.mode,
            "open_gripper": args.open_gripper,
            "close_gripper": args.close_gripper,
            "notes": f"Auto-extracted from {len(frames)}-frame continuous recording ({args.mode})",
        }
        for pk in pose_keys:
            out[pk] = result[pk]
        if "_frames" in result:
            out["source_frames"] = result["_frames"]
        if result.get("_warnings"):
            out["warnings"] = result["_warnings"]

        # Save
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and not args.overwrite:
            bu = backup_path(args.output)
            shutil.copy2(args.output, bu)
            print(f"\n  Backup: {bu}")

        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(f"  Saved:  {args.output}")

        # Optional CSV
        if args.trajectory_output:
            np.savetxt(args.trajectory_output, states,
                       delimiter=",", header="j1,j2,j3,j4,j5,j6,gripper",
                       comments="", fmt="%.6f")
            print(f"  Trajectory CSV: {args.trajectory_output}")

        # ── Return to start pose before disabling ──
        do_return = args.return_to_start and not args.no_return
        if do_return and "start_pose" in out:
            print("\n" + "=" * 60)
            print("  Returning to start pose ...")
            start_pose = np.array(out["start_pose"], dtype=np.float64)
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

        # ── Recommend ──
        if args.mode == "pick_place":
            print(f"\n  Next step:")
            print(f"  python3 scripts/quick_bottle_grasp.py \\")
            print(f"      --waypoints {args.output} \\")
            print(f"      --mode pick_place --step-confirm --velocity-pct 50")
        else:
            print(f"\n  Next step:")
            print(f"  python3 scripts/quick_bottle_grasp.py \\")
            print(f"      --waypoints {args.output} \\")
            print(f"      --step-confirm --velocity-pct 50")

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        return 1
    finally:
        print()
        print("  Done. Arm stays ENABLED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
