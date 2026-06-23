#!/usr/bin/env python3
"""
Replay a recorded LeRobot episode through the ROS safety bridge via UDP.

Pure-stdlib + numpy/pandas only — no rclpy, no Piper SDK.

Usage:
  python3 scripts/replay_recorded_trajectory_ros.py \
    --dataset data/cube_64_dual \
    --episode 0 \
    --action-source state_next \
    --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Constants (mirrored from ros_bridge.common — no rclpy import)
# ---------------------------------------------------------------------------
UDP_HOST = "127.0.0.1"
UDP_PORT = 50051
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
MAX_DELTA = {
    "joint1": 0.03, "joint2": 0.03, "joint3": 0.03,
    "joint4": 0.012, "joint5": 0.012, "joint6": 0.012,
    "gripper": 0.004,
}
ACT_UDP_STATE_PORT = 50052

# ---------------------------------------------------------------------------
#  Real-state read via UDP (pure stdlib)
# ---------------------------------------------------------------------------
def read_real_qpos(host="127.0.0.1", port=ACT_UDP_STATE_PORT, timeout=2.0):
    """Read current 7D joint state from ros_state_udp_publisher_node via UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 2)
    sock.settimeout(timeout)
    sock.bind((host, port))
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        try:
            data, _addr = sock.recvfrom(4096)
            msg = json.loads(data.decode("utf-8"))
            pos = msg.get("position")
            if isinstance(pos, list) and len(pos) == 7:
                latest = [float(v) for v in pos]
        except socket.timeout:
            break
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            continue
    sock.close()
    if latest is None:
        raise RuntimeError(
            f"No UDP state received within {timeout}s. "
            "Is ros_state_udp_publisher_node running?"
        )
    return latest


# ---------------------------------------------------------------------------
#  UDP send
# ---------------------------------------------------------------------------
class _UdpSender:
    def __init__(self, host=UDP_HOST, port=UDP_PORT, dry_run=False):
        self._dry_run = dry_run
        self._sock = None if dry_run else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)

    def send(self, action_7d):
        if hasattr(action_7d, "tolist"):
            action_7d = action_7d.tolist()
        action_7d = [float(v) for v in action_7d]
        if len(action_7d) != 7:
            raise ValueError(f"action len != 7: {len(action_7d)}")
        if not self._dry_run:
            payload = json.dumps({"action": action_7d})
            self._sock.sendto(payload.encode("utf-8"), self._addr)

    def close(self):
        if self._sock is not None:
            self._sock.close()


# ---------------------------------------------------------------------------
#  Dataset loading
# ---------------------------------------------------------------------------
def load_episode(dataset_dir: str, episode_idx: int):
    """Load a single episode from a LeRobot dataset."""
    root = Path(dataset_dir)
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {root}/data/")

    all_frames = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        ep_df = df[df["episode_index"] == episode_idx]
        if len(ep_df) == 0:
            continue
        for _, row in ep_df.sort_values("frame_index").iterrows():
            all_frames.append({
                "obs": np.asarray(row["observation.state"], dtype=np.float64),
                "action": np.asarray(row["action"], dtype=np.float64),
                "frame": int(row["frame_index"]),
                "episode": int(row["episode_index"]),
                "task": int(row["task_index"]),
            })
    if not all_frames:
        raise ValueError(f"Episode {episode_idx} not found in {dataset_dir}")
    return all_frames


def build_actions(frames, action_source: str):
    """Extract target positions from episode frames.

    ``action_source = "action"`` uses the recorded action column directly.
    ``action_source = "state_next"`` uses the NEXT frame's observation.state.
    """
    targets = []
    n = len(frames)
    for i, f in enumerate(frames):
        if action_source == "action":
            targets.append(np.asarray(f["action"], dtype=np.float64).copy())
        elif action_source == "state_next":
            if i + 1 < n:
                targets.append(np.asarray(frames[i + 1]["obs"], dtype=np.float64).copy())
            else:
                targets.append(np.asarray(f["obs"], dtype=np.float64).copy())
        else:
            raise ValueError(f"Unknown action_source: {action_source}")
    return targets


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------
def print_episode_diag(frames, targets, action_source: str):
    """Print episode-level diagnostics."""
    n = len(frames)
    obs0 = frames[0]["obs"]
    obs_final = frames[-1]["obs"]
    obs_array = np.array([f["obs"] for f in frames])
    gripper_vals = obs_array[:, 6]

    print(f"\n  Episode frames: {n}")
    print(f"  Action source:  {action_source}")
    print(f"  First state:    {fmt_vec(obs0)}")
    print(f"  Final state:    {fmt_vec(obs_final)}")
    print(f"  Gripper range:  [{gripper_vals.min():.5f}, {gripper_vals.max():.5f}]")

    # Gripper close/release timing
    grip_initial = gripper_vals[0]
    close_frame = None
    for i, g in enumerate(gripper_vals):
        if abs(g - grip_initial) > 0.02:
            close_frame = i
            break
    if close_frame is not None:
        print(f"  Gripper begins to close at frame {close_frame}")
    else:
        print(f"  Gripper held steady (no close detected)")

    # J2/J3 range
    j2 = obs_array[:, 1]
    j3 = obs_array[:, 2]
    print(f"  J2 range:       [{j2.min():+.5f}, {j2.max():+.5f}]  Δ={j2[-1]-j2[0]:+.5f}")
    print(f"  J3 range:       [{j3.min():+.5f}, {j3.max():+.5f}]  Δ={j3[-1]-j3[0]:+.5f}")

    # NaN/Inf check
    nan_count = 0
    inf_count = 0
    for f in frames:
        if np.any(np.isnan(f["obs"])):
            nan_count += 1
        if np.any(np.isinf(f["obs"])):
            inf_count += 1
    print(f"  NaN frames: {nan_count}  Inf frames: {inf_count}")

    # Large jumps
    max_jump = 0.0
    max_jump_frame = 0
    max_joint = ""
    for i in range(1, n):
        diff = np.abs(targets[i - 1] - targets[i - 1])  # target-to-target
        for j in range(7):
            d = abs(obs_array[i][j] - obs_array[i - 1][j])
            if d > max_jump:
                max_jump = d
                max_jump_frame = i
                max_joint = JOINT_NAMES[j]
    print(f"  Max frame-to-frame jump: {max_jump:.5f} ({max_joint} at frame {max_jump_frame})")

    # Target stats
    tgt_array = np.array(targets)
    for j in range(7):
        name = JOINT_NAMES[j]
        tgt_deltas = np.diff(tgt_array[:, j])
        abs_max = float(np.max(np.abs(tgt_deltas))) if len(tgt_deltas) > 0 else 0.0
        limit = MAX_DELTA.get(name, 0.03)
        hits = int(np.sum(np.abs(tgt_deltas) >= limit - 1e-6))
        print(f"  {name:>8s} target delta: max={abs_max:.5f}  hits>={limit:.3f}: {hits}/{n - 1}")


def fmt_vec(v, precision=5):
    return "[" + ", ".join(f"{float(x):.{precision}f}" for x in v) + "]"


# ---------------------------------------------------------------------------
#  Start pose check
# ---------------------------------------------------------------------------
def check_start_pose(current_qpos, episode_first_state):
    """Return (ok, error_string)."""
    errors = []
    for i, name in enumerate(JOINT_NAMES[:6]):
        err = abs(current_qpos[i] - float(episode_first_state[i]))
        if i < 3:  # J1/J2/J3
            limit = 0.10
        else:  # J4/J5/J6
            limit = 0.15
        if err > limit:
            errors.append(f"{name}: err={err:.4f} > {limit:.2f}")
    gripper_err = abs(current_qpos[6] - float(episode_first_state[6]))
    if errors:
        return False, " | ".join(errors) + f" | gripper err={gripper_err:.4f}"
    return True, f"OK (max arm err={max(abs(current_qpos[i]-float(episode_first_state[i])) for i in range(6)):.4f})"


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded LeRobot episode through ROS safety bridge via UDP."
    )
    parser.add_argument("--dataset", required=True, help="Path to LeRobot dataset dir")
    parser.add_argument("--episode", type=int, required=True, help="Episode index")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="Max frames to send (-1 = all)")
    parser.add_argument("--action-source", choices=("action", "state_next"), default="state_next",
                        help="action = use recorded action column; state_next = next frame obs")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale applied to delta (default: 1.0 = full)")
    parser.add_argument("--rate", type=float, default=4.0,
                        help="Replay send rate in Hz (default: 4)")
    parser.add_argument("--freeze-gripper", action="store_true", default=True,
                        help="Freeze gripper at current position (default)")
    parser.add_argument("--no-freeze-gripper", dest="freeze_gripper", action="store_false",
                        help="Allow gripper to move")
    parser.add_argument("--gripper-shadow", action="store_true", default=False,
                        help="Send gripper to UDP but mark as shadow (for gripper diagnostics)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print what would be sent, no UDP")
    parser.add_argument("--udp-host", default=UDP_HOST)
    parser.add_argument("--udp-port", type=int, default=UDP_PORT)

    args = parser.parse_args()

    # --- Load episode ---
    print(f"\nLoading {args.dataset} episode {args.episode} ...")
    frames = load_episode(args.dataset, args.episode)
    if args.num_frames > 0:
        frames = frames[: args.num_frames]
    targets = build_actions(frames, args.action_source)

    # --- Episode diagnostics ---
    print("=" * 60)
    print("EPISODE DIAGNOSTICS")
    print("=" * 60)
    print_episode_diag(frames, targets, args.action_source)

    # --- Read current real qpos ---
    print("\n" + "=" * 60)
    print("START POSE CHECK")
    print("=" * 60)
    try:
        current_qpos = read_real_qpos()
        print(f"  Current real qpos: {fmt_vec(current_qpos)}")
        print(f"  Episode first obs: {fmt_vec(frames[0]['obs'])}")
        start_ok, start_msg = check_start_pose(current_qpos, frames[0]["obs"])
        print(f"  Start pose check:  {start_msg}")
        if not start_ok:
            print("\n  *** START POSE MISMATCH — real-write BLOCKED ***")
            print("  Continuing in shadow-only mode (--dry-run forced).")
            args.dry_run = True
    except RuntimeError as e:
        print(f"  WARNING: {e}")
        print("  Cannot verify start pose. Continuing in shadow-only mode.")
        start_ok = False
        args.dry_run = True
        current_qpos = None

    # --- Replay ---
    print("\n" + "=" * 60)
    mode = "DRY-RUN (shadow)" if args.dry_run else f"REPLAY ({args.scale:.0%} scale, {args.rate}Hz)"
    print(f"REPLAY: {mode}")
    print(f"  Frames: {len(frames)}  Action source: {args.action_source}")
    print(f"  Freeze gripper: {args.freeze_gripper}  Gripper shadow: {args.gripper_shadow}")
    print("=" * 60)

    sender = _UdpSender(host=args.udp_host, port=args.udp_port, dry_run=args.dry_run)
    interval = 1.0 / max(1.0, args.rate)

    sent_count = 0
    max_delta_hits = {name: 0 for name in JOINT_NAMES}
    gripper_log = []

    try:
        for i, (frame, target) in enumerate(zip(frames, targets)):
            t0 = time.monotonic()

            # Compute action: scale delta from current target vs frame obs
            if args.action_source == "action":
                # action column = absolute target
                action_abs = target.copy()
            else:
                # state_next = next frame obs = absolute target
                action_abs = target.copy()

            # Apply scale: if using state_next, delta = target - current_obs
            frame_obs = np.asarray(frame["obs"], dtype=np.float64)
            delta = action_abs - frame_obs
            scaled_delta = delta * args.scale
            sent_target = frame_obs + scaled_delta

            # Gripper handling
            raw_grip = float(sent_target[6])
            if args.freeze_gripper and current_qpos is not None:
                sent_target[6] = current_qpos[6]

            # Log gripper
            grip_actual = current_qpos[6] if current_qpos is not None else float('nan')
            gripper_log.append((i, raw_grip, float(sent_target[6]), grip_actual))

            # Count MAX_DELTA hits
            for j, name in enumerate(JOINT_NAMES):
                if name == "gripper" and args.freeze_gripper:
                    continue
                if abs(scaled_delta[j]) > MAX_DELTA.get(name, 0.03) - 1e-6:
                    max_delta_hits[name] += 1

            # Send
            if not args.dry_run:
                sender.send(sent_target.tolist())
            sent_count += 1

            # Print progress
            if i == 0 or (i + 1) % 50 == 0 or i == len(frames) - 1:
                print(f"  [{i + 1}/{len(frames)}] sent={fmt_vec(sent_target, 4)}")

            # Rate-limit
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print(f"\n  Interrupted at frame {sent_count}/{len(frames)}")
    finally:
        sender.close()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("REPLAY SUMMARY")
    print("=" * 60)
    print(f"  Frames sent:  {sent_count} / {len(frames)}")
    print(f"  Scale:        {args.scale:.0%}")
    print(f"  Mode:         {'DRY-RUN' if args.dry_run else 'REAL-WRITE via UDP'}")
    print(f"  MAX_DELTA hits:")
    for name in JOINT_NAMES:
        if max_delta_hits[name] > 0:
            print(f"    {name:>8s}: {max_delta_hits[name]} / {sent_count}")
    if not any(max_delta_hits.values()):
        print("    (none)")

    print("\n  Gripper shadow:")
    print(f"  {'Frame':>6} {'raw':>10} {'sent':>10} {'actual':>10}")
    step = max(1, len(gripper_log) // 20)
    for i in range(0, len(gripper_log), step):
        fnum, raw, sent, actual = gripper_log[i]
        print(f"  {fnum:6d} {raw:10.5f} {sent:10.5f} {actual:10.5f}")

    grip_raws = [g[1] for g in gripper_log if not math.isnan(g[1])]
    if grip_raws:
        print(f"\n  Raw gripper range: [{min(grip_raws):.5f}, {max(grip_raws):.5f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
