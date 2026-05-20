#!/usr/bin/env python3
"""Extract key waypoints from a successful LeRobot episode for trajectory replay.

Usage:
  # Auto-detect waypoints from episode 12:
  python3 scripts/extract_success_waypoints.py \
      --dataset-root data/lerobot_dataset_v2_delta \
      --episode 12

  # Manually specify key frame indices:
  python3 scripts/extract_success_waypoints.py \
      --dataset-root data/lerobot_dataset_v2_delta \
      --episode 5 \
      --frames start=0,pre_grasp=40,approach=70,close_gripper=90,lift=180

Output: configs/bottle_grasp_waypoints.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WAYPOINT_KEYS = ["start", "pre_grasp", "approach", "close_gripper", "lift"]
POSE_JSON_KEYS = ["start_pose", "pre_grasp_pose", "approach_pose",
                   "close_gripper_pose", "lift_pose"]
GRIPPER_CLOSE_THRESHOLD = 0.005   # gripper value below this is "closed"
GRIPPER_OPEN_THRESHOLD = 0.01     # gripper value above this is "opening"
GRIPPER_CLOSING_DELTA = 0.001     # per-frame drop means gripper is closing


def load_dataset(root: Path):
    import pandas as pd
    paths = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {root}/data/")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def get_episode(df, ep: int):
    mask = df["episode_index"] == ep
    ep_df = df[mask].sort_values("frame_index").reset_index(drop=True)
    states = np.stack(ep_df["observation.state"].values)
    return ep_df, states


def auto_detect_waypoints(states: np.ndarray):
    """Heuristic waypoint detection based on gripper and arm motion."""
    n = len(states)
    grip = states[:, 6]
    j1 = states[:, 0]
    j2 = states[:, 1]
    j3 = states[:, 2]

    # --- start: first frame ---
    start_idx = 0

    # --- pre_grasp: just before gripper starts opening ---
    # Find first frame where gripper crosses OPEN_THRESHOLD
    open_start = None
    for i in range(n):
        if grip[i] > GRIPPER_OPEN_THRESHOLD:
            open_start = i
            break
    if open_start is None:
        open_start = n // 3  # fallback
    # pre_grasp = a few frames before gripper opens
    pre_grasp_idx = max(0, open_start - 5)

    # --- approach_pose: frame where gripper is at max opening ---
    grip_max_idx = int(np.argmax(grip))
    approach_idx = grip_max_idx

    # --- close_gripper_pose: frame where gripper starts closing from max ---
    close_start_idx = grip_max_idx
    for i in range(grip_max_idx, min(grip_max_idx + 30, n)):
        if grip[i] - grip[min(i + 1, n - 1)] > GRIPPER_CLOSING_DELTA:
            close_start_idx = i
            break
    # If no clear closing detected, use 10 frames after max
    if close_start_idx == grip_max_idx:
        close_start_idx = min(grip_max_idx + 10, n - 1)

    # --- lift_pose: after gripper closed, arm retracted ---
    # Find where gripper is closed again and arm has started moving back
    close_end_idx = n - 1
    for i in range(close_start_idx, n):
        if grip[i] < GRIPPER_CLOSE_THRESHOLD:
            close_end_idx = i
            break
    # lift is some frames after gripper closes
    lift_idx = min(close_end_idx + 30, n - 1)

    return {
        "start": start_idx,
        "pre_grasp": pre_grasp_idx,
        "approach": approach_idx,
        "close_gripper": close_start_idx,
        "lift": lift_idx,
        "_gripper_open_start": int(open_start) if open_start else -1,
        "_gripper_max": int(grip_max_idx),
        "_gripper_close_end": int(close_end_idx),
    }


def build_waypoint_json(states: np.ndarray, frame_indices: dict, ep: int):
    result = {}
    for waypoint_key, json_key in zip(WAYPOINT_KEYS, POSE_JSON_KEYS):
        idx = frame_indices[waypoint_key]
        result[json_key] = [float(x) for x in states[idx]]

    result["open_gripper"] = float(np.max(states[:, 6]))
    result["close_gripper"] = 0.0
    result["source_episode"] = ep
    result["source_frames"] = {k: int(frame_indices.get(k, -1)) for k in WAYPOINT_KEYS}
    result["_gripper_stats"] = {
        "min": float(np.min(states[:, 6])),
        "max": float(np.max(states[:, 6])),
        "first": float(states[0, 6]),
        "last": float(states[-1, 6]),
    }
    return result


def parse_manual_frames(spec: str):
    """Parse 'start=0,pre_grasp=40,approach=70,...' into dict."""
    result = {}
    for part in spec.split(","):
        key, val = part.strip().split("=")
        result[key.strip()] = int(val.strip())
    for k in WAYPOINT_KEYS:
        if k not in result:
            raise ValueError(f"Missing key '{k}' in manual frames spec: {spec}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract waypoints from a LeRobot episode")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "configs" / "bottle_grasp_waypoints.json")
    parser.add_argument("--frames", type=str, default=None,
                        help="Manual frame indices: 'start=0,pre_grasp=40,approach=70,"
                             "close_gripper=90,lift=180'")
    parser.add_argument("--plot", action="store_true",
                        help="Show trajectory plot (requires matplotlib)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_root} ...")
    df = load_dataset(args.dataset_root)
    total_eps = int(df["episode_index"].max()) + 1
    print(f"  Episodes available: 0–{total_eps - 1} ({len(df)} total frames)")

    ep_df, states = get_episode(df, args.episode)
    print(f"  Episode {args.episode}: {len(ep_df)} frames")
    print(f"  Gripper: first={states[0,6]:.4f}  max={states[:,6].max():.4f}  "
          f"last={states[-1,6]:.4f}  std={states[:,6].std():.4f}")
    print()

    if args.frames:
        frame_indices = parse_manual_frames(args.frames)
        print("Using MANUAL frame indices:")
    else:
        frame_indices = auto_detect_waypoints(states)
        print("Auto-detected waypoints:")

    for key in WAYPOINT_KEYS:
        idx = frame_indices[key]
        s = states[idx]
        print(f"  {key:>16}: frame {idx:4d}  "
              f"J1={s[0]:7.4f} J2={s[1]:7.4f} J3={s[2]:7.4f} "
              f"J4={s[3]:7.4f} J5={s[4]:7.4f} J6={s[5]:7.4f} Grip={s[6]:.4f}")

    if not args.frames:
        extra = {k: v for k, v in frame_indices.items() if k.startswith("_")}
        print(f"\n  Debug: gripper_open_start={extra.get('_gripper_open_start')}, "
              f"gripper_max={extra.get('_gripper_max')}, "
              f"gripper_close_end={extra.get('_gripper_close_end')}")

    # Build and save
    waypoints = build_waypoint_json(states, frame_indices, args.episode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(waypoints, indent=2) + "\n")
    print(f"\nWaypoints saved to: {args.output}")
    print(f"  open_gripper:  {waypoints['open_gripper']:.4f}")
    print(f"  close_gripper: {waypoints['close_gripper']:.4f}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            ax1, ax2 = axes
            for j in range(6):
                ax1.plot(states[:, j], label=f"J{j+1}", linewidth=1)
            ax1.set_ylabel("Joint position (rad)")
            ax1.legend(ncol=6, fontsize=8, loc="upper right")
            ax1.grid(True, alpha=0.3)
            ax2.plot(states[:, 6], "k", linewidth=1.5, label="Gripper")
            ax2.set_ylabel("Gripper (m)")
            ax2.set_xlabel("Frame")
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            for key in WAYPOINT_KEYS:
                idx = frame_indices[key]
                for ax in axes:
                    ax.axvline(idx, color="red", linestyle="--", alpha=0.5, linewidth=1)
                ax1.text(idx, ax1.get_ylim()[1] * 0.95, key, rotation=90,
                         fontsize=7, color="red", va="top")
            fig.suptitle(f"Episode {args.episode} — Waypoints", fontsize=13, weight="bold")
            plot_path = args.output.with_suffix(".png")
            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved to: {plot_path}")
        except ImportError:
            print("  (matplotlib not available; skip plot)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
