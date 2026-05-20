#!/usr/bin/env python3
"""Build a clean LeRobot dataset from selected episodes for Diffusion Policy training.

Filters episodes by:
  - Explicit episode list (--episodes 0,5,12,20)
  - Minimum gripper closure range (--min-gripper-range 0.03)
  - Crop first/last N frames (--crop-start 5 --crop-end 20)

Output: a new LeRobot v3.0 dataset directory.

USAGE:
  # List available episodes with stats
  python3 scripts/build_diffusion_dataset.py \
      --dataset-root data/lerobot_dataset_v2_delta \
      --list-episodes

  # Build dataset from specific episodes
  python3 scripts/build_diffusion_dataset.py \
      --dataset-root data/lerobot_dataset_v2_delta \
      --output data/diffusion_dataset_v1 \
      --episodes 0,5,12,20

  # Build from all episodes with gripper closure
  python3 scripts/build_diffusion_dataset.py \
      --dataset-root data/lerobot_dataset_v2_delta \
      --output data/diffusion_dataset_v1 \
      --min-gripper-range 0.03 \
      --crop-end 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset(root: Path):
    import pandas as pd
    paths = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {root}/data/")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    info = json.loads((root / "meta" / "info.json").read_text())
    return df, info


def compute_episode_stats(df):
    eps = sorted(df["episode_index"].unique())
    stats = []
    for ep in eps:
        mask = df["episode_index"] == ep
        ep_df = df[mask].sort_values("frame_index")
        states = np.stack(ep_df["observation.state"].values)
        grip = states[:, 6]
        stats.append({
            "episode": int(ep),
            "n_frames": len(ep_df),
            "grip_min": float(grip.min()),
            "grip_max": float(grip.max()),
            "grip_range": float(grip.max() - grip.min()),
            "grip_first": float(grip[0]),
            "grip_last": float(grip[-1]),
        })
    return stats


def compute_meta_stats(df):
    """Compute normalization stats from the filtered dataframe."""
    states = np.stack(df["observation.state"].values)
    actions = np.stack(df["action"].values)
    stats = {
        "observation.state": {
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
        },
        "action": {
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
        },
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build clean dataset for Diffusion Policy")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output dataset directory (default: data/diffusion_dataset_v1)")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode indices, e.g. '0,5,12'")
    parser.add_argument("--min-gripper-range", type=float, default=0.0,
                        help="Only include episodes with gripper range >= this value")
    parser.add_argument("--crop-start", type=int, default=0,
                        help="Frames to crop from the start of each episode")
    parser.add_argument("--crop-end", type=int, default=0,
                        help="Frames to crop from the end of each episode")
    parser.add_argument("--list-episodes", action="store_true",
                        help="Print per-episode stats and exit")
    parser.add_argument("--repo-id", type=str, default="piper/bottle_grasp_diffusion_v1",
                        help="Repo ID for the new dataset")
    args = parser.parse_args()

    print("Loading dataset ...")
    df, info = load_dataset(args.dataset_root)
    all_stats = compute_episode_stats(df)
    n_total = len(all_stats)

    # Print all episode stats
    print(f"\n{'='*70}")
    print(f"Episodes in source dataset: {n_total}")
    print(f"{'='*70}")
    print(f"{'Ep':>4}  {'Frames':>6}  {'Grip min':>10}  {'Grip max':>10}  "
          f"{'Range':>10}  {'First':>10}  {'Last':>10}  {'Include?':>10}")
    print("-" * 70)

    selected_eps = set()
    if args.episodes:
        selected_eps = set(int(x.strip()) for x in args.episodes.split(","))

    for s in all_stats:
        ep = s["episode"]
        include = True
        reason = ""
        if args.episodes and ep not in selected_eps:
            include = False
            reason = "not in list"
        if s["grip_range"] < args.min_gripper_range:
            include = False
            reason = f"range<{args.min_gripper_range:.3f}"
        print(f"  {ep:3d}  {s['n_frames']:6d}  {s['grip_min']:10.4f}  {s['grip_max']:10.4f}  "
              f"{s['grip_range']:10.4f}  {s['grip_first']:10.4f}  {s['grip_last']:10.4f}  "
              f"{'YES' if include else 'NO ('+reason+')':>10}")

    if args.list_episodes:
        print(f"\nPass --episodes <indices> or --min-gripper-range <val> to build a dataset.")
        return 0

    if not args.output:
        print("\n  Specify --output to build dataset.")
        return 1

    # Filter episodes
    filtered_eps = []
    for s in all_stats:
        ep = s["episode"]
        if args.episodes and ep not in selected_eps:
            continue
        if s["grip_range"] < args.min_gripper_range:
            continue
        filtered_eps.append(ep)

    print(f"\nBuilding dataset with {len(filtered_eps)} episodes: {filtered_eps}")

    if not filtered_eps:
        print("  No episodes selected. Aborting.")
        return 1

    # Build filtered dataframe with cropped frames
    import pandas as pd

    out_frames = []
    frame_counter = 0
    for ep in filtered_eps:
        mask = df["episode_index"] == ep
        ep_df = df[mask].sort_values("frame_index")
        if args.crop_start > 0:
            ep_df = ep_df.iloc[args.crop_start:]
        if args.crop_end > 0:
            ep_df = ep_df.iloc[:-args.crop_end] if len(ep_df) > args.crop_end else ep_df
        ep_df = ep_df.copy()
        ep_df["index"] = range(frame_counter, frame_counter + len(ep_df))
        ep_df["frame_index"] = range(len(ep_df))
        # Convert observation.state and action to proper list format for parquet
        out_frames.append(ep_df)
        frame_counter += len(ep_df)
        print(f"  ep {ep:03d}: {len(ep_df)} frames (after crop)")

    out_df = pd.concat(out_frames, ignore_index=True)
    out_df["task_index"] = 0

    # Compute new stats
    new_stats = compute_meta_stats(out_df)

    # Create output directory
    output_root = args.output
    out_data_dir = output_root / "data" / "chunk-000"
    out_meta_dir = output_root / "meta"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    # Write data
    out_parquet = out_data_dir / "file-000.parquet"
    out_df.to_parquet(out_parquet, index=False)
    print(f"\n  Data written: {out_parquet}  ({len(out_df)} frames)")

    # Write info.json (updated from source)
    new_info = dict(info)
    new_info["total_episodes"] = len(filtered_eps)
    new_info["repo_id"] = args.repo_id
    info_path = out_meta_dir / "info.json"
    info_path.write_text(json.dumps(new_info, indent=2) + "\n")
    print(f"  Info written: {info_path}")

    # Write stats.json
    stats_path = out_meta_dir / "stats.json"
    stats_path.write_text(json.dumps(new_stats, indent=2) + "\n")
    print(f"  Stats written: {stats_path}")

    # Print per-dim stats summary
    print(f"\n{'='*70}")
    print("New dataset action stats:")
    print("-" * 70)
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]
    act = new_stats["action"]
    print(f"  {'Dim':>6}  {'min':>12}  {'max':>12}  {'mean':>12}  {'std':>12}")
    for d in range(7):
        print(f"  {joint_names[d]:>6}  {act['min'][d]:12.6f}  {act['max'][d]:12.6f}  "
              f"{act['mean'][d]:12.6f}  {act['std'][d]:12.6f}")

    print(f"\n{'='*70}")
    print("Verification — per-episode gripper stats:")
    print("-" * 70)
    verif_stats = compute_episode_stats(out_df)
    for s in verif_stats:
        flag = "OK" if s["grip_range"] >= 0.001 else "NO_CLOSE"
        print(f"  ep {s['episode']:03d}: frames={s['n_frames']:4d}  "
              f"grip=[{s['grip_min']:.4f}, {s['grip_max']:.4f}]  range={s['grip_range']:.4f}  [{flag}]")

    print(f"\n  Dataset ready: {output_root}")
    print(f"  Repo ID: {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
