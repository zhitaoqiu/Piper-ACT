#!/usr/bin/env python3
"""Merge two adapter-v2 LeRobot v3 datasets into one, excluding a bad episode.

Usage:
    python3 scripts/merge_adapter_v2_datasets.py \
        --source data/lerobot_dataset_piper_bottle_adapter_v2_10demo \
        --source data/lerobot_dataset_piper_bottle_adapter_v2_new_demos \
        --target data/lerobot_dataset_piper_bottle_adapter_v2_25demo \
        --exclude-source 1 --exclude-episode 9
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CAMERA_KEY = "observation.images.global_rgb"
VIDEO_DIR = "videos"
DATA_DIR = "data"
META_DIR = "meta"
ADAPTER_META_DIR = "adapter_v2_episode_metadata"


def load_parquet(dataset_root: Path) -> pd.DataFrame:
    paths = sorted((dataset_root / DATA_DIR).glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {dataset_root / DATA_DIR}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _map_new_episode(old_ep: int, base: int, exclude_episodes: set[int]) -> int:
    """Map old new-episode index to new global episode index, skipping excluded."""
    offset = sum(1 for e in sorted(exclude_episodes) if e < old_ep)
    return base + old_ep - offset


def merge_episode_metadata(old_root: Path, new_root: Path, target_root: Path,
                           exclude_episodes: set[int], old_max_ep: int):
    """Merge meta/episodes parquet files from old and new datasets."""
    target_ep_dir = target_root / META_DIR / "episodes"
    old_ep_dir = old_root / META_DIR / "episodes"
    new_ep_dir = new_root / META_DIR / "episodes"

    # Copy old chunk-000 episode metadata unchanged
    if (old_ep_dir / "chunk-000").exists():
        shutil.copytree(old_ep_dir / "chunk-000", target_ep_dir / "chunk-000")

    # Load new episode metadata
    new_ep_paths = sorted(new_ep_dir.glob("chunk-*/file-*.parquet"))
    new_ep_df = pd.concat([pd.read_parquet(p) for p in new_ep_paths], ignore_index=True)

    # Filter and offset
    new_ep_df = new_ep_df[~new_ep_df["episode_index"].isin(exclude_episodes)].copy()
    new_ep_df["episode_index"] = new_ep_df["episode_index"].apply(
        lambda ep: _map_new_episode(ep, old_max_ep + 1, exclude_episodes)
    )
    # dataset_from_index / dataset_to_index stay as-is — they reference
    # frame positions within chunk-001's own video file.

    # Update chunk indices for new episodes
    for col in new_ep_df.columns:
        if "chunk_index" in col:
            new_ep_df[col] = 1
        if "file_index" in col:
            new_ep_df[col] = 0

    # Write to target chunk-001
    (target_ep_dir / "chunk-001").mkdir(parents=True, exist_ok=True)
    new_ep_df.to_parquet(target_ep_dir / "chunk-001" / "file-000.parquet")


def merge_adapter_metadata(old_root: Path, new_root: Path, target_root: Path,
                           exclude_episodes: set[int], old_max_ep: int):
    """Merge adapter_v2_episode_metadata JSON files."""
    target_adapter_dir = target_root / META_DIR / ADAPTER_META_DIR
    target_adapter_dir.mkdir(parents=True, exist_ok=True)

    # Copy old metadata unchanged (episodes 0-9)
    old_adapter_dir = old_root / META_DIR / ADAPTER_META_DIR
    for f in sorted(old_adapter_dir.glob("episode_*.json")):
        shutil.copy2(f, target_adapter_dir / f.name)

    # Copy new metadata, excluding bad episode, renumbering
    new_adapter_dir = new_root / META_DIR / ADAPTER_META_DIR
    for f in sorted(new_adapter_dir.glob("episode_*.json")):
        old_ep = int(f.stem.split("_")[1])
        if old_ep in exclude_episodes:
            continue
        new_ep = _map_new_episode(old_ep, old_max_ep + 1, exclude_episodes)
        data = json.loads(f.read_text())
        data["episode_id"] = new_ep
        target_name = f"episode_{new_ep:06d}.json"
        (target_adapter_dir / target_name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge adapter-v2 LeRobot datasets")
    parser.add_argument("--source", action="append", dest="sources", required=True,
                        help="Source dataset roots (first is primary/old).")
    parser.add_argument("--target", required=True,
                        help="Target dataset root.")
    parser.add_argument("--exclude-source", type=int, default=None,
                        help="Source index to exclude episodes from (0-indexed).")
    parser.add_argument("--exclude-episode", type=int, action="append",
                        dest="exclude_episodes", default=[],
                        help="Episode indices to exclude (can repeat).")
    args = parser.parse_args()

    if len(args.sources) < 2:
        print("[ERROR] Need at least 2 source datasets to merge.")
        return 1

    old_root = Path(args.sources[0])
    new_root = Path(args.sources[1])
    target_root = Path(args.target)

    exclude_episodes: set[int] = set(args.exclude_episodes)

    # Validate
    for src in args.sources:
        p = Path(src)
        if not (p / META_DIR / "info.json").exists():
            print(f"[ERROR] Not a valid dataset: {src}")
            return 1

    print("=" * 64)
    print("  Adapter v2 Dataset Merge")
    print("=" * 64)
    print(f"  Old  : {old_root}")
    print(f"  New  : {new_root}")
    print(f"  Target: {target_root}")
    if exclude_episodes:
        print(f"  Exclude new episodes: {sorted(exclude_episodes)}")
    print("=" * 64)

    # Load data
    old_df = load_parquet(old_root)
    new_df = load_parquet(new_root)
    old_max_ep = int(old_df["episode_index"].max())

    new_eps_before = sorted(new_df["episode_index"].unique())
    new_eps_after = [ep for ep in new_eps_before if ep not in exclude_episodes]

    print(f"\n  Old episodes: {old_max_ep + 1} ({len(old_df)} frames)")
    print(f"  New episodes before exclusion: {len(new_eps_before)} ({len(new_df)} frames)")
    print(f"  New episodes after exclusion: {len(new_eps_after)}")
    total_eps = (old_max_ep + 1) + len(new_eps_after)
    print(f"  Total combined episodes: {total_eps}")

    if target_root.exists():
        print(f"\n[ERROR] Target already exists: {target_root}")
        return 1

    # Create target structure
    target_root.mkdir(parents=True)

    # 1. Copy old data as chunk-000
    print("\nCopying old chunk-000...")
    old_data_dir = old_root / DATA_DIR / "chunk-000"
    target_data_dir = target_root / DATA_DIR / "chunk-000"
    shutil.copytree(old_data_dir, target_data_dir)

    # 2. Write filtered new data as chunk-001
    print("Writing new chunk-001 (filtered)...")
    new_filtered = new_df[~new_df["episode_index"].isin(exclude_episodes)].copy()
    new_filtered["episode_index"] = new_filtered["episode_index"].apply(
        lambda ep: _map_new_episode(ep, old_max_ep + 1, exclude_episodes)
    )
    target_data_chunk1 = target_root / DATA_DIR / "chunk-001"
    target_data_chunk1.mkdir(parents=True)
    new_filtered.to_parquet(target_data_chunk1 / "file-000.parquet", index=False)

    # 3. Copy videos
    print("Copying old video (chunk-000)...")
    old_vid_src = old_root / VIDEO_DIR / CAMERA_KEY / "chunk-000"
    old_vid_dst = target_root / VIDEO_DIR / CAMERA_KEY / "chunk-000"
    old_vid_dst.mkdir(parents=True)
    for f in old_vid_src.iterdir():
        shutil.copy2(f, old_vid_dst / f.name)

    print("Copying new video (chunk-001)...")
    new_vid_src = new_root / VIDEO_DIR / CAMERA_KEY / "chunk-000"
    new_vid_dst = target_root / VIDEO_DIR / CAMERA_KEY / "chunk-001"
    new_vid_dst.mkdir(parents=True)
    for f in new_vid_src.iterdir():
        shutil.copy2(f, new_vid_dst / f.name)

    # Copy images directory if present
    for subdir in ["images"]:
        old_img = old_root / subdir
        if old_img.exists():
            target_img = target_root / subdir
            if not target_img.exists():
                shutil.copytree(old_img, target_img)

    # 4. Merge episode metadata
    print("Merging episode metadata...")
    merge_episode_metadata(old_root, new_root, target_root, exclude_episodes, old_max_ep)

    # 5. Merge adapter v2 metadata
    print("Merging adapter v2 episode metadata...")
    merge_adapter_metadata(old_root, new_root, target_root, exclude_episodes, old_max_ep)

    # 6. Copy and update info.json
    print("Updating info.json...")
    old_info = json.loads((old_root / META_DIR / "info.json").read_text())
    old_info["total_episodes"] = total_eps
    old_info["total_frames"] = int(len(old_df) + len(new_filtered))
    (target_root / META_DIR).mkdir(parents=True, exist_ok=True)
    (target_root / META_DIR / "info.json").write_text(
        json.dumps(old_info, indent=2, ensure_ascii=False) + "\n"
    )

    # 7. Copy stats.json from old (approximate)
    old_stats = old_root / META_DIR / "stats.json"
    if old_stats.exists():
        shutil.copy2(old_stats, target_root / META_DIR / "stats.json")

    # 8. Copy tasks.parquet
    old_tasks = old_root / META_DIR / "tasks.parquet"
    if old_tasks.exists():
        shutil.copy2(old_tasks, target_root / META_DIR / "tasks.parquet")

    print(f"\nMerge complete: {target_root}")
    print(f"  {total_eps} episodes, {len(old_df) + len(new_filtered)} total frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
