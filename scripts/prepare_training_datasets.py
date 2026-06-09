#!/usr/bin/env python3
"""Prepare merged LeRobot datasets for cube-box ACT training experiments.

Creates 3 merged datasets in /tmp for training speed:
  1. blue_r0 + purple_r0_balanced → Training 2
  2. blue_r0 + blue_r90           → Training 3
  3. blue_r0 + blue_r90 + purple_r0_balanced + purple_r90_balanced → Training 4

Training 1 (blue_r0 alone) uses the original dataset directly — no merge needed.

Raw source datasets are NEVER modified. Excluded episodes are held out, not deleted.
"""
import os, sys, time, shutil
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from lerobot.datasets import LeRobotDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_BASE = Path("/tmp/piper_act_merged")

# ── Source datasets ──────────────────────────────────────────
SOURCES = {
    "blue_r0": {
        "root": str(DATA_ROOT / "lerobot_dataset_piper_blue_block_v1"),
        "repo_id": "piper/blue_block_v1",
    },
    "blue_r90": {
        "root": str(DATA_ROOT / "lerobot_dataset_piper_blue_block_paper90_v1"),
        "repo_id": "piper/blue_block_paper90_v1",
    },
    "purple_r0": {
        "root": str(DATA_ROOT / "lerobot_dataset_piper_purple_block_v1"),
        "repo_id": "piper/purple_block_v1",
        "exclude_episodes": [3],  # Held out from P1 — balanced subset
    },
    "purple_r90": {
        "root": str(DATA_ROOT / "lerobot_dataset_piper_purple_block_paper90_v1"),
        "repo_id": "piper/purple_block_paper90_v1",
        "exclude_episodes": [16, 9, 17, 10],  # Ep16 held out P1, Ep9 bad P3, Ep17 wrong pos, Ep10 knocked blue block
    },
}

# ── Merge definitions ────────────────────────────────────────
MERGES = {
    "train2_blue_purple_r0": {
        "name": "blue_r0 + purple_r0_balanced",
        "sources": ["blue_r0", "purple_r0"],
        "repo_id": "piper/cube_blue_purple_r0",
    },
    "train3_blue_r0_r90": {
        "name": "blue_r0 + blue_r90",
        "sources": ["blue_r0", "blue_r90"],
        "repo_id": "piper/cube_blue_r0_r90",
    },
    "train4_all_balanced": {
        "name": "blue_r0 + blue_r90 + purple_r0_balanced + purple_r90_balanced",
        "sources": ["blue_r0", "blue_r90", "purple_r0", "purple_r90"],
        "repo_id": "piper/cube_all_balanced",
    },
}


def merge_one(merge_name: str, merge_def: dict) -> Path:
    output_root = OUTPUT_BASE / merge_name
    if output_root.exists():
        print(f"[SKIP] {output_root} already exists, remove manually to re-merge")
        return output_root

    output_root.parent.mkdir(parents=True, exist_ok=True)

    source_list = []
    total_eps = 0
    for src_name in merge_def["sources"]:
        src = SOURCES[src_name]
        ds = LeRobotDataset(src["repo_id"], root=src["root"], tolerance_s=0.5)
        exclude = set(src.get("exclude_episodes", []))
        all_eps = sorted(set(int(e) for e in ds.hf_dataset["episode_index"]))
        kept = [e for e in all_eps if e not in exclude]
        total_eps += len(kept)
        source_list.append((ds, kept, src.get("exclude_episodes", [])))
        print(f"  {src_name}: {len(kept)}/{len(all_eps)} episodes kept"
              + (f" (excluded {sorted(exclude)})" if exclude else ""))

    print(f"\nMerging → {total_eps} episodes into {output_root}")
    first_ds = source_list[0][0]
    features = {k: first_ds.meta.features[k] for k in first_ds.meta.features}

    out = LeRobotDataset.create(
        repo_id=merge_def["repo_id"],
        fps=first_ds.meta.fps,
        features=features,
        root=str(output_root),
        use_videos=True,
        vcodec="h264",
        batch_encoding_size=1,
    )

    t_start = time.time()
    ep_count = 0

    for ds, episodes, excluded in source_list:
        ep_col = ds.hf_dataset["episode_index"]
        for ep in episodes:
            indices = [i for i, e in enumerate(ep_col) if e == ep]
            n_fr = len(indices)

            for src_idx in indices:
                frame = ds[int(src_idx)]
                fd = {
                    "observation.state": frame["observation.state"].numpy().astype(np.float32),
                    "action": frame["action"].numpy().astype(np.float32),
                    "task": frame.get("task", "merged demo"),
                }
                for ck in ds.meta.camera_keys:
                    img = frame[ck].numpy()
                    if img.dtype == np.float32:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    fd[ck] = img
                out.add_frame(fd)

            out.save_episode()
            ep_count += 1
            elapsed = time.time() - t_start
            rate = ep_count / elapsed * 60 if elapsed > 0 else 0
            if ep_count % 4 == 0 or ep_count == total_eps:
                print(f"  [{ep_count}/{total_eps}] ep {ep} ({n_fr} fr) — {elapsed:.0f}s, ~{rate:.1f} ep/min")

    print("  Finalizing...")
    out.finalize()
    elapsed = time.time() - t_start
    print(f"  Done: {total_eps} episodes, {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Quick verify
    vds = LeRobotDataset(merge_def["repo_id"], root=str(output_root))
    print(f"  Verify: {vds.num_episodes} eps, {vds.num_frames} fr  OK")
    return output_root


def main():
    for merge_name, merge_def in MERGES.items():
        print(f"\n{'='*60}")
        print(f"  {merge_name}: {merge_def['name']}")
        print(f"{'='*60}")
        try:
            merge_one(merge_name, merge_def)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll merges complete. Outputs in {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
