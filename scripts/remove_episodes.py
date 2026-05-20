#!/usr/bin/env python3
"""Remove specific episodes from a LeRobot dataset, updating all metadata.

Usage:
  PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 scripts/remove_episodes.py \
    --dataset-root data/lerobot_dataset_approach_20ep \
    --episodes 2
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episodes", required=True,
                        help="comma-separated episode indices to remove, e.g. 2,5,7")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    remove_eps = {int(x.strip()) for x in args.episodes.split(",")}

    # 1. Filter data parquet
    data_dir = root / "data"
    for pqf in sorted(data_dir.rglob("*.parquet")):
        t = pq.read_table(str(pqf))
        mask = [e not in remove_eps for e in t.column("episode_index").to_pylist()]
        if all(mask):
            continue
        n_before = len(t)
        t_new = t.filter(mask)
        n_after = len(t_new)
        print(f"  data/{pqf.relative_to(data_dir)}: {n_before} → {n_after} rows (removed {n_before - n_after})")
        pq.write_table(t_new, str(pqf))

    # 2. Filter meta/episodes parquet and recalculate offsets
    meta_ep_dir = root / "meta" / "episodes"
    for pqf in sorted(meta_ep_dir.rglob("*.parquet")):
        t = pq.read_table(str(pqf))
        mask = [e not in remove_eps for e in t.column("episode_index").to_pylist()]
        if all(mask):
            continue
        t_new = t.filter(mask)
        # Recalculate dataset_from_index / dataset_to_index
        lengths = t_new.column("length").to_pylist()
        ds_from = []
        ds_to = []
        offset = 0
        for leng in lengths:
            ds_from.append(offset)
            ds_to.append(offset + leng)
            offset += leng
        # Build new column
        n_rows = len(lengths)
        ds_from_arr = pa.array(ds_from, type=pa.int64())
        ds_to_arr = pa.array(ds_to, type=pa.int64())
        col_names = t_new.column_names
        col_dict = {}
        for cn in col_names:
            if cn == "dataset_from_index":
                col_dict[cn] = ds_from_arr
            elif cn == "dataset_to_index":
                col_dict[cn] = ds_to_arr
            else:
                col_dict[cn] = t_new.column(cn)
        t_new = pa.table(col_dict)
        print(f"  meta/episodes/{pqf.relative_to(meta_ep_dir)}: {len(t)} → {len(t_new)} rows")
        pq.write_table(t_new, str(pqf))

    # 3. Update info.json
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    total_frames = 0
    all_eps = set()
    for pqf in sorted(data_dir.rglob("*.parquet")):
        t = pq.read_table(str(pqf))
        total_frames += len(t)
        all_eps |= set(t.column("episode_index").to_pylist())

    info["total_episodes"] = len(all_eps)
    info["total_frames"] = total_frames
    if all_eps:
        info["splits"] = {"train": f"0:{len(all_eps)}"}

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # 4. Nuke stats.json (will be rebuilt on next load)
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        stats_path.unlink()
        print("  Deleted stats.json (will be rebuilt on next load)")

    print(f"\nRemoved episodes: {sorted(remove_eps)}")
    print(f"Remaining: {len(all_eps)} episodes, {total_frames} frames")
    print("Ready to resume recording.")


if __name__ == "__main__":
    import pyarrow as pa
    main()
