#!/usr/bin/env python3
"""Build a LeRobot dataset with phase input and relative waypoint actions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def set_hf_cache_defaults(cache_dir: Path) -> None:
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text())


def read_parquet_tree(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def stack_vectors(series: pd.Series, key: str) -> np.ndarray:
    try:
        return np.stack([np.asarray(value, dtype=np.float32) for value in series.to_list()])
    except ValueError as exc:
        raise ValueError(f"Could not stack vector column '{key}'") from exc


def tensor_or_array_to_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype)
    return value


def output_features_from_info(info: dict, no_phase: bool = False) -> dict:
    features = {}
    for key, feature in info["features"].items():
        if key in DEFAULT_FEATURES:
            continue
        copied = dict(feature)
        if "shape" in copied:
            copied["shape"] = tuple(copied["shape"])
        features[key] = copied

    state_feature = dict(features["observation.state"])
    old_names = list(state_feature.get("names") or [f"s{i}" for i in range(state_feature["shape"][0])])
    if not no_phase:
        state_feature["shape"] = (state_feature["shape"][0] + 1,)
        state_feature["names"] = old_names + ["phase"]
    features["observation.state"] = state_feature

    action_feature = dict(features["action"])
    action_feature["shape"] = (7,)
    action_feature["names"] = ["dj1", "dj2", "dj3", "dj4", "dj5", "dj6", "dgripper"]
    features["action"] = action_feature
    return features


def compute_keep_range(
    states: np.ndarray,
    actions: np.ndarray,
    motion_threshold: float,
    preroll_frames: int,
    tail_frames: int,
    trim_motion: bool,
) -> tuple[int, int, int, int]:
    ep_len = len(states)
    if not trim_motion:
        return 0, ep_len, -1, -1

    motion = np.max(np.abs(actions[:, :6] - states[:, :6]), axis=1)
    moving = np.flatnonzero(motion > motion_threshold)
    if len(moving) == 0:
        return 0, ep_len, -1, -1
    first_motion = int(moving[0])
    last_motion = int(moving[-1])
    start = max(0, first_motion - preroll_frames)
    end = min(ep_len, last_motion + tail_frames)
    if end <= last_motion:
        end = min(ep_len, last_motion + 1)
    if end <= start:
        end = min(ep_len, start + 1)
    return start, end, first_motion, last_motion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="piper/bottle_grasp")
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument("--delta-horizon-frames", type=int, default=5,
                        help="Predict state[t+horizon] - state[t] as the action label.")
    parser.add_argument("--no-phase", action="store_true",
                        help="Do NOT append phase to observation.state. Keeps it at native 7D.")
    parser.add_argument("--trim-motion", action="store_true",
                        help="Also trim static prefix/suffix while rebuilding.")
    parser.add_argument("--motion-threshold", type=float, default=0.005)
    parser.add_argument("--preroll-frames", type=int, default=5)
    parser.add_argument("--tail-frames", type=int, default=8)
    parser.add_argument("--episode", type=int, action="append", default=None,
                        help="Only include this episode. May be provided multiple times.")
    parser.add_argument("--report-path", type=Path, default=Path("reports/delta_phase_report.csv"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/piper_act_hf_cache"))
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--encoder-threads", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.delta_horizon_frames < 1:
        raise ValueError("--delta-horizon-frames must be >= 1")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input dataset root does not exist: {args.input_root}")
    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {args.output_root}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)

    set_hf_cache_defaults(args.cache_dir)
    sys.path.insert(0, str(PROJECT_ROOT))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = load_info(args.input_root)
    fps = int(info.get("fps", 30))
    output_repo_id = args.output_repo_id or f"{args.repo_id}_delta_phase"
    features = output_features_from_info(info, no_phase=args.no_phase)
    use_videos = any(feature.get("dtype") == "video" for feature in features.values())

    data_df = read_parquet_tree(args.input_root / "data").reset_index(drop=True)
    data_df["_row_pos"] = np.arange(len(data_df), dtype=np.int64)
    episodes = sorted(int(ep) for ep in data_df["episode_index"].unique())
    if args.episode is not None:
        requested = set(args.episode)
        episodes = [ep for ep in episodes if ep in requested]
        missing = requested - set(episodes)
        if missing:
            raise ValueError(f"Requested episodes not found: {sorted(missing)}")

    print(f"Loading source dataset: {args.input_root}")
    source = LeRobotDataset(
        args.repo_id,
        root=args.input_root,
        return_uint8=True,
        video_backend=args.video_backend,
    )

    print(f"Creating delta+phase dataset: {args.output_root}")
    target = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        features=features,
        root=args.output_root,
        robot_type=info.get("robot_type"),
        use_videos=use_videos,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        data_files_size_in_mb=info.get("data_files_size_in_mb"),
        video_files_size_in_mb=info.get("video_files_size_in_mb"),
    )

    rows = []
    try:
        for ep_id in episodes:
            ep_df = data_df[data_df["episode_index"] == ep_id].copy()
            ep_df = ep_df.sort_values("index" if "index" in ep_df else "frame_index")
            states = stack_vectors(ep_df["observation.state"], "observation.state")[:, :7]
            actions = stack_vectors(ep_df["action"], "action")[:, :7]
            start, end, first_motion, last_motion = compute_keep_range(
                states,
                actions,
                args.motion_threshold,
                args.preroll_frames,
                args.tail_frames,
                args.trim_motion,
            )

            kept = ep_df.iloc[start:end].copy()
            kept_states = states[start:end]
            old_len = len(ep_df)
            new_len = len(kept)
            if new_len == 0:
                raise RuntimeError(f"Episode {ep_id} trim range is empty")

            for local_idx, (_, raw_row) in enumerate(kept.iterrows()):
                src_item = source[int(raw_row["_row_pos"])]
                state7 = kept_states[local_idx].astype(np.float32)
                future_idx = min(local_idx + args.delta_horizon_frames, new_len - 1)
                future_state7 = kept_states[future_idx].astype(np.float32)
                delta_action = future_state7 - state7
                phase = 0.0 if new_len <= 1 else local_idx / float(new_len - 1)

                frame = {"task": src_item.get("task", "Grasp the bottle from the table")}
                for key, feature in features.items():
                    if key == "observation.state":
                        if args.no_phase:
                            frame[key] = state7.copy()
                        else:
                            frame[key] = np.concatenate([state7, np.asarray([phase], dtype=np.float32)])
                    elif key == "action":
                        frame[key] = delta_action.astype(np.float32)
                    elif feature["dtype"] in ("image", "video"):
                        frame[key] = tensor_or_array_to_numpy(src_item[key], dtype=np.uint8)
                    else:
                        frame[key] = tensor_or_array_to_numpy(src_item[key], dtype=np.float32)
                target.add_frame(frame)

            target.save_episode()

            delta_norm = np.linalg.norm(
                np.stack([
                    kept_states[min(i + args.delta_horizon_frames, new_len - 1)] - kept_states[i]
                    for i in range(new_len)
                ])[:, :6],
                axis=1,
            )
            row = {
                "episode_id": ep_id,
                "old_len": old_len,
                "new_len": new_len,
                "removed_prefix": int(start),
                "removed_suffix": int(old_len - end),
                "first_motion": first_motion,
                "last_motion": last_motion,
                "delta_horizon_frames": args.delta_horizon_frames,
                "delta_arm_norm_mean": float(np.mean(delta_norm)),
                "delta_arm_norm_p95": float(np.quantile(delta_norm, 0.95)),
                "delta_arm_norm_max": float(np.max(delta_norm)),
            }
            rows.append(row)
            print(
                f"ep {ep_id:03d}: {old_len}->{new_len}, "
                f"delta_norm mean/p95/max="
                f"{row['delta_arm_norm_mean']:.4f}/"
                f"{row['delta_arm_norm_p95']:.4f}/"
                f"{row['delta_arm_norm_max']:.4f}"
            )
    finally:
        target.finalize()

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote delta+phase dataset to {args.output_root}")
    print(f"Wrote report to {args.report_path}")
    print("Train this dataset with --policy.n_action_steps=1 or deploy with --action-mode delta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
