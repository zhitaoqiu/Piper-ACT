#!/usr/bin/env python3
"""Select and export ~30 clean old single-camera Piper demos.

Default source:
  /home/huatec/piper_diffusion_bottle_grasp-master/data/lerobot_dataset_env2_30fixed

Default target:
  data/lerobot_dataset_piper_bottle_old_singlecam_30demo_clean/

The script ranks all 40 source episodes with deterministic quality heuristics,
then exports the top 30 clean episodes into a fresh LeRobot dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception as exc:
    print("[ERROR] pandas could not be imported. Activate the piper_act environment first.")
    print(f"        import error: {exc}")
    sys.exit(2)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path("/home/huatec/piper_diffusion_bottle_grasp-master/data/lerobot_dataset_env2_30fixed")
DEFAULT_TARGET = PROJECT_ROOT / "data" / "lerobot_dataset_piper_bottle_old_singlecam_30demo_clean"
DEFAULT_REPO_ID = "piper/old_singlecam_30demo_clean"

EXPECTED_STATE_DIM = 7
EXPECTED_ACTION_DIM = 7
EXPECTED_OPEN = 0.0995
OPEN_TOL = 0.012          # allow grip_start 0.0875-0.1115
ONSET_DROP = 0.015         # grip must drop at least 15mm from open
CLOSE_MIN = 0.035          # absolute minimum grip (bottle width floor)
CLOSE_MAX = 0.070          # absolute maximum grip (must actually close)
IDEAL_CLOSE_MIN = 0.045    # ideal close range
IDEAL_CLOSE_MAX = 0.058
MIN_FRAMES = 120            # too short = bad demo
RELEASE_GRIP = 0.075        # grip must rise back above this after close


@dataclass
class EpisodeScore:
    episode: int
    frames: int
    score: float
    pass_basic: bool
    gripper_start: float
    gripper_min: float
    gripper_drop: float
    gripper_end: float
    first_close_frame: int
    release_frame: int
    failures: list[str]
    warnings: list[str]


@contextlib.contextmanager
def suppress_stderr_fd():
    saved_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_parquet_data(dataset_root: Path) -> pd.DataFrame:
    files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def stack_column(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in series.to_list()])


def video_frame_count(path: Path) -> int | None:
    try:
        import cv2
    except Exception:
        return None
    with suppress_stderr_fd():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return count if count > 0 else None
        finally:
            cap.release()


def camera_keys_from_features(features: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key, value in features.items()
        if key.startswith("observation.images.") and value.get("dtype") == "video"
    )


def user_features_from_info(info: dict[str, Any], camera_key: str) -> dict[str, Any]:
    features = info["features"]
    keep = ["observation.state", "action", camera_key]
    return {key: features[key] for key in keep}


def check_source_images(dataset_root: Path, camera_key: str, total_rows: int) -> tuple[bool, str]:
    files = sorted((dataset_root / "videos" / camera_key).glob("chunk-*/file-*.mp4"))
    if not files:
        return False, f"missing video files for {camera_key}"
    counts = [video_frame_count(path) for path in files]
    if any(count is None for count in counts):
        return False, f"could not read frame count for {camera_key}: {files}"
    total = int(sum(counts))
    if total < total_rows:
        return False, f"{camera_key} video frames {total} < parquet rows {total_rows}"
    return True, f"{camera_key} video frames {total} for {total_rows} parquet rows"


def score_episode(ep: int, edf: pd.DataFrame) -> EpisodeScore:
    qpos = stack_column(edf["observation.state"])
    action = stack_column(edf["action"])
    failures: list[str] = []
    warnings: list[str] = []

    frames = len(edf)
    if frames < MIN_FRAMES:
        failures.append(f"too few frames ({frames} < {MIN_FRAMES})")
    if qpos.shape[1] != EXPECTED_STATE_DIM:
        failures.append(f"qpos dim {qpos.shape[1]} != {EXPECTED_STATE_DIM}")
    if action.shape[1] != EXPECTED_ACTION_DIM:
        failures.append(f"action dim {action.shape[1]} != {EXPECTED_ACTION_DIM}")
    if not np.isfinite(qpos).all() or not np.isfinite(action).all():
        failures.append("NaN/Inf in qpos or action")
    if np.max(np.abs(action)) < 1e-8 or np.max(np.ptp(action, axis=0)) < 1e-5:
        failures.append("all-zero or constant actions")

    grip = qpos[:, 6]
    grip_start = float(grip[0])
    grip_min = float(np.min(grip))
    grip_end = float(grip[-1])
    grip_drop = grip_start - grip_min
    close_candidates = np.where(grip < grip_start - ONSET_DROP)[0]
    first_close = int(close_candidates[0]) if len(close_candidates) else -1

    # Release detection: after grip reaches minimum, it rises back above RELEASE_GRIP.
    # We look from the grip-minimum point (deepest close) onward to find the reopen.
    release_frame = -1
    if first_close >= 0:
        grip_min_idx = first_close + int(np.argmin(grip[first_close:]))
        post_min_grip = grip[grip_min_idx:]
        release_rel = np.where(post_min_grip > RELEASE_GRIP)[0]
        if len(release_rel) > 0:
            release_frame = grip_min_idx + int(release_rel[0])

    # Basic pass/fail
    if abs(grip_start - EXPECTED_OPEN) > OPEN_TOL:
        failures.append(f"gripper start {grip_start:.5f} not near {EXPECTED_OPEN:.5f}")
    if first_close < 0:
        failures.append("no open-to-close transition")
    if not (CLOSE_MIN <= grip_min <= CLOSE_MAX):
        failures.append(f"gripper min {grip_min:.5f} outside [{CLOSE_MIN:.3f}, {CLOSE_MAX:.3f}]")
    elif not (IDEAL_CLOSE_MIN <= grip_min <= IDEAL_CLOSE_MAX):
        warnings.append(f"gripper min {grip_min:.5f} outside ideal [{IDEAL_CLOSE_MIN:.3f}, {IDEAL_CLOSE_MAX:.3f}]")
    if release_frame < 0:
        failures.append("no release/reopen after close")

    # Quality scoring (lower = better)
    median_target = 185.0
    score = 0.0
    score += abs(frames - median_target) / 30.0
    score += abs(grip_start - EXPECTED_OPEN) * 80.0
    score += abs(grip_min - 0.050) * 50.0
    score += 0.0 if first_close >= 0 else 100.0
    if first_close >= 0:
        # prefer close happening around 30-40% into trajectory
        ideal_close_pct = 0.35
        close_pct = first_close / max(frames, 1)
        score += abs(close_pct - ideal_close_pct) * 12.0
    if release_frame >= 0:
        # reward clear release
        score -= 1.0
        release_pct = release_frame / max(frames, 1)
        # prefer release at 80-95% (near end)
        if release_pct < 0.70:
            score += (0.70 - release_pct) * 5.0
    if warnings:
        score += 0.25 * len(warnings)
    if failures:
        score += 1000.0 + 50.0 * len(failures)

    return EpisodeScore(
        episode=ep,
        frames=frames,
        score=score,
        pass_basic=not failures,
        gripper_start=grip_start,
        gripper_min=grip_min,
        gripper_drop=grip_drop,
        gripper_end=grip_end,
        first_close_frame=first_close,
        release_frame=release_frame,
        failures=failures,
        warnings=warnings,
    )


def print_ranking(scores: list[EpisodeScore], selected: set[int], args) -> None:
    print(f"Episode ranking (target: {args.num_episodes}), lower score is cleaner:\n")
    header = f"{'ep':>3}  {'pick':>4}  {'score':>8}  {'frames':>6}  {'start':>10}  {'min':>8}  {'end':>10}  {'drop':>6}  {'close@':>8}  {'release@':>9}  {'status':>5}"
    print(header)
    print("-" * len(header))
    for s in scores:
        mark = " * " if s.episode in selected else "   "
        status = "PASS" if s.pass_basic else "FAIL"
        rel_str = f"{s.release_frame:5d}" if s.release_frame >= 0 else "     -"
        close_str = f"{s.first_close_frame:5d}" if s.first_close_frame >= 0 else "     -"
        print(
            f"{s.episode:3d}  {mark}  {s.score:8.3f}  {s.frames:6d}  "
            f"{s.gripper_start:10.5f}  {s.gripper_min:8.5f}  {s.gripper_end:10.5f}  "
            f"{s.gripper_drop:6.4f}  {close_str}  {rel_str}  {status}"
        )
        for failure in s.failures:
            print(f"      FAIL: {failure}")
        for warning in s.warnings:
            print(f"      WARN: {warning}")
    print()


def prepare_cache_env() -> None:
    cache_root = Path(os.environ.get("HF_CACHE_ROOT", "/tmp/piper_act_hf_cache"))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


def export_dataset(
    source_root: Path,
    target_root: Path,
    repo_id: str,
    selected_episodes: list[int],
    info: dict[str, Any],
    camera_key: str,
    source_df: pd.DataFrame,
) -> None:
    prepare_cache_env()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = user_features_from_info(info, camera_key)
    fps = int(info.get("fps", 10))
    target = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=target_root,
        use_videos=True,
        vcodec="libsvtav1",
        image_writer_threads=4,
    )
    source = LeRobotDataset(
        repo_id="piper/old_singlecam_source",
        root=source_root,
        video_backend="pyav",
        return_uint8=True,
    )

    try:
        for new_ep, old_ep in enumerate(selected_episodes):
            edf = source_df[source_df["episode_index"] == old_ep].sort_values("frame_index")
            n = len(edf)
            print(f"Exporting old ep {old_ep:03d} -> new ep {new_ep:03d} ({n} frames)")
            for frame_i, row in enumerate(edf.itertuples(index=False)):
                global_index = int(getattr(row, "index"))
                item = source[global_index]
                frame = {
                    "observation.state": item["observation.state"],
                    "action": item["action"],
                    camera_key: item[camera_key],
                    "task": item.get("task", "pick up the bottle"),
                }
                target.add_frame(frame)
                if frame_i == 0 or (frame_i + 1) % 100 == 0 or frame_i + 1 == n:
                    print(f"  frame {frame_i + 1:03d}/{n}")
            target.save_episode()
    finally:
        target.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank and export best old single-camera Piper demos.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source old LeRobot dataset root.")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="Fresh target dataset root.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--camera-key", default=None, help="Expected single camera key. Auto-detect when omitted.")
    parser.add_argument("--dry-run", action="store_true", help="Only rank; do not export.")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    target_root = Path(args.target).expanduser().resolve()
    if not source_root.exists():
        print(f"[ERROR] Source dataset not found: {source_root}")
        return 1
    if target_root.exists() and any(target_root.iterdir()) and not args.dry_run:
        print(f"[ERROR] Target already exists and is not empty: {target_root}")
        print("        Refusing to overwrite. Move it aside manually if you want to regenerate.")
        return 1

    info = load_json(source_root / "meta" / "info.json")
    features = info.get("features", {})
    camera_keys = camera_keys_from_features(features)
    if args.camera_key:
        camera_key = args.camera_key
        if camera_key not in camera_keys:
            print(f"[ERROR] Requested camera key {camera_key!r} not in source cameras: {camera_keys}")
            return 1
    else:
        if len(camera_keys) != 1:
            print(f"[ERROR] Expected one source camera, found {camera_keys}")
            return 1
        camera_key = camera_keys[0]

    df = load_parquet_data(source_root)
    episodes = sorted(int(ep) for ep in df["episode_index"].unique())
    image_ok, image_msg = check_source_images(source_root, camera_key, len(df))

    print("=" * 78)
    print("Old Single-Camera Piper 30-Demo Clean Dataset Selector")
    print("=" * 78)
    print(f"Source:       {source_root}")
    print(f"Target:       {target_root}")
    print(f"Episodes:     {len(episodes)}")
    print(f"Frames:       {len(df)}")
    print(f"FPS:          {info.get('fps')}")
    print(f"Camera key:   {camera_key}")
    print(f"Images:       {'OK' if image_ok else 'FAIL'} - {image_msg}")
    print()

    if not image_ok:
        print("[ERROR] Source image validation failed. Not exporting.")
        return 1

    scores = [score_episode(ep, df[df["episode_index"] == ep].sort_values("frame_index")) for ep in episodes]
    ranked = sorted(scores, key=lambda s: (not s.pass_basic, s.score, s.episode))
    passing = [s for s in ranked if s.pass_basic]
    print(f"Passing episodes: {len(passing)} / {len(ranked)}\n")

    if len(passing) < args.num_episodes:
        print(f"[INFO] Only {len(passing)} passing episodes; will export all {len(passing)}.")
        print(f"       (requested {args.num_episodes})")
        num_to_export = len(passing)
    else:
        num_to_export = args.num_episodes

    selected = [s.episode for s in passing[:num_to_export]]
    print_ranking(ranked, set(selected), args)
    print(f"Selected {len(selected)} episodes: {selected}")

    if args.dry_run:
        print("Dry run complete. No dataset exported.")
        return 0

    export_dataset(source_root, target_root, args.repo_id, selected, info, camera_key, df)
    manifest = {
        "source_dataset": str(source_root),
        "target_dataset": str(target_root),
        "repo_id": args.repo_id,
        "camera_key": camera_key,
        "selected_episodes": selected,
        "num_passing": len(passing),
        "num_total": len(ranked),
    }
    (target_root / "selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print()
    print(f"Export complete: {target_root}")
    print(f"Total episodes:  {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
