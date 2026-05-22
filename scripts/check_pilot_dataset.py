#!/usr/bin/env python3
"""Sanity check the 10-demo Piper ACT pilot dataset.

This script does not train and does not touch the robot.
It inspects LeRobot v3-style parquet/video data and exits non-zero unless
at least 8 episodes pass.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - environment diagnostic
    print("[ERROR] pandas could not be imported. Activate the piper_act environment first.")
    print(f"        import error: {exc}")
    sys.exit(2)


EXPECTED_STATE_DIM = 7
EXPECTED_ACTION_DIM = 7
EXPECTED_GRIPPER_OPEN = 0.0995
GRIPPER_OPEN_TOL = 0.010
GRIPPER_ONSET_DROP = 0.015
GRIPPER_STRONG_CLOSE_MIN = 0.035
GRIPPER_STRONG_CLOSE_MAX = 0.070
GRIPPER_IDEAL_CLOSE_MIN = 0.045
GRIPPER_IDEAL_CLOSE_MAX = 0.060
MIN_FRAMES_PER_EPISODE = 80
MAX_FRAMES_PER_EPISODE = 1200
DEFAULT_MIN_PASS_EPISODES = 8
MIN_LIFT_FRAMES_AFTER_CLOSE = 20
MIN_POST_CLOSE_ARM_RANGE = 0.025


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_parquet_data(dataset_root: Path) -> pd.DataFrame:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset_root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)


def array_column(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in series.to_list()])


def finite_min_max(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)


def fmt_scalar(value: float | int | None, precision: int = 5) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    return f"{float(value):.{precision}f}"


def fmt_vec(values: np.ndarray, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def video_frame_count(path: Path) -> int | None:
    try:
        import cv2
    except Exception:
        return None

    # OpenCV/FFmpeg can print very noisy AV1 hardware-decoder warnings even
    # when frame counting succeeds. Keep the sanity output focused.
    with suppress_stderr_fd():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count > 0:
                return count
            count = 0
            while True:
                ok, _frame = cap.read()
                if not ok:
                    break
                count += 1
            return count
        finally:
            cap.release()


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


def camera_totals(dataset_root: Path, camera_keys: list[str]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for key in camera_keys:
        folder = dataset_root / "videos" / key
        files = sorted(folder.glob("chunk-*/file-*.mp4"))
        frame_counts = [video_frame_count(path) for path in files]
        known_counts = [count for count in frame_counts if count is not None]
        totals[key] = {
            "files": files,
            "frame_counts": frame_counts,
            "total_frames": sum(known_counts) if len(known_counts) == len(frame_counts) else None,
            "unknown": len(known_counts) != len(frame_counts),
        }
    return totals


def camera_episode_counts(
    camera_info: dict[str, dict[str, Any]],
    expected_total_frames: int,
    episode_frames: int,
) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for key, info in camera_info.items():
        total_frames = info["total_frames"]
        if not info["files"] or total_frames is None:
            counts[key] = None
        elif total_frames == expected_total_frames:
            counts[key] = episode_frames
        else:
            counts[key] = -1
    return counts


def episode_lift_heuristic(qpos: np.ndarray, close_onset: int) -> tuple[bool, str]:
    if close_onset < 0:
        return False, "no close onset"
    remaining = qpos[close_onset + 1 :]
    if len(remaining) < MIN_LIFT_FRAMES_AFTER_CLOSE:
        return False, f"only {len(remaining)} frames after close onset"
    arm_range = np.ptp(remaining[:, :6], axis=0)
    max_post_close_range = float(np.max(arm_range))
    if max_post_close_range < MIN_POST_CLOSE_ARM_RANGE:
        return False, f"post-close arm range too small ({max_post_close_range:.4f})"
    return True, f"post-close arm range {max_post_close_range:.4f}"


def analyze_episode(
    ep: int,
    edf: pd.DataFrame,
    features: dict[str, Any],
    camera_info: dict[str, dict[str, Any]],
    total_rows: int,
    expected_start_qpos: np.ndarray | None = None,
    start_arm_tol: float = 0.04,
    start_gripper_tol: float = 0.01,
) -> tuple[bool, list[str]]:
    n_frames = len(edf)
    failures: list[str] = []
    warnings: list[str] = []

    qpos = array_column(edf["observation.state"])
    action = array_column(edf["action"])

    image_counts = camera_episode_counts(camera_info, total_rows, n_frames)
    qpos_min, qpos_max = finite_min_max(qpos)
    action_min, action_max = finite_min_max(action)

    qpos_shape = tuple(qpos.shape[1:])
    action_shape = tuple(action.shape[1:])
    qgrip = qpos[:, 6]
    agrip = action[:, 6]
    first_grip = float(qgrip[0])
    min_grip = float(np.nanmin(qgrip))
    onset_threshold = first_grip - GRIPPER_ONSET_DROP
    onset_candidates = np.where(qgrip < onset_threshold)[0]
    first_onset = int(onset_candidates[0]) if len(onset_candidates) else -1
    lift_present, lift_reason = episode_lift_heuristic(qpos, first_onset)
    start_diff = None

    if n_frames < MIN_FRAMES_PER_EPISODE:
        failures.append(f"frame count too low ({n_frames} < {MIN_FRAMES_PER_EPISODE})")
    if n_frames > MAX_FRAMES_PER_EPISODE:
        warnings.append(f"frame count high ({n_frames} > {MAX_FRAMES_PER_EPISODE})")
    if qpos.shape[1] != EXPECTED_STATE_DIM:
        failures.append(f"qpos dim {qpos.shape[1]} != {EXPECTED_STATE_DIM}")
    if action.shape[1] != EXPECTED_ACTION_DIM:
        failures.append(f"action dim {action.shape[1]} != {EXPECTED_ACTION_DIM}")
    if not np.isfinite(qpos).all() or not np.isfinite(action).all():
        failures.append("contains NaN or Inf")
    if expected_start_qpos is not None and qpos.shape[1] == EXPECTED_STATE_DIM:
        start_diff = np.abs(qpos[0] - expected_start_qpos)
        if np.any(start_diff[:6] > start_arm_tol) or start_diff[6] > start_gripper_tol:
            failures.append(
                "start qpos differs from expected start beyond "
                f"arm/gripper tolerance ({start_arm_tol:.4f} rad, {start_gripper_tol:.4f} m)"
            )
    if np.max(np.abs(action)) < 1e-8:
        failures.append("all-zero actions")
    if np.max(np.ptp(action, axis=0)) < 1e-5:
        failures.append("action trajectory is effectively constant")
    if abs(first_grip - EXPECTED_GRIPPER_OPEN) > GRIPPER_OPEN_TOL:
        failures.append(
            f"gripper does not start open enough ({first_grip:.5f}, expected about {EXPECTED_GRIPPER_OPEN:.5f})"
        )
    if first_onset < 0:
        failures.append(f"no gripper drop below open - {GRIPPER_ONSET_DROP:.3f}")
    if not (GRIPPER_STRONG_CLOSE_MIN <= min_grip <= GRIPPER_STRONG_CLOSE_MAX):
        failures.append(
            f"gripper min {min_grip:.5f} outside broad close range "
            f"[{GRIPPER_STRONG_CLOSE_MIN:.3f}, {GRIPPER_STRONG_CLOSE_MAX:.3f}]"
        )
    elif not (GRIPPER_IDEAL_CLOSE_MIN <= min_grip <= GRIPPER_IDEAL_CLOSE_MAX):
        warnings.append(
            f"gripper min {min_grip:.5f} outside ideal strong-close range "
            f"[{GRIPPER_IDEAL_CLOSE_MIN:.3f}, {GRIPPER_IDEAL_CLOSE_MAX:.3f}]"
        )
    if not lift_present:
        failures.append(f"lift phase not apparent: {lift_reason}")

    for key, count in image_counts.items():
        if count is None:
            failures.append(f"{key} missing or unreadable video frames")
        elif count == -1:
            total = camera_info[key]["total_frames"]
            failures.append(f"{key} total video frames {total} != parquet rows {total_rows}")
        elif count != n_frames:
            failures.append(f"{key} frame count {count} != episode frames {n_frames}")

    status = "PASS" if not failures else "FAIL"
    print(f"Episode {ep:03d}: {status}")
    print(f"  frames: {n_frames}")
    print(f"  image frame count per camera: {image_counts}")
    print(f"  qpos shape: {qpos_shape}  action shape: {action_shape}")
    print(f"  qpos min:   {fmt_vec(qpos_min)}")
    print(f"  qpos max:   {fmt_vec(qpos_max)}")
    print(f"  action min: {fmt_vec(action_min)}")
    print(f"  action max: {fmt_vec(action_max)}")
    print(f"  gripper qpos min/max:   {float(np.nanmin(qgrip)):.5f} / {float(np.nanmax(qgrip)):.5f}")
    print(f"  gripper action min/max: {float(np.nanmin(agrip)):.5f} / {float(np.nanmax(agrip)):.5f}")
    print(f"  first frame gripper value: {first_grip:.5f}")
    if start_diff is not None:
        print(f"  start qpos abs diff:       {fmt_vec(start_diff, precision=5)}")
    print(f"  minimum gripper value:     {min_grip:.5f}")
    if first_onset >= 0:
        print(f"  first frame below open - {GRIPPER_ONSET_DROP:.3f}: {first_onset}")
    else:
        print(f"  first frame below open - {GRIPPER_ONSET_DROP:.3f}: NOT FOUND")
    print(f"  lift phase appears present: {lift_present} ({lift_reason})")
    if warnings:
        for warning in warnings:
            print(f"  warning: {warning}")
    if failures:
        for failure in failures:
            print(f"  fail: {failure}")
    print()

    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Piper 10-demo pilot dataset before ACT training.")
    parser.add_argument("--dataset", required=True, help="Dataset root, e.g. data/lerobot_dataset_piper_bottle_pilot_10demo/")
    parser.add_argument("--expected-episodes", type=int, default=None, help="Require exactly this many episodes.")
    parser.add_argument("--min-pass-episodes", type=int, default=DEFAULT_MIN_PASS_EPISODES)
    parser.add_argument("--camera-key", default=None, help="Require this exact camera key to exist.")
    parser.add_argument("--require-single-camera", action="store_true", help="Fail unless the dataset has exactly one video camera.")
    parser.add_argument(
        "--expected-start-qpos",
        default="",
        help="Optional comma-separated [j1,j2,j3,j4,j5,j6,gripper] start pose check.",
    )
    parser.add_argument("--start-arm-tol", type=float, default=0.04, help="Start-pose arm tolerance in rad.")
    parser.add_argument("--start-gripper-tol", type=float, default=0.01, help="Start-pose gripper tolerance in m.")
    args = parser.parse_args()

    expected_start_qpos = None
    if args.expected_start_qpos:
        try:
            expected_start_qpos = np.asarray(
                [float(value.strip()) for value in args.expected_start_qpos.split(",")],
                dtype=np.float32,
            )
        except ValueError as exc:
            print(f"[ERROR] Could not parse --expected-start-qpos: {exc}")
            return 1
        if expected_start_qpos.shape != (EXPECTED_STATE_DIM,) or not np.isfinite(expected_start_qpos).all():
            print(
                "[ERROR] --expected-start-qpos must contain 7 finite values: "
                "[j1,j2,j3,j4,j5,j6,gripper]"
            )
            return 1

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.exists():
        print(f"[ERROR] Dataset does not exist: {dataset_root}")
        return 1

    info = load_json(dataset_root / "meta" / "info.json")
    features = info.get("features", {})
    camera_keys = sorted(k for k, v in features.items() if k.startswith("observation.images.") and v.get("dtype") == "video")

    try:
        df = load_parquet_data(dataset_root)
    except Exception as exc:
        print(f"[ERROR] Could not load dataset parquet: {exc}")
        return 1

    required_columns = {"observation.state", "action", "episode_index"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        print(f"[ERROR] Missing required columns: {missing_columns}")
        return 1

    camera_info = camera_totals(dataset_root, camera_keys)
    episodes = sorted(int(ep) for ep in df["episode_index"].unique())
    global_failures: list[str] = []

    if args.expected_episodes is not None and len(episodes) != args.expected_episodes:
        global_failures.append(f"expected {args.expected_episodes} episodes, found {len(episodes)}")
    if args.require_single_camera and len(camera_keys) != 1:
        global_failures.append(f"expected exactly one camera, found {len(camera_keys)}: {camera_keys}")
    if args.camera_key is not None and args.camera_key not in camera_keys:
        global_failures.append(f"required camera key missing: {args.camera_key}")

    print("=" * 72)
    print("Piper Pilot 10-Demo Dataset Sanity Check")
    print("=" * 72)
    print(f"Dataset: {dataset_root}")
    print(f"Metadata episodes: {info.get('total_episodes', 'N/A')}")
    print(f"Parquet episodes:  {len(episodes)}")
    print(f"Total rows:        {len(df)}")
    print(f"FPS:               {info.get('fps', 'N/A')}")
    print(f"State feature:     {features.get('observation.state', {}).get('shape', 'N/A')}")
    print(f"Action feature:    {features.get('action', {}).get('shape', 'N/A')}")
    print(f"Cameras:           {camera_keys}")
    for key, cinfo in camera_info.items():
        print(f"  {key}: files={len(cinfo['files'])}, total_frames={cinfo['total_frames']}")
    print()

    if global_failures:
        print("Dataset-level failures:")
        for failure in global_failures:
            print(f"  fail: {failure}")
        print()

    passed = 0
    failed = 0
    for ep in episodes:
        edf = df[df["episode_index"] == ep].sort_values("frame_index" if "frame_index" in df.columns else "index")
        ok, _failures = analyze_episode(
            ep,
            edf,
            features,
            camera_info,
            len(df),
            expected_start_qpos=expected_start_qpos,
            start_arm_tol=args.start_arm_tol,
            start_gripper_tol=args.start_gripper_tol,
        )
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 72)
    print(f"Summary: {passed} passed, {failed} failed, {len(episodes)} total")
    if global_failures:
        print("RESULT: FAIL. Do not train. Dataset-level requirements failed.")
        return 1
    if passed < args.min_pass_episodes:
        print(f"RESULT: FAIL. Do not train. Need at least {args.min_pass_episodes} passing episodes.")
        return 1
    print(f"RESULT: PASS. Dataset is trainable ({passed} >= {args.min_pass_episodes}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
