#!/usr/bin/env python3
"""Record one adapter-v2 demo through the validated single-CAN mirror path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapter_v2.schema import STANDARD_START_QPOS, as_qpos
from teleop import data_collector

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "lerobot_dataset_piper_adapter_v2_one_demo"
DEFAULT_DATASET_REPO_ID = "piper/adapter_v2_one_demo"


def parse_q_start(text: str):
    if not text:
        return STANDARD_START_QPOS.copy()
    return as_qpos([float(value.strip()) for value in text.split(",")], label="--q-start")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapter-v2 one-CAN mirror recorder with reset guard preflight."
    )
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--global-camera", default="auto")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument(
        "--task-mode",
        choices=("approach_only", "full_pick_place"),
        default="full_pick_place",
    )
    parser.add_argument(
        "--q-start",
        default="",
        help="Optional comma-separated [j1,j2,j3,j4,j5,j6,gripper] start guard override.",
    )
    parser.add_argument(
        "--disable-motion-start-detect",
        action="store_true",
        help="Forwarded to the mirror recorder. Default keeps reset/idle motion out of the episode.",
    )
    args = parser.parse_args()
    q_start = parse_q_start(args.q_start)

    print()
    print("Opening single-CAN mirror recorder.")
    print("  The recorder will enforce the adapter-v2 start guard after Piper connects.")
    print("  SPACE starts/stops the saved episode after the guarded start pose.")
    print("  Save one episode, then quit with Q/ESC for adapter-v2 Step 4.")
    print("  Recorder disconnect keeps the v2 guard path torque-retaining.")
    print()

    data_collector.CAN_PORT = args.can_port
    forwarded = [
        "teleop/data_collector.py",
        "--task-mode",
        args.task_mode,
        "--dataset-root",
        args.dataset_root,
        "--dataset-repo-id",
        args.dataset_repo_id,
        "--record-gripper-action",
        "true",
        "--keep-enabled-on-exit",
        "--required-start-qpos",
        ",".join(f"{float(value):.6f}" for value in q_start),
        "--no-wrist",
        "--global-camera",
        args.global_camera,
    ]
    if args.disable_motion_start_detect:
        forwarded.append("--disable-motion-start-detect")

    previous_argv = sys.argv
    try:
        sys.argv = forwarded
        return data_collector.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
