#!/usr/bin/env python3
"""Capture the current Piper qpos as the adapter v2 collection start pose.

Read-only save: sends no motion, writes no dataset.
The saved pose is used by --start-pose-file in record_adapter_v2.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapter_v2.piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from adapter_v2.schema import STATE_DIM, ZONE_GRIPPER_OPEN_MIN_M, as_qpos

DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "adapter_v2_start_pose.json"


def fmt_qpos(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture adapter v2 collection start pose (read-only, no motion)."
    )
    parser.add_argument("--can-port", default="can0")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path for the start pose JSON file (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    print("=" * 64)
    print("Adapter v2 — capture collection start pose")
    print(f"  can_port : {args.can_port}")
    print(f"  output   : {output_path}")
    print("  motion   : none (read-only)")
    print("=" * 64)
    print()

    bus = PiperMotorsBusV2(PiperMotorsBusV2Config(can_port=args.can_port))
    try:
        bus.connect()
        qpos = bus.read_qpos()
    finally:
        bus.disconnect()

    qpos_arr = as_qpos(qpos, label="captured start qpos")
    print(f"Current qpos: {fmt_qpos(qpos_arr)}")
    print(f"  gripper  : {float(qpos_arr[6]):.6f} m")
    print()

    if float(qpos_arr[6]) < ZONE_GRIPPER_OPEN_MIN_M:
        print(f"  WARNING: gripper {float(qpos_arr[6]):.5f} m < {ZONE_GRIPPER_OPEN_MIN_M} m (zone min).")
        print("  For collection, the gripper should be open.")
        print()

    answer = input("Save this qpos as the adapter v2 collection start pose? Type SAVE to confirm: ").strip()
    if answer != "SAVE":
        print("Cancelled. No file written.")
        return 0

    data = {
        "qpos": fmt_qpos(qpos_arr),
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "source": "adapter_v2_capture_start_pose.py",
        "note": "adapter v2 collection start pose",
        "gripper_m": float(qpos_arr[6]),
        "state_dim": STATE_DIM,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  Saved: {output_path}")
    print("Ready for: --start-pose-file " + str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
