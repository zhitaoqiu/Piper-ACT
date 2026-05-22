#!/usr/bin/env python3
"""Reset Piper adapter v2 to the baseline standard start qpos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapter_v2.piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from adapter_v2.reset import qpos_diff, reset_to_standard_start
from adapter_v2.schema import STANDARD_START_QPOS, as_qpos


def parse_q_start(text: str):
    if not text:
        return STANDARD_START_QPOS.copy()
    return as_qpos([float(value.strip()) for value in text.split(",")], label="--q-start")


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter v2 reset-to-standard-start.")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument(
        "--q-start",
        default="",
        help="Optional comma-separated [j1,j2,j3,j4,j5,j6,gripper] override.",
    )
    args = parser.parse_args()
    q_start = parse_q_start(args.q_start)

    print("Adapter v2 reset-to-standard-start")
    print(f"  q_start: {[round(float(value), 6) for value in q_start]}")
    answer = input("Type RESET to move the arm: ").strip()
    if answer != "RESET":
        print("Cancelled before motion.")
        return 1

    bus = PiperMotorsBusV2(PiperMotorsBusV2Config(can_port=args.can_port))
    try:
        bus.connect()
        ok, final_qpos, steps = reset_to_standard_start(bus, q_start=q_start, hz=args.hz)
        print(f"  steps    : {steps}")
        print(f"  final    : {[round(float(value), 6) for value in final_qpos]}")
        print(f"  abs diff : {[round(float(value), 6) for value in qpos_diff(final_qpos, q_start)]}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        bus.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
