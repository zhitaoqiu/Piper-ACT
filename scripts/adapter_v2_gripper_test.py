#!/usr/bin/env python3
"""Human-confirmed gripper test for adapter v2."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapter_v2.piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from adapter_v2.reset import open_gripper
from adapter_v2.schema import GRIPPER_OPEN_M, GRIPPER_STRONG_CLOSE_MAX_M


def command_gripper(bus, target_m: float, settle_s: float):
    qpos = bus.read_qpos()
    qpos[6] = target_m
    bus.write_qpos(qpos)
    time.sleep(settle_s)
    return bus.read_qpos()


def confirm(message: str) -> None:
    answer = input(f"{message} Type YES to continue: ").strip()
    if answer != "YES":
        raise SystemExit("Cancelled before motion.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter v2 Piper gripper open/close/reopen test.")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--open", type=float, default=GRIPPER_OPEN_M)
    parser.add_argument("--safe-close", type=float, default=GRIPPER_STRONG_CLOSE_MAX_M)
    parser.add_argument("--settle-s", type=float, default=0.5)
    args = parser.parse_args()

    print("This script only moves the gripper while holding the current arm qpos.")
    confirm(f"Open to {args.open:.4f} m?")

    bus = PiperMotorsBusV2(PiperMotorsBusV2Config(can_port=args.can_port))
    try:
        bus.connect()
        opened = open_gripper(bus, args.open, settle_s=args.settle_s)
        print(f"  open feedback : {opened[6]:.6f} m")
        confirm(f"Close to the safe test value {args.safe_close:.4f} m?")
        closed = command_gripper(bus, args.safe_close, args.settle_s)
        print(f"  close feedback: {closed[6]:.6f} m")
        confirm(f"Reopen to {args.open:.4f} m?")
        reopened = open_gripper(bus, args.open, settle_s=args.settle_s)
        print(f"  reopen feedback: {reopened[6]:.6f} m")
        print("RESULT: operator must visually confirm PASS.")
        print("Arm stays ENABLED at the current pose after this gripper-only test.")
        return 0
    finally:
        bus.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
