#!/usr/bin/env python3
"""Human-confirmed gripper test for adapter v2."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapter_v2.piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from adapter_v2.schema import GRIPPER_OPEN_M, GRIPPER_STRONG_CLOSE_MAX_M, PIPER_GRIPPER_MAX_M


def sweep_gripper(
    bus,
    target_m: float,
    *,
    hz: float,
    max_step_m: float,
    settle_s: float,
    velocity_pct: int,
):
    """Sweep the gripper while repeatedly holding the current arm joints."""
    start = bus.read_qpos()
    target_m = max(0.0, min(float(target_m), PIPER_GRIPPER_MAX_M))
    n_steps = max(1, math.ceil(abs(target_m - float(start[6])) / max_step_m))
    arm_hold = start[:6].copy()
    print(
        f"  sweep target={target_m:.6f} m from={float(start[6]):.6f} m "
        f"steps={n_steps} max_step={max_step_m:.4f} m"
    )

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        target = start.copy()
        target[:6] = arm_hold
        target[6] = float(start[6]) + (target_m - float(start[6])) * alpha
        bus.write_qpos(target, velocity_pct=velocity_pct)
        time.sleep(1.0 / hz)

    ramp_feedback = bus.read_qpos()
    print(f"    ramp feedback: {float(ramp_feedback[6]):.6f} m")

    hold_steps = max(1, math.ceil(settle_s * hz))
    hold = ramp_feedback.copy()
    hold[:6] = arm_hold
    hold[6] = target_m
    for _ in range(hold_steps):
        bus.write_qpos(hold, velocity_pct=velocity_pct)
        time.sleep(1.0 / hz)
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
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-step-gripper", type=float, default=0.004)
    parser.add_argument("--velocity-pct", type=int, default=25)
    args = parser.parse_args()
    if args.hz <= 0:
        raise SystemExit("--hz must be positive.")
    if args.max_step_gripper <= 0:
        raise SystemExit("--max-step-gripper must be positive.")

    print("This script only sweeps the gripper while repeatedly holding the current arm qpos.")
    confirm(f"Open to {args.open:.4f} m?")

    bus = PiperMotorsBusV2(PiperMotorsBusV2Config(can_port=args.can_port))
    try:
        bus.connect()
        opened = sweep_gripper(
            bus,
            args.open,
            hz=args.hz,
            max_step_m=args.max_step_gripper,
            settle_s=args.settle_s,
            velocity_pct=args.velocity_pct,
        )
        print(f"  open feedback : {opened[6]:.6f} m")
        confirm(f"Close to the safe test value {args.safe_close:.4f} m?")
        closed = sweep_gripper(
            bus,
            args.safe_close,
            hz=args.hz,
            max_step_m=args.max_step_gripper,
            settle_s=args.settle_s,
            velocity_pct=args.velocity_pct,
        )
        print(f"  close feedback: {closed[6]:.6f} m")
        confirm(f"Reopen to {args.open:.4f} m?")
        reopened = sweep_gripper(
            bus,
            args.open,
            hz=args.hz,
            max_step_m=args.max_step_gripper,
            settle_s=args.settle_s,
            velocity_pct=args.velocity_pct,
        )
        print(f"  reopen feedback: {reopened[6]:.6f} m")
        print("RESULT: operator must visually confirm PASS.")
        print("Arm stays ENABLED at the current pose after this gripper-only test.")
        return 0
    finally:
        bus.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
