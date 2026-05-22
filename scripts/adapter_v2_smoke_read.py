#!/usr/bin/env python3
"""Read-only adapter-v2 smoke test for Piper state schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapter_v2.piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from adapter_v2.schema import STATE_DIM


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter v2 Piper read-only smoke test.")
    parser.add_argument("--can-port", default="can0")
    args = parser.parse_args()

    bus = PiperMotorsBusV2(PiperMotorsBusV2Config(can_port=args.can_port))
    try:
        bus.connect()
        qpos = bus.read_qpos()
        print("Adapter v2 Piper read-only smoke test")
        print(f"  can_port : {args.can_port}")
        print(f"  qpos     : {[round(float(value), 6) for value in qpos]}")
        print(f"  gripper  : {float(qpos[6]):.6f} m")
        print(f"  state dim: {qpos.shape[0]}")
        if qpos.shape != (STATE_DIM,) or not np.isfinite(qpos).all():
            print("RESULT: FAIL")
            return 1
        print("RESULT: PASS")
        return 0
    finally:
        bus.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
