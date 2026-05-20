#!/usr/bin/env python3
"""Save current waypoints JSON as a versioned success baseline.

Usage:
  python3 scripts/save_success_waypoints.py \
      --input configs/bottle_grasp_waypoints_today.json

  python3 scripts/save_success_waypoints.py \
      --input configs/bottle_grasp_waypoints_today.json \
      --output configs/bottle_grasp_waypoints_success_v1.json
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_KEYS = [
    "start_pose", "pre_grasp_pose", "approach_pose",
    "close_gripper_pose", "lift_pose",
]


def next_versioned_path(base: Path) -> Path:
    """Return the next available vN filename, e.g. success_v1 → success_v2."""
    stem = base.stem
    suffix = base.suffix
    parent = base.parent
    v = 1
    while True:
        candidate = parent / f"{stem}_v{v}{suffix}"
        if not candidate.exists():
            return candidate
        v += 1


def main():
    parser = argparse.ArgumentParser(description="Version-copy a successful waypoints file")
    parser.add_argument("--input", type=Path, required=True,
                        help="Source waypoints JSON (e.g. bottle_grasp_waypoints_today.json)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Target path. Auto-generates vN if not specified.")
    args = parser.parse_args()

    src = args.input.resolve()
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    data = json.loads(src.read_text())

    # Validate required pose keys
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise SystemExit(f"Missing required keys in waypoints file: {missing}")

    dst = args.output
    if dst is None:
        dst = next_versioned_path(
            PROJECT_ROOT / "configs" / "bottle_grasp_waypoints_success.json"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Enrich metadata
    out = {
        "source": data.get("source", "unknown"),
        "source_file": str(src.name),
        "saved_at": datetime.now().isoformat(),
        "version": dst.stem,
        "open_gripper": data.get("open_gripper", 0.10),
        "close_gripper": data.get("close_gripper", 0.0),
        "notes": data.get("notes", ""),
    }
    for k in REQUIRED_KEYS:
        out[k] = data[k]
    if "source_frames" in data:
        out["source_frames"] = data["source_frames"]

    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  Saved success waypoints → {dst}")
    print(f"  open_gripper={out['open_gripper']}  close_gripper={out['close_gripper']}")
    print(f"\n  Next step:")
    print(f"  python3 scripts/quick_bottle_grasp.py \\")
    print(f"      --waypoints {dst} \\")
    print(f"      --step-confirm --velocity-pct 50 --log-result")


if __name__ == "__main__":
    main()
