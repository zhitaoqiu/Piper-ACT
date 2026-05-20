#!/usr/bin/env python3
"""Seal a successful waypoints file as versioned baseline with markdown log.

Usage (grasp):
  python3 scripts/save_success_baseline.py \
      --input configs/bottle_grasp_waypoints_today.json \
      --mode grasp --success-count 8 --attempt-count 8 --velocity-pct 50

Usage (pick_place):
  python3 scripts/save_success_baseline.py \
      --input configs/bottle_pick_place_waypoints_today.json \
      --mode pick_place --success-count 5 --attempt-count 5 --velocity-pct 50
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRASP_KEYS = [
    "start_pose", "pre_grasp_pose", "approach_pose",
    "close_gripper_pose", "lift_pose",
]

PICK_PLACE_KEYS = [
    "start_pose", "pre_grasp_pose", "approach_pose",
    "close_gripper_pose", "lift_pose",
    "place_pre_pose", "place_pose", "release_pose",
    "retreat_pose", "home_pose",
]


def next_versioned_path(base: Path):
    stem = base.stem
    suffix = base.suffix
    parent = base.parent
    v = 1
    while True:
        candidate = parent / f"{stem}_v{v}{suffix}"
        if not candidate.exists():
            return candidate, v
        v += 1


def main():
    parser = argparse.ArgumentParser(description="Seal successful waypoints as versioned baseline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=("grasp", "pick_place"), default="grasp")
    parser.add_argument("--success-count", type=int, required=True)
    parser.add_argument("--attempt-count", type=int, required=True)
    parser.add_argument("--velocity-pct", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    src = args.input.resolve()
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    data = json.loads(src.read_text())
    pose_keys = PICK_PLACE_KEYS if args.mode == "pick_place" else GRASP_KEYS

    # Validate all required poses are 7D
    for k in pose_keys:
        if k not in data:
            raise SystemExit(f"Missing key '{k}' in {src}")
        if len(data[k]) != 7:
            raise SystemExit(f"Key '{k}' has {len(data[k])} dims, expected 7")

    # Determine output path
    default_name = "bottle_grasp_waypoints_success.json" if args.mode == "grasp" else "bottle_pick_place_waypoints_success.json"
    dst_json, version = next_versioned_path(
        args.output if args.output
        else PROJECT_ROOT / "configs" / default_name
    )

    now_iso = datetime.now().isoformat()

    # Build output JSON
    out = {
        "source": data.get("source", "unknown"),
        "source_file": src.name,
        "mode": args.mode,
        "saved_at": now_iso,
        "version": dst_json.stem,
        "success_count": args.success_count,
        "attempt_count": args.attempt_count,
        "velocity_pct": args.velocity_pct,
        "open_gripper": data.get("open_gripper", 0.10),
        "close_gripper": data.get("close_gripper", 0.0),
        "notes": args.notes or data.get("notes", ""),
    }
    for k in pose_keys:
        out[k] = data[k]
    if "source_frames" in data:
        out["source_frames"] = data["source_frames"]

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  Saved → {dst_json}")

    # Write markdown log
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"success_baseline_{args.mode}_v{version}.md"
    lines = [
        f"# Success Baseline v{version} ({args.mode})",
        "",
        f"- **Created**: {now_iso}",
        f"- **Source file**: {src.name}",
        f"- **Success**: {args.success_count}/{args.attempt_count}",
        f"- **Velocity pct**: {args.velocity_pct}",
        f"- **open_gripper**: {out['open_gripper']:.4f}",
        f"- **close_gripper**: {out['close_gripper']:.4f}",
        f"- **Notes**: {out['notes']}",
        "",
        "## Poses",
        "",
    ]
    for k in pose_keys:
        pose = out[k]
        lines.append(f"- **{k}**: `[{', '.join(f'{float(x):.4f}' for x in pose)}]`")

    log_path.write_text("\n".join(lines) + "\n")
    print(f"  Log  → {log_path}")

    if args.mode == "pick_place":
        print(f"\n  Next step:")
        print(f"  python3 scripts/quick_bottle_grasp.py \\")
        print(f"      --waypoints {dst_json} \\")
        print(f"      --mode pick_place --step-confirm --velocity-pct {args.velocity_pct} --log-result")
    else:
        print(f"\n  Next step:")
        print(f"  python3 scripts/quick_bottle_grasp.py \\")
        print(f"      --waypoints {dst_json} \\")
        print(f"      --step-confirm --velocity-pct {args.velocity_pct} --log-result")


if __name__ == "__main__":
    main()
