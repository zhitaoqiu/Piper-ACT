#!/usr/bin/env python3
"""Diagnostic tool: test every /dev/video* candidate for adapter v2 global camera.

Saves a sample frame from each working candidate to --save-dir so the operator
can visually confirm which device is the real global camera.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from camera.rs_camera import require_opencv, find_usb_video_devices, describe_video_devices

cv2 = require_opencv()


def _device_name(path: str) -> str:
    try:
        name_path = Path("/sys/class/video4linux") / Path(path).name / "name"
        return name_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"

BLACK_MEAN_THRESHOLD = 5.0
DEFAULT_WARMUP_FRAMES = 15


def _video_index(path: str) -> int | None:
    import re
    match = re.search(r"(\d+)$", path)
    return int(match.group(1)) if match else None


def _open_candidate(device_id):
    idx = _video_index(device_id) if isinstance(device_id, str) else device_id
    if idx is None:
        idx = device_id
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(idx)


def test_candidate(device_id: str, *, warmup: int, save_dir: Path | None) -> dict:
    result = {
        "path": device_id,
        "opened": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "mean": 0.0,
        "min": 0,
        "max": 0,
        "is_black": True,
        "sample_saved": None,
    }
    cap = _open_candidate(device_id)
    if not cap.isOpened():
        cap.release()
        return result

    result["opened"] = True
    result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result["fps"] = cap.get(cv2.CAP_PROP_FPS)

    frame = None
    for _ in range(max(1, warmup)):
        ret, f = cap.read()
        if ret and f is not None and f.size:
            frame = f
        time.sleep(0.03)

    if frame is None:
        cap.release()
        result["opened"] = False
        return result

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result["mean"] = float(gray.mean())
    result["min"] = int(gray.min())
    result["max"] = int(gray.max())
    result["is_black"] = result["mean"] < BLACK_MEAN_THRESHOLD

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{Path(device_id).name}_sample.jpg"
        fpath = save_dir / fname
        cv2.imwrite(str(fpath), frame)
        result["sample_saved"] = str(fpath)

    cap.release()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose adapter v2 global camera candidates."
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=find_usb_video_devices(),
        help="Device paths to test (default: all /dev/video*).",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP_FRAMES,
    )
    parser.add_argument(
        "--save-dir",
        default=str(PROJECT_ROOT / "logs" / "camera_debug"),
    )
    args = parser.parse_args()

    if not args.candidates:
        print("No video device candidates found.")
        return 1

    save_dir = Path(args.save_dir)

    print("=" * 64)
    print("Adapter v2 global camera diagnostic")
    print(f"  candidates: {args.candidates}")
    print(f"  warmup frames: {args.warmup}")
    print(f"  save dir: {save_dir}")
    print(f"  black threshold: mean < {BLACK_MEAN_THRESHOLD}")
    print("=" * 64)
    print()

    results = []
    working = []
    for device in args.candidates:
        r = test_candidate(device, warmup=args.warmup, save_dir=save_dir)
        results.append(r)
        print(f"{device} ({_device_name(device)}):")
        print(f"  opened  : {r['opened']}")
        if r["opened"]:
            print(f"  res     : {r['width']}x{r['height']} @ {r['fps']:.1f}fps")
            print(f"  mean    : {r['mean']:.2f}")
            print(f"  min/max : {r['min']} / {r['max']}")
            print(f"  is_black: {r['is_black']}")
            if r["sample_saved"]:
                print(f"  sample  : {r['sample_saved']}")
            if not r["is_black"]:
                working.append(device)
        print()

    # Prefer USB cameras over RealSense sub-devices for global camera
    def _is_realsense(dev_path: str) -> bool:
        name = _device_name(dev_path).lower()
        return "realsense" in name or "intel" in name

    usb_working = [d for d in working if not _is_realsense(d)]
    rs_working = [d for d in working if _is_realsense(d)]

    print("=" * 64)
    print("Summary:")
    opened = [r for r in results if r["opened"]]
    print(f"  opened       : {len(opened)} / {len(results)}")
    print(f"  not black    : {len(working)}")
    if usb_working:
        print(f"  USB camera   : {usb_working} (preferred for global)")
    if rs_working:
        print(f"  RealSense sub: {rs_working} (likely depth/IR, avoid for global)")
    if working:
        rec = usb_working[0] if usb_working else working[0]
        print()
        print("Recommended record command:")
        print(f"  --global-camera {rec}")
        if len(working) > 1:
            others = [d for d in working if d != rec]
            print(f"  (also try {' or '.join(others)} if the first is wrong)")
    else:
        print("  WARNING: no working global camera found!")
        print("  Check physical connection and device permissions.")
    print("=" * 64)

    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
