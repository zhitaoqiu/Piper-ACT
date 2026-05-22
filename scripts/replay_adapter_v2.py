#!/usr/bin/env python3
"""Register adapter-v2 Piper classes, then enter standard LeRobot replay."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import adapter_v2  # noqa: F401 - registration side effects
from lerobot.scripts.lerobot_replay import main


if __name__ == "__main__":
    main()
