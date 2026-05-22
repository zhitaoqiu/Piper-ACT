#!/usr/bin/env python3
"""Compatibility entrypoint for the guarded adapter-v2 recorder."""

from __future__ import annotations

from record_adapter_v2 import main


if __name__ == "__main__":
    print("record_adapter_v2_mirror.py now forwards to guarded record_adapter_v2.py.")
    raise SystemExit(main())
