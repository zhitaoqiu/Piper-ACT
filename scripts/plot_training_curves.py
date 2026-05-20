#!/usr/bin/env python3
"""Plot training loss and gradient norm curves from CSV."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    csv_path = PROJECT_ROOT / "reports/delta_v2_training_loss.csv"
    if not csv_path.exists():
        print(f"Missing: {csv_path}", file=sys.stderr)
        return 1

    def parse_step(s):
        s = s.strip()
        if s.endswith("K"):
            return float(s[:-1]) * 1000
        return float(s)

    steps, losses, gradnorms = [], [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f, fieldnames=["step", "loss", "grdn"]):
            steps.append(parse_step(row["step"]))
            losses.append(float(row["loss"]))
            gradnorms.append(float(row["grdn"]))

    steps = np.array(steps) / 1000  # K steps
    losses = np.array(losses)
    gradnorms = np.array(gradnorms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss curve ---
    ax1.plot(steps, losses, color="#2563eb", linewidth=1.5, alpha=0.9)
    # Rolling smooth
    window = 10
    if len(losses) > window:
        smooth = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax1.plot(steps[window-1:], smooth, color="#ef4444", linewidth=2, label=f"MA{window} smooth")
    ax1.set_xlabel("Steps (K)", fontsize=12)
    ax1.set_ylabel("Training Loss (MSE)", fontsize=12)
    ax1.set_title("ACT Delta v2 — Training Loss", fontsize=14, weight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Annotate start & end
    ax1.annotate(f"{losses[0]:.3f}", (steps[0], losses[0]), textcoords="offset points",
                 xytext=(10, 10), fontsize=9, color="#2563eb")
    ax1.annotate(f"{losses[-1]:.3f}", (steps[-1], losses[-1]), textcoords="offset points",
                 xytext=(10, -10), fontsize=9, color="#ef4444")

    # --- Gradient norm ---
    ax2.plot(steps, gradnorms, color="#7c3aed", linewidth=1.0, alpha=0.7)
    if len(gradnorms) > window:
        smooth_g = np.convolve(gradnorms, np.ones(window)/window, mode="valid")
        ax2.plot(steps[window-1:], smooth_g, color="#f97316", linewidth=2, label=f"MA{window} smooth")
    ax2.set_xlabel("Steps (K)", fontsize=12)
    ax2.set_ylabel("Gradient Norm", fontsize=12)
    ax2.set_title("ACT Delta v2 — Gradient Norm", fontsize=14, weight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Config: delta_horizon=1, no phase, 7D state, 50 episodes", fontsize=11,
                 color="gray", y=1.01)

    out_path = PROJECT_ROOT / "docs" / "fig_training_loss.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
