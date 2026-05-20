#!/usr/bin/env python3
"""Debug dataset alignment: check whether action_t ≈ qpos_t or action_t ≈ qpos_{t+1}.

Focus on J2 because the model currently gets stuck at J2≈0.49.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

J2_IDX = 1
NEAR_ZERO_THRESH = 0.003
J2_INTERVALS = [
    (0.0, 0.2, "0.0-0.2"),
    (0.2, 0.5, "0.2-0.5"),
    (0.45, 0.55, "0.45-0.55"),
    (0.8, 1.2, "0.8-1.2"),
    (1.2, 1.6, "1.2-1.6"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path,
                        default=PROJECT_ROOT / "data/lerobot_dataset_today_approach_1ep")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--print-frames-around", type=float, default=None,
                        help="Print all frames near a specific J2 value, e.g. 0.49")
    parser.add_argument("--max-print", type=int, default=200,
                        help="Max frames to print in detail mode")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id="piper/bottle_approach_today_1ep",
        root=args.dataset_root,
        episodes=[args.episode],
    )
    n = len(dataset)
    print(f"Dataset: {n} frames, {dataset.num_episodes} episode(s)")
    print()

    # Collect all states and actions
    qpos_all = np.zeros((n, 7), dtype=np.float32)
    action_all = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        frame = dataset[i]
        qpos_all[i] = frame["observation.state"].numpy()
        action_all[i] = frame["action"].numpy()

    # ── Per-frame alignment check ──
    print(f"{'t':>4s}  {'qpos_J2':>10s}  {'action_J2':>10s}  {'qpos_next_J2':>10s}  "
          f"{'act-qpos':>10s}  {'act-qnext':>10s}  {'closer_to':>10s}")
    print("-" * 80)

    closer_to_next_count = 0
    closer_to_cur_count = 0

    printed = 0
    for t in range(n - 1):
        qj2 = float(qpos_all[t, J2_IDX])
        aj2 = float(action_all[t, J2_IDX])
        qn_j2 = float(qpos_all[t + 1, J2_IDX])

        d_cur = abs(aj2 - qj2)
        d_next = abs(aj2 - qn_j2)
        if d_cur < d_next:
            closer = "qpos_t"
            closer_to_cur_count += 1
        elif d_next < d_cur:
            closer = "qpos_next"
            closer_to_next_count += 1
        else:
            closer = "equal"

        # Print frames near target J2 value or first/last N frames
        do_print = False
        if args.print_frames_around is not None:
            if abs(qj2 - args.print_frames_around) < 0.03:
                do_print = True
        elif printed < 30 or t >= n - 15 or (0.44 <= qj2 <= 0.56):
            do_print = True

        if do_print and printed < args.max_print:
            print(f"{t:4d}  {qj2:10.5f}  {aj2:10.5f}  {qn_j2:10.5f}  "
                  f"{aj2-qj2:10.5f}  {aj2-qn_j2:10.5f}  {closer:>10s}")
            printed += 1

    print(f"\nCloser to qpos_t:      {closer_to_cur_count}")
    print(f"Closer to qpos_next:   {closer_to_next_count}")
    print(f"Alignment verdict: ", end="")
    if closer_to_next_count > closer_to_cur_count:
        print("OK — action[t] ≈ qpos[t+1], mirror mode correct.")
    else:
        print("SUSPICIOUS — action[t] may be qpos[t], model may learn identity.")

    # ── Per-interval true_delta_j2 stats ──
    print("\n" + "=" * 80)
    print("True delta J2 stats by interval")
    print("=" * 80)
    print(f"{'Interval':>12s}  {'count':>6s}  {'mean_delta':>10s}  "
          f"{'min_delta':>10s}  {'max_delta':>10s}  {'near_zero':>8s}")
    print("-" * 70)

    for lo, hi, label in J2_INTERVALS:
        mask = (qpos_all[:n-1, J2_IDX] >= lo) & (qpos_all[:n-1, J2_IDX] < hi)
        if not mask.any():
            print(f"{label:>12s}  {'(empty)':>6s}")
            continue
        true_delta = action_all[:n-1, J2_IDX][mask] - qpos_all[:n-1, J2_IDX][mask]
        near_zero = (np.abs(true_delta) < NEAR_ZERO_THRESH).sum()
        print(f"{label:>12s}  {mask.sum():6d}  {true_delta.mean():10.5f}  "
              f"{true_delta.min():10.5f}  {true_delta.max():10.5f}  {near_zero:8d}")

    # ── Detailed look at 0.45-0.55 zone ──
    print("\n" + "=" * 80)
    print("CRITICAL: J2 0.45-0.55 zone — all frames")
    print("=" * 80)
    crit_mask = (qpos_all[:n-1, J2_IDX] >= 0.45) & (qpos_all[:n-1, J2_IDX] <= 0.55)
    crit_idxs = np.where(crit_mask)[0]
    if len(crit_idxs) == 0:
        print("  No frames in this range.")
    else:
        print(f"  {len(crit_idxs)} frames in J2 0.45-0.55")
        print(f"  {'t':>4s}  {'qpos_J2':>10s}  {'action_J2':>10s}  {'qpos_next':>10s}  "
              f"{'delta':>10s}  {'action-qnext':>10s}")
        for t in crit_idxs:
            qj2 = float(qpos_all[t, J2_IDX])
            aj2 = float(action_all[t, J2_IDX])
            qn_j2 = float(qpos_all[t + 1, J2_IDX])
            print(f"  {t:4d}  {qj2:10.5f}  {aj2:10.5f}  {qn_j2:10.5f}  "
                  f"{aj2-qj2:10.5f}  {aj2-qn_j2:10.5f}")

        true_delta_crit = action_all[crit_idxs, J2_IDX] - qpos_all[crit_idxs, J2_IDX]
        near_zero_crit = (np.abs(true_delta_crit) < NEAR_ZERO_THRESH).sum()
        print(f"\n  Mean delta: {true_delta_crit.mean():.5f}")
        print(f"  Min delta:  {true_delta_crit.min():.5f}")
        print(f"  Max delta:  {true_delta_crit.max():.5f}")
        print(f"  Near-zero:  {near_zero_crit}/{len(crit_idxs)}")
        if true_delta_crit.mean() < 0.001:
            print("  VERDICT: Data has a PAUSE in this zone — the model learned to stop here.")
        else:
            print("  VERDICT: Data has positive delta in this zone — model should keep moving.")


if __name__ == "__main__":
    main()
