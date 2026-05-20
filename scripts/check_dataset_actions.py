#!/usr/bin/env python3
"""Check action representation in a LeRobot dataset.

Tells you whether actions are absolute next-state or delta (state-relative).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def read_parquet_tree(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {root}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def stack_vec(series, key):
    return np.stack([np.asarray(v, dtype=np.float32) for v in series.to_list()])


def feature_names(info: dict, key: str, dim: int) -> list[str]:
    names = info.get("features", {}).get(key, {}).get("names")
    if names and len(names) == dim:
        return list(names)
    if key == "observation.state":
        base = [f"J{i+1}" for i in range(min(dim, 6))]
        if dim >= 7:
            base.append("Grip")
        if dim >= 8:
            base.append("phase")
        return base + [f"dim{i}" for i in range(len(base), dim)]
    if key == "action":
        base = [f"J{i+1}" for i in range(min(dim, 6))]
        if dim >= 7:
            base.append("Grip")
        return base + [f"dim{i}" for i in range(len(base), dim)]
    return [f"dim{i}" for i in range(dim)]


def gripper_index(names: list[str], dim: int) -> int:
    for idx, name in enumerate(names):
        if str(name).lower() in {"gripper", "grip", "dgripper", "dgrip"}:
            return idx
    return 6 if dim >= 7 else dim - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--repo-id", default="piper/bottle_grasp")
    args = p.parse_args()

    root = args.dataset_root
    info = load_info(root)
    df = read_parquet_tree(root / "data")

    states = stack_vec(df["observation.state"], "observation.state")
    actions = stack_vec(df["action"], "action")
    eps = df["episode_index"].values

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    n_frames = len(df)
    n_eps = len(np.unique(eps))

    print(f"Dataset: {root}")
    print(f"Frames: {n_frames}")
    print(f"Episodes: {n_eps}")
    print(f"observation.state shape: ({state_dim},)")
    print(f"action shape: ({action_dim},)")
    print()

    state_names = feature_names(info, "observation.state", state_dim)
    action_names = feature_names(info, "action", action_dim)
    state_grip_idx = gripper_index(state_names, state_dim)
    action_grip_idx = gripper_index(action_names, action_dim)

    # Per-dim stats
    print("=" * 80)
    print("1. observation.state per-dim stats")
    print("-" * 80)
    print(f"{'Dim':>6}  {'min':>12}  {'max':>12}  {'std':>12}  {'mean':>12}")
    for d in range(state_dim):
        name = state_names[d] if d < len(state_names) else f"dim{d}"
        print(f"{name:>6}  {states[:, d].min():12.6f}  {states[:, d].max():12.6f}  "
              f"{states[:, d].std():12.6f}  {states[:, d].mean():12.6f}")
    print()

    print("=" * 80)
    print("2. action per-dim stats")
    print("-" * 80)
    print(f"{'Dim':>6}  {'min':>12}  {'max':>12}  {'std':>12}  {'mean':>12}")
    for d in range(action_dim):
        name = action_names[d] if d < len(action_names) else f"dim{d}"
        print(f"{name:>6}  {actions[:, d].min():12.6f}  {actions[:, d].max():12.6f}  "
              f"{actions[:, d].std():12.6f}  {actions[:, d].mean():12.6f}")
    print()

    # Compute delta = action - observation.state
    if state_dim == action_dim:
        deltas = actions - states
        print("=" * 80)
        print("3. delta = action - observation.state per-dim stats")
        print("-" * 80)
        print(f"{'Dim':>6}  {'min':>12}  {'max':>12}  {'std':>12}  {'mean':>12}")
        for d in range(action_dim):
            name = action_names[d] if d < len(action_names) else f"dim{d}"
            print(f"{name:>6}  {deltas[:, d].min():12.6f}  {deltas[:, d].max():12.6f}  "
                  f"{deltas[:, d].std():12.6f}  {deltas[:, d].mean():12.6f}")
        print()

        # Diagnostic: is action absolute or delta?
        state_range = states.max() - states.min()
        action_range = actions.max() - actions.min()
        delta_range = deltas.max() - deltas.min()

        if action_range > delta_range * 10:
            print(f"[INFO] action range ({action_range:.3f}) >> delta range ({delta_range:.3f})")
            print(f"[INFO] → action appears to be ABSOLUTE next-state, NOT delta.")
            print(f"[INFO] → deployment should use --action-mode absolute (NOT delta).")
        elif delta_range > action_range * 0.5:
            print(f"[INFO] delta range ({delta_range:.3f}) comparable to action range ({action_range:.3f})")
            print(f"[INFO] → action appears to be DELTA (state-relative).")
            print(f"[INFO] → deployment should use --action-mode delta.")
        else:
            print(f"[INFO] action range={action_range:.3f}, delta range={delta_range:.3f}")
            print(f"[INFO] → ambiguous, inspect manually.")
        print()

    # Gripper check
    print("=" * 80)
    print("4. GRIPPER (dim 7) detailed check")
    print("-" * 80)
    state_grip = states[:, state_grip_idx]
    action_grip = actions[:, action_grip_idx]
    print(f"indices: observation.state[{state_grip_idx}], action[{action_grip_idx}]")
    print(f"observation.state gripper: min={state_grip.min():.6f}  max={state_grip.max():.6f}  "
          f"std={state_grip.std():.6f}  mean={state_grip.mean():.6f}")
    print(f"action gripper:          min={action_grip.min():.6f}  max={action_grip.max():.6f}  "
          f"std={action_grip.std():.6f}  mean={action_grip.mean():.6f}")

    if state_dim == action_dim:
        delta_grip = deltas[:, action_grip_idx]
        print(f"delta gripper:           min={delta_grip.min():.6f}  max={delta_grip.max():.6f}  "
              f"std={delta_grip.std():.6f}  mean={delta_grip.mean():.6f}")

    grip_std_threshold = 0.0001
    if state_grip.std() < grip_std_threshold and action_grip.std() < grip_std_threshold:
        print()
        print("[FAIL] Gripper dimension has almost NO variation in BOTH state and action.")
        print("[FAIL] The policy CANNOT learn closing/opening from this data.")
        print("[FAIL] Data must be recollected with gripper open/close during teleop.")
    elif state_grip.std() < grip_std_threshold:
        print()
        print("[WARN] observation.state gripper std ≈ 0 — gripper was not moving during collection.")
        print("[WARN] But action gripper has some variation — model may still learn to open/close.")
    elif state_dim == action_dim and delta_grip.std() < grip_std_threshold:
        print()
        print("[WARN] delta gripper std ≈ 0 — the model CANNOT learn gripper open/close.")
        print("[WARN] The gripper value changes from frame to frame are near zero.")
    else:
        print()
        print("[OK] Gripper has usable variation.")

    # Per-episode gripper check
    print()
    print("=" * 80)
    print("5. Per-episode gripper check")
    print("-" * 80)
    unique_eps = sorted(np.unique(eps))
    any_grip_changed = False
    all_grip_changed = True
    for ep in unique_eps:
        mask = eps == ep
        ep_states = states[mask]
        ep_actions = actions[mask]
        grip_s_std = ep_states[:, state_grip_idx].std()
        grip_a_std = ep_actions[:, action_grip_idx].std()
        grip_s_range = ep_states[:, state_grip_idx].max() - ep_states[:, state_grip_idx].min()
        grip_a_range = ep_actions[:, action_grip_idx].max() - ep_actions[:, action_grip_idx].min()
        flag = ""
        if grip_a_range < 0.001:
            flag = "  <<< NO GRIPPER CHANGE"
            all_grip_changed = False
        else:
            any_grip_changed = True
        print(f"  ep {ep:03d}: frames={mask.sum():4d}  "
              f"state_grip range={grip_s_range:.6f} std={grip_s_std:.6f}  "
              f"action_grip range={grip_a_range:.6f} std={grip_a_std:.6f}{flag}")

    print()
    if not any_grip_changed:
        print("[FAIL] NO episode has gripper variation. The model CANNOT learn gripping.")
        print("[FAIL] Recollect data with gripper open/close during each episode.")
    elif not all_grip_changed:
        n_no_grip = sum(1 for ep in unique_eps
                        if (
                            actions[eps == ep][:, action_grip_idx].max()
                            - actions[eps == ep][:, action_grip_idx].min()
                        ) < 0.001)
        print(f"[WARN] {n_no_grip}/{len(unique_eps)} episodes have NO gripper change.")
        print(f"[WARN] Model training will be severely limited on gripper control.")
    else:
        print("[OK] All episodes have gripper variation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
