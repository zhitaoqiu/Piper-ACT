#!/usr/bin/env python3
"""Evaluate predicted trajectory vs ground truth on a full episode."""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def fmt_vec(v):
    return "[" + ", ".join(f"{float(x):.4f}" for x in v) + "]"


def evaluate(checkpt: str, dataset_root: str, episode: int, device: str = "cuda"):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    ds = LeRobotDataset("piper/bottle_grasp", root=dataset_root)
    eps = np.array(ds.hf_dataset["episode_index"])
    indices = np.where(eps == episode)[0]

    policy = ACTPolicy.from_pretrained(checkpt).to(dev).eval()

    from lerobot.policies.factory import make_pre_post_processors
    preprocessor_overrides = {
        "device_processor": {"device": dev.type},
        "normalizer_processor": {"device": dev.type},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {"device": dev.type},
        "device_processor": {"device": "cpu"},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpt,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    chunk_size = policy.config.chunk_size
    print(f"Model: chunk_size={chunk_size}, episode={episode}, frames={len(indices)}")

    gt_states = []
    pred_deltas = []
    gt_deltas = []
    integrated_states = []
    current_integrated = None

    for i, ds_idx in enumerate(indices):
        item = ds[int(ds_idx)]
        state = item["observation.state"].numpy()  # 8D
        gt_delta = item["action"].numpy()  # 7D

        if current_integrated is None:
            current_integrated = state[:7].copy()

        gt_states.append(state[:7].copy())

        with torch.inference_mode():
            batch = {}
            for key in item:
                if key == "observation.state" or key.startswith("observation.images."):
                    batch[key] = item[key].unsqueeze(0).to(dev)
            norm = preprocessor(batch)
            chunk = policy.predict_action_chunk(norm)
            chunk = postprocessor(chunk).detach().cpu().numpy()[0]

        pred_delta = chunk[0]  # first action
        pred_deltas.append(pred_delta)
        gt_deltas.append(gt_delta)

        # Integrate prediction
        current_integrated = current_integrated + pred_delta
        integrated_states.append(current_integrated.copy())

    pred_deltas = np.array(pred_deltas)
    gt_deltas = np.array(gt_deltas)
    gt_states = np.array(gt_states)
    integrated_states = np.array(integrated_states)

    # --- Report ---
    # 1. Delta prediction error
    delta_mse_joint = np.mean((pred_deltas - gt_deltas) ** 2, axis=0)
    delta_mae_joint = np.mean(np.abs(pred_deltas - gt_deltas), axis=0)

    print(f"\n=== Delta Prediction Error (per joint) ===")
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]
    print(f"{'Joint':>6}  {'MSE':>10}  {'MAE':>10}  {'GT std':>10}  {'Pred std':>10}")
    for j in range(7):
        print(f"{joint_names[j]:>6}  {delta_mse_joint[j]:10.6f}  {delta_mae_joint[j]:10.6f}  "
              f"{np.std(gt_deltas[:, j]):10.6f}  {np.std(pred_deltas[:, j]):10.6f}")

    # 2. Integrated trajectory error
    integrated_mse = np.mean((integrated_states - gt_states) ** 2)
    final_pos_error = np.linalg.norm(integrated_states[-1] - gt_states[-1])
    final_arm_error = np.linalg.norm(integrated_states[-1][:6] - gt_states[-1][:6])

    print(f"\n=== Integrated Trajectory Error ===")
    print(f"Integrated MSE: {integrated_mse:.6f}")
    print(f"Final position L2 error: {final_pos_error:.4f}")
    print(f"Final arm L2 error: {final_arm_error:.4f}")

    # 3. Per-step integration error
    step_errors = np.linalg.norm(integrated_states - gt_states, axis=1)
    print(f"\n=== Per-step Position Error (L2) ===")
    print(f"Mean:  {np.mean(step_errors):.4f}")
    print(f"Max:   {np.max(step_errors):.4f}")
    print(f"Final: {step_errors[-1]:.4f}")

    # 4. Trajectory endpoints comparison
    print(f"\n=== Trajectory Endpoints ===")
    print(f"GT start:      {fmt_vec(gt_states[0])}")
    print(f"GT end:        {fmt_vec(gt_states[-1])}")
    print(f"Integrated end: {fmt_vec(integrated_states[-1])}")
    print(f"GT delta range:  {fmt_vec(np.max(np.abs(gt_deltas), axis=0))}")
    print(f"Pred delta range:{fmt_vec(np.max(np.abs(pred_deltas), axis=0))}")

    # 5. Divergence diagnosis
    # How fast does the integrated trajectory diverge?
    print(f"\n=== Divergence over Time ===")
    for pct in [10, 25, 50, 75, 100]:
        n = max(1, int(len(indices) * pct / 100))
        err = np.linalg.norm(integrated_states[n-1] - gt_states[n-1])
        print(f"  {pct:3d}% ({n:3d} frames): position error = {err:.4f}")

    # 6. Chunk consistency check
    # Does the model produce a coherent 10-step chunk?
    print(f"\n=== First Chunk Internal Consistency ===")
    item0 = ds[int(indices[0])]
    with torch.inference_mode():
        batch = {}
        for key in item0:
            if key == "observation.state" or key.startswith("observation.images."):
                batch[key] = item0[key].unsqueeze(0).to(dev)
        norm = preprocessor(batch)
        chunk0 = policy.predict_action_chunk(norm)
        chunk0 = postprocessor(chunk0).detach().cpu().numpy()[0]
    chunk_range = np.ptp(chunk0[:, :6], axis=0)
    chunk_step_range = np.max(np.abs(np.diff(chunk0[:, :6], axis=0)), axis=1)
    print(f"Chunk joint ranges: {fmt_vec(chunk_range)}")
    print(f"Chunk step-to-step max diff: {np.max(chunk_step_range):.4f}")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", required=True)
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_delta_phase")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.checkpt, args.dataset_root, args.episode, args.device)


if __name__ == "__main__":
    main()
