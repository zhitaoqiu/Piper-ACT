#!/usr/bin/env python3
"""Deep diagnosis: where does the prediction bias come from?"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def diagnose(checkpt: str, dataset_root: str, episode: int = 0):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LeRobotDataset("piper/bottle_grasp", root=dataset_root)
    eps = np.array(ds.hf_dataset["episode_index"])
    indices = np.where(eps == episode)[0]

    policy = ACTPolicy.from_pretrained(checkpt).to(dev).eval()
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpt,
        preprocessor_overrides={"device_processor": {"device": dev.type}, "normalizer_processor": {"device": dev.type}},
        postprocessor_overrides={"unnormalizer_processor": {"device": dev.type}, "device_processor": {"device": "cpu"}},
    )

    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]
    gt_deltas_all = []
    pred_deltas_all = []
    phases_all = []
    gt_states_all = []

    for frame_i, ds_idx in enumerate(indices):
        item = ds[int(ds_idx)]
        state = item["observation.state"].numpy()
        gt_delta = item["action"].numpy()
        with torch.inference_mode():
            batch = {}
            for key in item:
                if key == "observation.state" or key.startswith("observation.images."):
                    batch[key] = item[key].unsqueeze(0).to(dev)
            norm = pre(batch)
            chunk = policy.predict_action_chunk(norm)
            chunk = post(chunk).detach().cpu().numpy()[0]
        pred_deltas_all.append(chunk[0])
        gt_deltas_all.append(gt_delta)
        has_phase = len(state) > 7
        phases_all.append(state[7] if has_phase else float(frame_i) / len(indices))
        gt_states_all.append(state[:7])

    gt_deltas = np.array(gt_deltas_all)
    pred_deltas = np.array(pred_deltas_all)
    phases = np.array(phases_all)
    gt_states = np.array(gt_states_all)

    # =============================================
    # 1. BIAS ANALYSIS: is the model consistently over/under-predicting?
    # =============================================
    errors = pred_deltas - gt_deltas  # (N, 7)
    bias = np.mean(errors, axis=0)  # mean error per joint
    bias_abs = np.mean(np.abs(errors), axis=0)

    print("=" * 60)
    print("1. SYSTEMATIC BIAS CHECK")
    print("=" * 60)
    print(f"{'Joint':>6}  {'Mean Error':>12}  {'Abs Mean Err':>12}  {'GT Mean':>12}  {'Pred Mean':>12}")
    for j in range(7):
        print(f"{joint_names[j]:>6}  {bias[j]:12.6f}  {bias_abs[j]:12.6f}  "
              f"{np.mean(gt_deltas[:, j]):12.6f}  {np.mean(pred_deltas[:, j]):12.6f}")

    # Are errors biased (non-zero mean)?
    print(f"\nMean error sign per joint: {np.sign(bias)}")
    print(f"Is error biased? {'YES - all same sign' if np.all(np.abs(bias) > bias_abs*0.3) else 'mixed'}")

    # =============================================
    # 2. PHASE-DEPENDENT ERROR: does error vary with phase?
    # =============================================
    print(f"\n{'='*60}")
    print("2. ERROR vs PHASE (does model understand temporal position?)")
    print("=" * 60)
    phase_bins = [0, 0.25, 0.5, 0.75, 1.0]
    for j in range(7):
        print(f"\n  {joint_names[j]}:")
        print(f"  {'Phase range':<16} {'Mean Error':>12} {'GT std':>12} {'Pred std':>12} {'N':>6}")
        for i in range(len(phase_bins)-1):
            mask = (phases >= phase_bins[i]) & (phases < phase_bins[i+1])
            if mask.sum() == 0:
                continue
            print(f"  [{phase_bins[i]:.2f}, {phase_bins[i+1]:.2f})     "
                  f"{np.mean(errors[mask, j]):12.6f}  {np.std(gt_deltas[mask, j]):12.6f}  "
                  f"{np.std(pred_deltas[mask, j]):12.6f}  {mask.sum():>6}")

    # =============================================
    # 3. CORRELATION: pred vs gt per joint
    # =============================================
    print(f"\n{'='*60}")
    print("3. PREDICTION QUALITY (correlation + R^2)")
    print("=" * 60)
    for j in range(7):
        corr = np.corrcoef(gt_deltas[:, j], pred_deltas[:, j])[0, 1]
        ss_res = np.sum((gt_deltas[:, j] - pred_deltas[:, j])**2)
        ss_tot = np.sum((gt_deltas[:, j] - np.mean(gt_deltas[:, j]))**2)
        r2 = 1 - ss_res / ss_tot
        print(f"  {joint_names[j]:>6}: corr={corr:+.4f}, R²={r2:+.4f}")

    # =============================================
    # 4. BASELINE: naive zero-delta predictor
    # =============================================
    print(f"\n{'='*60}")
    print("4. BASELINE COMPARISON")
    print("=" * 60)
    zero_mse = np.mean(gt_deltas ** 2)
    model_mse = np.mean((pred_deltas - gt_deltas) ** 2)
    constant_mse = np.mean((gt_deltas - np.mean(gt_deltas, axis=0)) ** 2)
    print(f"  Zero-delta baseline MSE:  {zero_mse:.6f}")
    print(f"  Constant-mean baseline MSE: {constant_mse:.6f}")
    print(f"  Model MSE:                {model_mse:.6f}")
    print(f"  Model vs zero: {zero_mse/model_mse:.1f}x better")
    print(f"  Model vs constant: {constant_mse/model_mse:.1f}x better")

    # =============================================
    # 5. PER-JOINT INTEGRATION DRIFT
    # =============================================
    print(f"\n{'='*60}")
    print("5. CUMULATIVE DRIFT (integrated position over time)")
    print("=" * 60)
    integrated = np.cumsum(pred_deltas, axis=0)
    gt_integrated = np.cumsum(gt_deltas, axis=0)
    # Add first state
    integrated = integrated + gt_states[0]
    gt_integrated = gt_integrated + gt_states[0]

    for j in range(7):
        drift_rate = (integrated[-1, j] - gt_integrated[-1, j]) / len(indices)
        print(f"  {joint_names[j]:>6}: final drift={integrated[-1,j]-gt_integrated[-1,j]:+.4f}, "
              f"drift/frame={drift_rate:+.6f}, "
              f"GT range=[{np.min(gt_integrated[:,j]):.3f},{np.max(gt_integrated[:,j]):.3f}]")

    # =============================================
    # 6. SUMMARY
    # =============================================
    print(f"\n{'='*60}")
    print("6. SUMMARY")
    print("=" * 60)

    # Is the model better than a dumb baseline?
    better_than_zero = model_mse < zero_mse
    better_than_constant = model_mse < constant_mse

    # Does error accumulate or oscillate?
    drift_signs = np.sign([integrated[-1, j] - gt_integrated[-1, j] for j in range(7)])
    all_same_direction = len(set(drift_signs)) <= 2  # most joints drift in same way

    print(f"  Model beats zero-delta baseline: {better_than_zero}")
    print(f"  Model beats constant-mean baseline: {better_than_constant}")
    print(f"  Joint drift direction consistent: {all_same_direction}")

    if not better_than_constant:
        print(f"\n  *** CRITICAL: Model is WORSE than predicting the mean delta! ***")
        print(f"  *** This means the model has learned NOTHING useful. ***")
    elif all_same_direction and np.mean(np.abs(bias)) > 0.01:
        print(f"\n  *** Model has systematic bias: all joints drift in same direction ***")
        print(f"  *** Likely cause: model is attending to wrong features or phase encoding broken ***")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", required=True)
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_delta_phase")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    diagnose(args.checkpt, args.dataset_root, args.episode)


if __name__ == "__main__":
    main()
