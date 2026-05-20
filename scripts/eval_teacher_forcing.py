#!/usr/bin/env python3
"""Offline teacher-forcing evaluation with proper pre/post processing."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", required=True)
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_approach_20ep")
    parser.add_argument("--episodes", default="1,2,3,4,7,10,14,15,20,22,23,24,25,26")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {"device": device.type},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {"device": device.type},
        "device_processor": {"device": "cpu"},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpt,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    print(f"Policy loaded: {sum(p.numel() for p in policy.parameters()):,} params")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    episode_list = [int(x.strip()) for x in args.episodes.split(",")]
    ds = LeRobotDataset("piper/bottle_approach_20ep", root=args.dataset_root, episodes=episode_list)
    print(f"Dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")

    per_ep_preds = {}
    per_ep_acts = {}
    all_preds = []
    all_acts = []

    n_total = min(args.max_frames, ds.num_frames) if args.max_frames > 0 else ds.num_frames
    print(f"Evaluating up to {n_total} frames...")

    count = 0
    for ep in ds.episodes:
        ep_preds = []
        ep_acts = []
        for i in range(ds.num_frames):
            if count >= n_total:
                break
            item = ds[i]
            ep_idx = int(item["episode_index"])
            if ep_idx != ep:
                continue
            obs = {}
            for key in policy.config.input_features:
                v = item[key]
                if hasattr(v, "numpy"): v = v.numpy()
                obs[key] = np.asarray(v, dtype=np.float32)

            # Apply preprocessor
            obs_batch = preprocessor({k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()})

            with torch.no_grad():
                pred_chunk = policy.predict_action_chunk(obs_batch)
            pred_raw = pred_chunk[:, 0, :] if pred_chunk.dim() == 3 else pred_chunk

            # Apply postprocessor (unnormalize) — takes tensor directly
            pred_batch = postprocessor(pred_raw)
            pred_np = pred_batch.squeeze(0).cpu().numpy()

            act = item["action"]
            if hasattr(act, "numpy"): act = act.numpy()
            act = np.asarray(act, dtype=np.float32)

            ep_preds.append(pred_np)
            ep_acts.append(act)
            count += 1
        if ep_preds:
            per_ep_preds[ep] = np.stack(ep_preds)
            per_ep_acts[ep] = np.stack(ep_acts)
            all_preds.append(per_ep_preds[ep])
            all_acts.append(per_ep_acts[ep])

    all_preds = np.concatenate(all_preds)
    all_acts = np.concatenate(all_acts)
    print(f"Frames evaluated: {len(all_preds)}")

    dim_names = ["J1","J2","J3","J4","J5","J6","grip"]
    print(f"\n{'Dim':>6}  {'pred_mean':>10}  {'pred_std':>10}  {'true_mean':>10}  {'true_std':>10}  {'status'}")
    for d, name in enumerate(dim_names):
        ps = all_preds[:, d]
        ts = all_acts[:, d]
        ok = "COLLAPSED" if np.std(ps) < 0.0001 else "OK"
        print(f"{name:>6}  {np.mean(ps):10.6f}  {np.std(ps):10.6f}  {np.mean(ts):10.6f}  {np.std(ts):10.6f}  {ok}")

    mse_per_dim = np.mean((all_preds - all_acts) ** 2, axis=0)
    mean_baseline = np.mean(all_acts, axis=0)
    mse_baseline = np.mean((all_acts - mean_baseline) ** 2, axis=0)

    print(f"\n{'Dim':>6}  {'model_mse':>12}  {'baseline_mse':>12}  {'improvement_ratio':>16}")
    collapsed = []
    for d, name in enumerate(dim_names):
        ratio = mse_per_dim[d] / mse_baseline[d] if mse_baseline[d] > 1e-10 else float('inf')
        flag = " *** COLLAPSED ***" if ratio > 0.9 else ""
        if ratio > 0.9:
            collapsed.append(name)
        print(f"{name:>6}  {mse_per_dim[d]:12.6f}  {mse_baseline[d]:12.6f}  {ratio:16.6f}{flag}")

    overall = np.mean(mse_per_dim[:6]) / np.mean(mse_baseline[:6])
    print(f"\nOverall improvement_ratio (arm only): {overall:.6f}")
    if overall > 0.9:
        print("*** COLLAPSED ***")
    elif overall > 0.5:
        print("WARNING: model weak")
    else:
        print("PASSED")

    print("\n--- Per-Ep J2 ---")
    for ep in sorted(per_ep_preds.keys()):
        p = per_ep_preds[ep][:, 1]
        t = per_ep_acts[ep][:, 1]
        print(f"Ep{ep:2d}: pred_J2=[{p[0]:.3f}→{p[-1]:.3f}] std={np.std(p):.4f}  "
              f"true_J2=[{t[0]:.3f}→{t[-1]:.3f}]")

if __name__ == "__main__":
    main()
