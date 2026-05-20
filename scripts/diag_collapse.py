#!/usr/bin/env python3
"""Quick collapse diagnosis: per-traj std, input sensitivity, normalized state range."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run(checkpt: str, dataset_root: str, episodes: str, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpt}")
    print(f"Dataset: {dataset_root}  episodes=[{episodes}]")
    print("=" * 70)

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(checkpt)
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
        pretrained_path=checkpt,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
    print(f"Params: {sum(p.numel() for p in policy.parameters()):,}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    episode_list = [int(x.strip()) for x in episodes.split(",")]
    ds = LeRobotDataset("piper/bottle_approach_20ep", root=dataset_root,
                        episodes=episode_list)
    print(f"Episodes: {ds.num_episodes}, frames: {ds.num_frames}")

    # ── 1. Per-episode prediction stats ──
    print("\n" + "=" * 70)
    print("1) PER-TRAJECTORY PREDICTION STATS")
    print("-" * 70)

    all_preds = []
    all_acts = []
    per_ep = {}

    for ep in episode_list:
        ep_preds = []
        ep_acts = []
        ep_nstates = []  # normalized state J2 values
        ep_raw_robot_j2 = []  # raw robot_state J2 before normalization
        for i in range(ds.num_frames):
            item = ds[i]
            if int(item["episode_index"]) != ep:
                continue
            # Raw obs
            obs = {}
            for key in policy.config.input_features:
                v = item[key]
                if hasattr(v, "numpy"): v = v.numpy()
                obs[key] = np.asarray(v, dtype=np.float32)

            # Preprocess
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
            obs_batch = preprocessor(obs_t)

            # Capture normalized state J2
            ns = obs_batch["observation.state"].squeeze(0).cpu().numpy()
            ep_nstates.append(ns[1])  # J2 in normalized space
            ep_raw_robot_j2.append(np.asarray(obs["observation.state"], dtype=np.float32)[1])

            with torch.no_grad():
                pred_chunk = policy.predict_action_chunk(obs_batch)
            pred_raw = pred_chunk[:, 0, :] if pred_chunk.dim() == 3 else pred_chunk
            pred_batch = postprocessor(pred_raw)
            pred_np = pred_batch.squeeze(0).cpu().numpy()

            act = item["action"]
            if hasattr(act, "numpy"): act = act.numpy()
            act = np.asarray(act, dtype=np.float32)

            ep_preds.append(pred_np)
            ep_acts.append(act)

        if ep_preds:
            pa = np.stack(ep_preds)
            aa = np.stack(ep_acts)
            per_ep[ep] = {"pred": pa, "act": aa, "nstate_j2": np.array(ep_nstates),
                          "raw_robot_j2": np.array(ep_raw_robot_j2)}
            all_preds.append(pa)
            all_acts.append(aa)

    all_preds = np.concatenate(all_preds)
    all_acts = np.concatenate(all_acts)

    # Per-ep J2 detail
    print(f"{'Ep':>5}  {'n_frames':>8}  {'pred_J2_std':>12}  {'pred_J2_min':>10}  {'pred_J2_max':>10}  "
          f"{'raw_J2_range':>12}  {'norm_J2_range':>13}  {'status'}")
    for ep in sorted(per_ep.keys()):
        p = per_ep[ep]["pred"]
        nj2 = per_ep[ep]["nstate_j2"]
        rj2 = per_ep[ep]["raw_robot_j2"]
        std_j2 = float(np.std(p[:, 1]))
        flag = "COLLAPSED" if std_j2 < 0.01 else "OK"
        print(f"{ep:5d}  {len(p):8d}  {std_j2:12.6f}  {float(np.min(p[:,1])):10.4f}  "
              f"{float(np.max(p[:,1])):10.4f}  [{float(np.min(rj2)):.4f}, {float(np.max(rj2)):.4f}]  "
              f"[{float(np.min(nj2)):.4f}, {float(np.max(nj2)):.4f}]  {flag}")

    # Global stats
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
    print(f"\n{'Dim':>6}  {'model_mse':>12}  {'baseline_mse':>12}  {'ratio':>10}  {'status'}")
    for d, name in enumerate(dim_names):
        ratio = mse_per_dim[d] / mse_baseline[d] if mse_baseline[d] > 1e-10 else float("inf")
        flag = "*** COLLAPSED ***" if ratio > 0.9 else ""
        print(f"{name:>6}  {mse_per_dim[d]:12.6f}  {mse_baseline[d]:12.6f}  {ratio:10.6f}  {flag}")
    overall = np.mean(mse_per_dim[:6]) / np.mean(mse_baseline[:6])
    print(f"\nArm-only improvement_ratio: {overall:.6f}  {'PASSED' if overall < 0.5 else 'COLLAPSED'}")

    # ── 2. INPUT SENSITIVITY TEST ──
    print("\n" + "=" * 70)
    print("2) INPUT SENSITIVITY — perturb J2 and observe Δraw_action[J2]")
    print("-" * 70)

    # Use first frame of first episode as anchor
    item0 = ds[0]
    obs0 = {}
    for key in policy.config.input_features:
        v = item0[key]
        if hasattr(v, "numpy"): v = v.numpy()
        obs0[key] = np.asarray(v, dtype=np.float32)

    def predict_single(obs_dict):
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs_dict.items()}
        obs_n = preprocessor(obs_t)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(obs_n)
        raw = chunk[:, 0, :] if chunk.dim() == 3 else chunk
        pred = postprocessor(raw).squeeze(0).cpu().numpy()
        return pred

    base_pred = predict_single(obs0)
    base_state = np.asarray(obs0["observation.state"], dtype=np.float32)

    print(f"Base observation.state (raw): {base_state}")
    print(f"Base prediction:              {np.round(base_pred, 6)}")
    print(f"Base pred J2: {base_pred[1]:.6f}")
    print()

    # Sensitivity tests: A, B, C
    tests = {
        "A: J2 += 0.01 rad (raw)": 0.01,
        "B: J2 += 0.10 rad (raw)": 0.10,
        "C: J2 += 0.50 rad (raw)": 0.50,
    }
    for label, delta in tests.items():
        obs_mod = {k: v.copy() for k, v in obs0.items()}
        obs_mod["observation.state"][1] += delta
        pred_mod = predict_single(obs_mod)
        d_pred = pred_mod - base_pred
        raw_norm_j2_before = None  # We'll get from preprocessor
        raw_norm_j2_after = None

        # Compute norm state J2 change
        obs_t_before = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs0.items()}
        obs_t_after = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs_mod.items()}
        ns_before = preprocessor(obs_t_before)["observation.state"][0, 1].item()
        ns_after = preprocessor(obs_t_after)["observation.state"][0, 1].item()
        d_norm_j2 = ns_after - ns_before

        print(f"  {label}:")
        print(f"    raw J2: {base_state[1]:.4f} → {base_state[1]+delta:.4f}  (Δ={delta:.4f})")
        print(f"    norm J2: {ns_before:.6f} → {ns_after:.6f}  (Δ={d_norm_j2:.6f})")
        print(f"    Δpred_J2: {d_pred[1]:.6f}  |Δpred|={np.linalg.norm(d_pred):.6f}")
        print(f"    pred_mod: {np.round(pred_mod, 6)}")
        sensitivity = float(np.abs(d_pred[1]) / (delta + 1e-12))
        print(f"    sensitivity: |Δpred_J2|/|Δraw_J2| = {sensitivity:.4f}")
        print()

    # D: different image (different episode)
    print("  D: Different episode (different image) test:")
    # Find first frame of ep 2
    for i in range(ds.num_frames):
        if int(ds[i]["episode_index"]) == episode_list[1 % len(episode_list)]:
            item_ep2 = ds[i]
            break
    else:
        item_ep2 = ds[min(40, ds.num_frames - 1)]

    obs_ep2 = {}
    for key in policy.config.input_features:
        v = item_ep2[key]
        if hasattr(v, "numpy"): v = v.numpy()
        obs_ep2[key] = np.asarray(v, dtype=np.float32)

    pred_ep2 = predict_single(obs_ep2)
    d_pred_img = pred_ep2 - base_pred

    # Same state as ep2 but with ep0 image
    # (just for reference, show state difference too)
    state_ep2 = np.asarray(obs_ep2["observation.state"], dtype=np.float32)
    state_diff = np.linalg.norm(state_ep2 - base_state)

    print(f"    ep0 state:      {base_state}")
    print(f"    ep{episode_list[1 % len(episode_list)]} state:      {state_ep2}")
    print(f"    ||state_diff||: {state_diff:.4f}")
    print(f"    ep0 pred:       {np.round(base_pred, 6)}")
    print(f"    epX pred:       {np.round(pred_ep2, 6)}")
    print(f"    Δpred_J2:       {d_pred_img[1]:.6f}")
    print(f"    |Δpred|:        {np.linalg.norm(d_pred_img):.6f}")
    print()

    # ── 3. NORMALIZED STATE RANGE CHECK ──
    print("=" * 70)
    print("3) NORMALIZED STATE RANGE (across all frames)")
    print("-" * 70)
    all_norm_j2 = []
    for i in range(ds.num_frames):
        item = ds[i]
        obs = {}
        for key in policy.config.input_features:
            v = item[key]
            if hasattr(v, "numpy"): v = v.numpy()
            obs[key] = np.asarray(v, dtype=np.float32)
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        obs_n = preprocessor(obs_t)
        nj2 = obs_n["observation.state"][0, 1].item()
        all_norm_j2.append(nj2)

    all_norm_j2 = np.array(all_norm_j2)
    print(f"normalized_state[J2]: mean={np.mean(all_norm_j2):.6f}  std={np.std(all_norm_j2):.6f}  "
          f"min={np.min(all_norm_j2):.6f}  max={np.max(all_norm_j2):.6f}  "
          f"range={np.max(all_norm_j2)-np.min(all_norm_j2):.6f}")

    # Compare with raw J2
    all_raw_j2 = []
    for i in range(ds.num_frames):
        item = ds[i]
        s = item["observation.state"]
        if hasattr(s, "numpy"): s = s.numpy()
        all_raw_j2.append(float(np.asarray(s, dtype=np.float32)[1]))
    all_raw_j2 = np.array(all_raw_j2)
    print(f"raw robot_state[J2]:   mean={np.mean(all_raw_j2):.6f}  std={np.std(all_raw_j2):.6f}  "
          f"min={np.min(all_raw_j2):.6f}  max={np.max(all_raw_j2):.6f}  "
          f"range={np.max(all_raw_j2)-np.min(all_raw_j2):.6f}")

    # ── 4. SUMMARY ──
    print("\n" + "=" * 70)
    print("4) DIAGNOSIS SUMMARY")
    print("-" * 70)

    # Check: normalized_state varies?
    norm_range = np.max(all_norm_j2) - np.min(all_norm_j2)
    print(f"[{'PASS' if norm_range > 0.1 else 'FAIL'}] normalized_state[J2] range = {norm_range:.6f} {'(varies)' if norm_range > 0.1 else '(CONSTANT — input/processor issue)'}")

    # Check: per-ep pred J2 varies?
    min_per_ep_std = min(np.std(per_ep[ep]["pred"][:, 1]) for ep in per_ep)
    max_per_ep_std = max(np.std(per_ep[ep]["pred"][:, 1]) for ep in per_ep)
    print(f"[{'PASS' if min_per_ep_std > 0.01 else 'FAIL'}] per-ep pred_J2_std: min={min_per_ep_std:.6f} max={max_per_ep_std:.6f} {'(varies within trajectory)' if min_per_ep_std > 0.01 else '(CONSTANT — collapsed)'}")

    # Check: input sensitivity
    # J2 +0.5 should change prediction
    obs_mod_big = {k: v.copy() for k, v in obs0.items()}
    obs_mod_big["observation.state"][1] += 0.5
    pred_big = predict_single(obs_mod_big)
    delta_big_j2 = float(pred_big[1] - base_pred[1])
    print(f"[{'PASS' if abs(delta_big_j2) > 0.01 else 'FAIL'}] sensitivity: J2 +0.5 rad → Δpred_J2 = {delta_big_j2:.6f} {'(model responds to input)' if abs(delta_big_j2) > 0.01 else '(model IGNORES input — collapse)'}")

    # improvement ratio
    print(f"[{'PASS' if overall < 0.5 else 'FAIL'}] improvement_ratio (arm only) = {overall:.6f} {'(model beats baseline)' if overall < 0.5 else '(model ≈ baseline or worse — collapse)'}")

    print()
    if min_per_ep_std < 0.01 or abs(delta_big_j2) < 0.01 or overall > 0.5:
        print("VERDICT: MODEL IS COLLAPSED — do not deploy on real robot")
    else:
        print("VERDICT: MODEL APPEARS HEALTHY — may proceed with real robot smoke test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", required=True)
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_approach_20ep")
    parser.add_argument("--episodes", default="1,2,3,4,7,10,14,15,20,22,23,24,25,26")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.checkpt, args.dataset_root, args.episodes, args.device)
