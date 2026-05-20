#!/usr/bin/env python3
"""
Offline evaluation for Hybrid v4 Delta Policy.

Checks:
1. Teacher-forcing endpoint: all episodes pred_J2 >= 1.45
2. Fixed point check: at J2=[0.9, 1.0, 1.1, 1.2], pred_delta[J2] > 0.015
3. Failure point replay: at J2=0.98, 1.12 (v3 stall points), check forward progress
4. Image sensitivity: same state, different images
5. Base/residual decomposition

Usage:
  python3 scripts/debug_hybrid_v4_delta.py \
    --checkpt outputs/train/hybrid_v4_delta_k10.pt
"""
import argparse, sys, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import torch, torchvision.transforms.functional as TF

MIN_NORM_STD = 0.01
PROGRESS_J2_THRESHOLD = 1.40
MIN_PROGRESS_DELTA = 0.015


def load_model(checkpt_path, device):
    from policies.hybrid_delta_policy import HybridDeltaPolicy

    ckpt = torch.load(checkpt_path, map_location=device, weights_only=False)
    model_args = ckpt["args"]

    model = HybridDeltaPolicy(
        state_dim=model_args.get("state_dim", 7),
        delta_dim=model_args.get("delta_dim", 6),
        img_feat_dim=model_args.get("img_feat_dim", 256),
        state_feat_dim=model_args.get("state_feat_dim", 64),
        state_hidden=model_args.get("state_hidden", 128),
        action_hidden=model_args.get("action_hidden", 256),
        use_global_img=model_args.get("use_global_img", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm_stats = ckpt["norm_stats"]
    state_mean = np.array(norm_stats["state_mean"], dtype=np.float32)
    state_std = np.maximum(np.array(norm_stats["state_std"], dtype=np.float32), MIN_NORM_STD)
    delta_mean = np.array(norm_stats["delta_mean"], dtype=np.float32)
    delta_std = np.maximum(np.array(norm_stats["delta_std"], dtype=np.float32), MIN_NORM_STD)
    lookahead_k = model_args.get("lookahead_k", 10)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  lookahead_k={lookahead_k}")
    print(f"  residual_scale={model.residual_scale.tolist()}")
    print(f"  improvement_ratio={ckpt.get('improvement_ratio', '?'):.4f}")

    return model, state_mean, state_std, delta_mean, delta_std, lookahead_k, ckpt


def load_dataset_frames(dataset_root, episodes, img_h=160):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    img_w = int(img_h * 4 / 3)
    ds = LeRobotDataset("piper/bottle_approach_20ep", root=dataset_root, episodes=episodes)

    all_frames = []
    for i in range(ds.num_frames):
        item = ds[i]
        ep_idx = int(item["episode_index"])
        if ep_idx not in episodes:
            continue
        img = item["observation.images.wrist_rgb"]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        if img.dtype != torch.float32:
            img = img.float() / 255.0
        img = TF.resize(img, (img_h, img_w), antialias=True)
        s = item["observation.state"]
        if hasattr(s, "numpy"):
            s = s.numpy()
        s = np.asarray(s, dtype=np.float32)
        a = item["action"]
        if hasattr(a, "numpy"):
            a = a.numpy()
        a = np.asarray(a, dtype=np.float32)
        all_frames.append({"ep": ep_idx, "img": img, "state": s, "action": a})

    print(f"Loaded {len(all_frames)} frames, {len(set(f['ep'] for f in all_frames))} episodes")
    return all_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", default="outputs/train/hybrid_v4_delta_k10.pt")
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_approach_20ep")
    parser.add_argument("--episodes", default="1,2,3,4,7,10,14,15,20,22,23,24,25,26")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    episode_list = [int(x.strip()) for x in args.episodes.split(",")]

    # Load model
    model, state_mean, state_std, delta_mean, delta_std, lookahead_k, ckpt = \
        load_model(args.checkpt, device)

    # Load dataset
    all_frames = load_dataset_frames(args.dataset_root, episode_list)

    per_ep = {}
    for f in all_frames:
        per_ep.setdefault(f["ep"], []).append(f)

    def norm_state(s):
        return np.clip((s - state_mean) / state_std, -5.0, 5.0)

    def denorm_delta(d_norm):
        return d_norm * delta_std + delta_mean

    # ════════════════════════════════════════════════════
    # CHECK 1: Teacher-forcing endpoint
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("CHECK 1: Teacher-Forcing Per-Ep Endpoint J2")
    print("=" * 65)

    endpoints = {}
    for ep in sorted(per_ep.keys()):
        epf = per_ep[ep]
        imgs = torch.stack([f["img"] for f in epf]).to(device)
        st_norm = np.clip((np.array([f["state"] for f in epf], dtype=np.float32)
                           - state_mean) / state_std, -5.0, 5.0)
        st = torch.from_numpy(st_norm).float().to(device)
        with torch.inference_mode():
            delta_norm = model(imgs, st).cpu().numpy()
        delta_robot = delta_norm * delta_std + delta_mean
        # Absolute action = state + delta
        abs_action = np.array([f["state"][:6] for f in epf]) + delta_robot
        endpoints[ep] = float(abs_action[-1, 1])
        ok = "OK" if abs_action[-1, 1] >= 1.45 else "FAIL"
        print(f"  Ep{ep:2d}: abs_J2={abs_action[0,1]:.4f}→{abs_action[-1,1]:.4f}"
              f"  δ_J2={delta_robot[0,1]:.4f}→{delta_robot[-1,1]:.4f}  [{ok}]")

    min_ep = min(endpoints.values())
    all_ep_ok = min_ep >= 1.45
    print(f"\n  Min endpoint: {min_ep:.4f}  {'PASS' if all_ep_ok else 'FAIL'}")

    # ════════════════════════════════════════════════════
    # CHECK 2: Fixed point
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("CHECK 2: Fixed Point — action[J2] vs current J2 (mid-ep image)")
    print("=" * 65)

    ep1 = per_ep[1]
    mid_idx = len(ep1) // 2
    mid_img = ep1[mid_idx]["img"].unsqueeze(0).to(device)
    mid_state = np.asarray(ep1[mid_idx]["state"], dtype=np.float32).copy()

    # Register hooks
    caps = {}
    model.state_head.register_forward_hook(lambda m, inp, out: caps.update(base_norm=out.detach()))
    model.image_delta_head.register_forward_hook(lambda m, inp, out: caps.update(img_res_norm=out.detach()))

    test_points = [0.0, 0.3, 0.6, 0.9, 0.98, 1.0, 1.12, 1.2, 1.4, 1.6]
    print(f"  {'J2_in':>8} {'base_δ[J2]':>12} {'img_res[J2]':>12} {'δ[J2]':>10} {'action[J2]':>12} {'gap':>10} {'status'}")
    fp_ok = True
    for test_j2 in test_points:
        st = mid_state.copy()
        st[1] = test_j2
        st_norm = norm_state(st)
        with torch.inference_mode():
            _ = model(mid_img, torch.from_numpy(st_norm).float().unsqueeze(0).to(device))
        # Raw head outputs are normalized, need to denorm with delta_std
        base_norm = caps["base_norm"].squeeze(0).cpu().numpy()
        img_res_norm = caps["img_res_norm"].squeeze(0).cpu().numpy()
        # img_res_norm goes through tanh then residual_scale, so it's already in normalized delta space

        base_robot = base_norm * delta_std + delta_mean
        img_res_robot = img_res_norm * delta_std + delta_mean
        delta_robot = base_robot + img_res_robot  # same as model output
        action_j2 = test_j2 + delta_robot[1]
        gap = delta_robot[1]

        status = ""
        if test_j2 < PROGRESS_J2_THRESHOLD:
            if delta_robot[1] < MIN_PROGRESS_DELTA:
                status = "STALL?"
                fp_ok = False
            else:
                status = "OK"
        print(f"  {test_j2:8.3f} {base_robot[1]:12.4f} {img_res_robot[1]:12.4f} "
              f"{delta_robot[1]:10.4f} {action_j2:12.4f} {gap:+10.4f}  {status}")

    print(f"\n  Fixed point check: {'PASS' if fp_ok else 'FAIL'}")

    # ════════════════════════════════════════════════════
    # CHECK 3: Failure point replay (v3 stall points)
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("CHECK 3: V3 Stall Point Replay (J2=0.98, J2=1.12)")
    print("=" * 65)

    for stall_j2 in [0.98, 1.12]:
        st = mid_state.copy()
        st[1] = stall_j2
        st_norm = norm_state(st)
        with torch.inference_mode():
            delta_norm, base_norm, img_res_norm = model.forward_with_internals(
                mid_img, torch.from_numpy(st_norm).float().unsqueeze(0).to(device))
        delta_robot = delta_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        base_robot = base_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        img_res_robot = img_res_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        action_j2 = stall_j2 + delta_robot[1]
        ok = delta_robot[1] > MIN_PROGRESS_DELTA
        print(f"  J2={stall_j2:.2f}: base_δ={base_robot[1]:.4f}  img_res={img_res_robot[1]:.4f}"
              f"  δ={delta_robot[1]:.4f}  action={action_j2:.4f}  gap={delta_robot[1]:+.4f}"
              f"  [{'OK' if ok else 'STALL'}]")

    # ════════════════════════════════════════════════════
    # CHECK 4: Image sensitivity
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("CHECK 4: Image Sensitivity (same state[J2=0.98], different ep images)")
    print("=" * 65)

    test_state = mid_state.copy()
    test_state[1] = 0.98
    st_norm = norm_state(test_state)
    st_t = torch.from_numpy(st_norm).float().unsqueeze(0).to(device)

    print(f"  {'ep':>6} {'base_δ[J2]':>12} {'img_res[J2]':>12} {'δ[J2]':>10} {'action[J2]':>12}")

    all_img_res = []
    for ep in sorted(per_ep.keys())[:6]:
        epf = per_ep[ep]
        best_i = min(range(len(epf)), key=lambda i: abs(epf[i]["state"][1] - 0.98))
        img_t = epf[best_i]["img"].unsqueeze(0).to(device)

        with torch.inference_mode():
            delta_norm, base_norm, img_res_norm = model.forward_with_internals(img_t, st_t)
        delta_robot = delta_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        base_robot = base_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        img_res_robot = img_res_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        action_j2 = 0.98 + delta_robot[1]
        all_img_res.append(img_res_robot)
        print(f"  {ep:6d} {base_robot[1]:12.4f} {img_res_robot[1]:12.4f} "
              f"{delta_robot[1]:10.4f} {action_j2:12.4f}")

    if all_img_res:
        img_res_std = np.std([r[1] for r in all_img_res])
        print(f"\n  Image residual[J2] std across episodes: {img_res_std:.6f}")
        if img_res_std > 0.0005:
            print(f"  Image sensitivity: PASS (residual varies with image)")
        else:
            print(f"  Image sensitivity: FAIL (residual is constant, image ignored)")

    # ════════════════════════════════════════════════════
    # CHECK 5: Base/residual decomposition across trajectory
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("CHECK 5: Base/Residual Decomposition Along Ep1 Trajectory")
    print("=" * 65)

    ep1 = per_ep[1]
    n_ep1 = len(ep1)
    sample_idxs = [0, n_ep1 // 5, n_ep1 // 3, n_ep1 // 2, 2 * n_ep1 // 3, n_ep1 - 1]

    print(f"  {'idx':>6} {'true_J2':>8} {'base_δ[J2]':>12} {'img_res[J2]':>12} {'δ[J2]':>10} {'act[J2]':>10}")
    for idx in sample_idxs:
        f = ep1[idx]
        st_norm = norm_state(f["state"])
        img_t = f["img"].unsqueeze(0).to(device)
        with torch.inference_mode():
            delta_norm, base_norm, img_res_norm = model.forward_with_internals(
                img_t, torch.from_numpy(st_norm).float().unsqueeze(0).to(device))
        delta_robot = delta_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        base_robot = base_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        img_res_robot = img_res_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        action_j2 = f["state"][1] + delta_robot[1]
        print(f"  {idx:6d} {f['state'][1]:8.4f} {base_robot[1]:12.4f} {img_res_robot[1]:12.4f} "
              f"{delta_robot[1]:10.4f} {action_j2:10.4f}")

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  [{'PASS' if all_ep_ok else 'FAIL'}] All ep endpoint >= 1.45 (min={min_ep:.4f})")
    print(f"  [{'PASS' if fp_ok else 'FAIL'}] Fixed point: delta[J2] > {MIN_PROGRESS_DELTA} when J2 < {PROGRESS_J2_THRESHOLD}")
    print(f"  [{'PASS' if img_res_std > 0.0005 else 'FAIL'}] Image sensitivity non-zero")
    print(f"  improvement_ratio: {ckpt.get('improvement_ratio', '?'):.4f}")

    deploy_ready = all_ep_ok and fp_ok
    if deploy_ready:
        print("\nREADY for real robot Test A.")
    else:
        print("\nNOT ready — fix issues before deploying.")


if __name__ == "__main__":
    main()
