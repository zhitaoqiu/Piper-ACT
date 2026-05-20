#!/usr/bin/env python3
"""
Train Hybrid v4: Lookahead Delta Policy.

Predicts delta from current state to action[t+K], not absolute action.
Key: avoids identity fixed point by construction — action = state + delta,
delta learned from future frames guarantees forward progress.

Usage:
  python3 scripts/train_hybrid_v4_delta.py  # default K=10
  python3 scripts/train_hybrid_v4_delta.py --lookahead-k 5  # less aggressive
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policies.hybrid_delta_policy import HybridDeltaPolicy

MIN_NORM_STD = 0.01
NORM_STATE_CLIP = 5.0
MIN_PROGRESS_DELTA = 0.015  # minimum desired delta[J2] when progress is expected
PROGRESS_J2_THRESHOLD = 1.40  # only apply progress loss below this J2


def load_dataset_cached(dataset_root, episodes, img_size=(160, 213), device="cuda"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torchvision.transforms.functional as TF

    ds = LeRobotDataset("piper/bottle_approach_20ep", root=dataset_root, episodes=episodes)
    print(f"Dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")

    all_images, all_states, all_actions, all_ep_indices = [], [], [], []
    ep_to_frames = {}

    print("Loading frames into memory...")
    t0 = time.time()
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
        img = TF.resize(img, img_size, antialias=True)

        s = item["observation.state"]
        if hasattr(s, "numpy"):
            s = s.numpy()
        s = np.asarray(s, dtype=np.float32)

        a = item["action"]
        if hasattr(a, "numpy"):
            a = a.numpy()
        a = np.asarray(a, dtype=np.float32)

        all_images.append(img)
        all_states.append(s)
        all_actions.append(a)
        all_ep_indices.append(ep_idx)
        ep_to_frames.setdefault(ep_idx, []).append(len(all_states) - 1)

    elapsed = time.time() - t0
    print(f"Loaded {len(all_states)} frames in {elapsed:.1f}s")

    images = torch.stack(all_images)
    states = torch.from_numpy(np.stack(all_states)).float()
    actions = torch.from_numpy(np.stack(all_actions)).float()
    ep_indices_arr = np.array(all_ep_indices)

    per_ep = {}
    for ep, frame_ids in ep_to_frames.items():
        per_ep[ep] = {
            "images": images[frame_ids],
            "states": states[frame_ids],
            "actions": actions[frame_ids],
        }

    print(f"Images: {images.shape}, States: {states.shape}, Actions: {actions.shape}")
    return images, states, actions, ep_indices_arr, per_ep, episodes


def compute_lookahead_deltas(states, actions, ep_indices_arr, lookahead_k):
    """
    For each frame, target_delta = actions[t+K][:6] - states[t][:6].
    If t+K exceeds episode end, use the last action of that episode.
    Returns: target_deltas (N, 6) numpy array.
    """
    states_np = states.numpy()
    actions_np = actions.numpy()
    target_deltas = np.zeros((len(states), 6), dtype=np.float32)

    unique_eps = np.unique(ep_indices_arr)
    for ep in unique_eps:
        ep_mask = ep_indices_arr == ep
        ep_indices = np.where(ep_mask)[0]
        n_ep = len(ep_indices)

        for i, global_idx in enumerate(ep_indices):
            future_idx = min(i + lookahead_k, n_ep - 1)
            future_global = ep_indices[future_idx]
            # Delta = future_action[arm] - current_state[arm]
            target_deltas[global_idx] = actions_np[future_global, :6] - states_np[global_idx, :6]

    return target_deltas


def compute_norm_stats(all_states, all_deltas):
    state_mean = all_states.mean(axis=0)
    state_std = np.maximum(all_states.std(axis=0), MIN_NORM_STD)
    delta_mean = all_deltas.mean(axis=0)
    delta_std = np.maximum(all_deltas.std(axis=0), MIN_NORM_STD)
    return {
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "delta_mean": delta_mean.tolist(),
        "delta_std": delta_std.tolist(),
        "min_std": MIN_NORM_STD,
        "state_clip": NORM_STATE_CLIP,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_approach_20ep")
    parser.add_argument("--episodes", default="1,2,3,4,7,10,14,15,20,22,23,24,25,26")
    parser.add_argument("--lookahead-k", type=int, default=10,
                        help="Number of steps to look ahead for delta target (default: 10)")
    parser.add_argument("--img-feat-dim", type=int, default=256)
    parser.add_argument("--state-feat-dim", type=int, default=64)
    parser.add_argument("--state-hidden", type=int, default=128)
    parser.add_argument("--action-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/train/hybrid_v4_delta_k10.pt")
    parser.add_argument("--progress-loss-weight", type=float, default=0.2,
                        help="Weight for progress loss (default: 0.2)")
    parser.add_argument("--grad-log-every", type=int, default=250)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Lookahead K: {args.lookahead_k}")

    episode_list = [int(x.strip()) for x in args.episodes.split(",")]
    h, w = args.img_size, int(args.img_size * 4 / 3)
    img_size = (h, w)
    print(f"Image resize: {img_size}")

    # ── Load dataset ──
    images, states, actions, ep_indices, per_ep, eps_loaded = load_dataset_cached(
        args.dataset_root, episode_list, img_size=img_size
    )
    n_total = len(images)

    # ── Compute lookahead deltas ──
    target_deltas_raw = compute_lookahead_deltas(states, actions, ep_indices, args.lookahead_k)
    print(f"\nTarget deltas computed (K={args.lookahead_k}):")
    print(f"  Delta mean:     {np.round(target_deltas_raw.mean(axis=0), 4)}")
    print(f"  Delta std:      {np.round(target_deltas_raw.std(axis=0), 4)}")
    print(f"  Delta[J2] range: [{target_deltas_raw[:,1].min():.4f}, {target_deltas_raw[:,1].max():.4f}]")
    print(f"  Delta[J2] < 0:  {(target_deltas_raw[:,1] < 0).sum()}/{n_total} frames")

    # ── Compute norm stats ──
    states_np = states.numpy()
    norm_stats = compute_norm_stats(states_np, target_deltas_raw)
    state_mean = np.array(norm_stats["state_mean"], dtype=np.float32)
    state_std = np.array(norm_stats["state_std"], dtype=np.float32)
    delta_mean = np.array(norm_stats["delta_mean"], dtype=np.float32)
    delta_std = np.array(norm_stats["delta_std"], dtype=np.float32)
    delta_mean_t = torch.from_numpy(delta_mean).to(device)
    delta_std_t = torch.from_numpy(delta_std).to(device)

    print(f"\nState mean:  {np.round(state_mean, 4)}")
    print(f"State std:   {np.round(state_std, 4)}")
    print(f"Delta mean:  {np.round(delta_mean, 4)}")
    print(f"Delta std:   {np.round(delta_std, 4)}")

    # ── Normalize (on CPU first, then move to device) ──
    target_deltas_t = torch.from_numpy(target_deltas_raw).float()
    state_mean_t = torch.from_numpy(state_mean).float()
    state_std_t = torch.from_numpy(state_std).float()
    delta_mean_t_cpu = torch.from_numpy(delta_mean).float()
    delta_std_t_cpu = torch.from_numpy(delta_std).float()
    states_norm = (states - state_mean_t) / state_std_t
    target_deltas_norm = (target_deltas_t - delta_mean_t_cpu) / delta_std_t_cpu

    # ── Train/val split ──
    n_val = max(int(n_total * 0.2), 64)
    indices = torch.randperm(n_total)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    images = images.to(device)
    states_norm = states_norm.to(device)
    target_deltas_norm = target_deltas_norm.to(device)
    target_deltas_t = target_deltas_t.to(device)
    # Keep raw states on device for progress loss condition
    states_raw_t = states.to(device)

    # ── Model ──
    model = HybridDeltaPolicy(
        state_dim=7, delta_dim=6,
        img_feat_dim=args.img_feat_dim,
        state_feat_dim=args.state_feat_dim,
        state_hidden=args.state_hidden,
        action_hidden=args.action_hidden,
        use_global_img=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params")
    print(f"  residual_scale: {model.residual_scale.tolist()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    print(f"\nTraining {args.steps} steps, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  progress_loss_weight={args.progress_loss_weight}")
    print(f"  MIN_PROGRESS_DELTA={MIN_PROGRESS_DELTA}")
    t0 = time.time()
    n_train = len(train_idx)

    val_states_raw = states_raw_t[val_idx]
    val_states_norm = states_norm[val_idx]
    val_targets_norm = target_deltas_norm[val_idx]
    val_img = images[val_idx]

    train_losses, val_losses = [], []
    progress_losses = []
    grad_log = []

    for step in range(args.steps):
        model.train()
        n_batch = max(1, n_train // args.batch_size)
        if step % n_batch == 0:
            train_perm = torch.randperm(n_train)
            train_idx_shuf = train_idx[train_perm]

        b_start = (step % n_batch) * args.batch_size
        bidx = train_idx_shuf[b_start:b_start + args.batch_size]

        x_img = images[bidx]
        x_state_norm = states_norm[bidx]
        y_delta_norm = target_deltas_norm[bidx]
        x_state_raw = states_raw_t[bidx]
        y_delta_raw = target_deltas_t[bidx]

        pred_delta = model(x_img, x_state_norm)

        # ── MSE loss on delta ──
        mse = mse_loss(pred_delta, y_delta_norm)

        # ── Progress loss: encourage delta[J2] >= MIN_PROGRESS_DELTA when progress expected ──
        progress_loss = torch.tensor(0.0, device=device)
        # Condition: current_qpos[J2] < PROGRESS_J2_THRESHOLD AND true_delta[J2] > 0
        progress_mask = (x_state_raw[:, 1] < PROGRESS_J2_THRESHOLD) & (y_delta_raw[:, 1] > 0)
        if progress_mask.any():
            pred_delta_j2_norm = pred_delta[progress_mask, 1]
            # Convert normalized delta[J2] to robot units for threshold
            pred_delta_j2_robot = pred_delta_j2_norm * delta_std_t[1] + delta_mean_t[1]
            # relu(MIN_PROGRESS_DELTA - pred_delta_j2_robot)^2
            shortfall = F.relu(MIN_PROGRESS_DELTA - pred_delta_j2_robot)
            progress_loss = (shortfall ** 2).mean()

        loss = mse + args.progress_loss_weight * progress_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── Gradient norm logging ──
        if (step == 0 or (step + 1) % args.grad_log_every == 0):
            ie_gn = sum(p.grad.norm().item() ** 2 for p in model.image_encoder.parameters()
                        if p.grad is not None) ** 0.5
            sm_gn = sum(p.grad.norm().item() ** 2 for p in model.state_mlp.parameters()
                        if p.grad is not None) ** 0.5
            sh_gn = sum(p.grad.norm().item() ** 2 for p in model.state_head.parameters()
                        if p.grad is not None) ** 0.5
            id_gn = sum(p.grad.norm().item() ** 2 for p in model.image_delta_head.parameters()
                        if p.grad is not None) ** 0.5
            grad_log.append({
                "step": step + 1,
                "img_enc_grad": ie_gn,
                "state_mlp_grad": sm_gn,
                "state_head_grad": sh_gn,
                "img_delta_grad": id_gn,
            })

        if step == 0 or (step + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_img, val_states_norm)
                val_mse = mse_loss(val_pred, val_targets_norm).item()
                train_mse = mse.item()
            train_losses.append((step + 1, train_mse))
            val_losses.append((step + 1, val_mse))
            progress_losses.append((step + 1, progress_loss.item()))

            gl = grad_log[-1] if grad_log else {}
            print(f"  step {step+1:5d}/{args.steps}  train_mse={train_mse:.6f}  val_mse={val_mse:.6f}"
                  f"  prog_loss={progress_loss.item():.6f}"
                  f"  |gie|={gl.get('img_enc_grad', 0):.3f}  |gst|={gl.get('state_mlp_grad', 0):.3f}"
                  f"  |gsh|={gl.get('state_head_grad', 0):.3f}  |gid|={gl.get('img_delta_grad', 0):.3f}")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # EVALUATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()
    dim_names = ["J1", "J2", "J3", "J4", "J5", "J6"]

    # Predict all frames
    all_preds_norm = []
    bs_eval = 64
    for i in range(0, n_total, bs_eval):
        with torch.no_grad():
            bp = model(images[i:i + bs_eval], states_norm[i:i + bs_eval])
        all_preds_norm.append(bp.cpu().numpy())
    all_preds_norm = np.concatenate(all_preds_norm)
    all_preds_delta = all_preds_norm * delta_std + delta_mean

    # Compute predicted absolute actions: action = state + delta
    all_preds_abs = states_np[:, :6] + all_preds_delta
    all_actions_np = actions.numpy()

    # 1. Per-dim delta stats
    print(f"\n{'Dim':>6}  {'pred_δ_mean':>12}  {'pred_δ_std':>12}  {'true_δ_mean':>12}  {'true_δ_std':>12}  {'status'}")
    for d, name in enumerate(dim_names):
        ps = all_preds_delta[:, d]
        ts = target_deltas_raw[:, d]
        ok = "COLLAPSED" if np.std(ps) < 0.0001 else "OK"
        print(f"{name:>6}  {np.mean(ps):12.6f}  {np.std(ps):12.6f}  {np.mean(ts):12.6f}  {np.std(ts):12.6f}  {ok}")

    # 2. improvement_ratio on delta
    mse_delta = np.mean((all_preds_delta - target_deltas_raw) ** 2, axis=0)
    baseline_delta = np.mean((target_deltas_raw - target_deltas_raw.mean(axis=0)) ** 2, axis=0)
    print(f"\n{'Dim':>6}  {'model_mse(δ)':>14}  {'baseline_mse(δ)':>14}  {'ratio':>10}  {'status'}")
    for d, name in enumerate(dim_names):
        ratio = mse_delta[d] / baseline_delta[d] if baseline_delta[d] > 1e-10 else float("inf")
        flag = "*** COLLAPSED ***" if ratio > 0.9 else ""
        print(f"{name:>6}  {mse_delta[d]:14.6f}  {baseline_delta[d]:14.6f}  {ratio:10.6f}  {flag}")
    overall = np.mean(mse_delta) / np.mean(baseline_delta)
    improvement_ratio = overall
    print(f"\nOverall improvement_ratio (delta): {overall:.6f}")

    # 3. Per-ep pred_J2 curves (convert delta to absolute action)
    print("\n--- Per-Ep pred_J2 (delta → absolute action) ---")
    ep_endpoints = {}
    for ep in sorted(per_ep.keys()):
        ep_data = per_ep[ep]
        ep_states_raw = ep_data["states"]
        ep_states_norm = (ep_states_raw - torch.from_numpy(state_mean)) / torch.from_numpy(state_std)
        with torch.no_grad():
            ep_pred_delta_norm = model(ep_data["images"].to(device),
                                        ep_states_norm.to(device)).cpu().numpy()
        ep_pred_delta = ep_pred_delta_norm * delta_std + delta_mean
        # Absolute action = state + delta
        ep_pred_abs = ep_states_raw.numpy()[:, :6] + ep_pred_delta
        pj2 = ep_pred_abs[:, 1]
        tj2 = ep_data["actions"].numpy()[:, 1]
        ep_endpoints[ep] = float(pj2[-1])
        print(f"Ep{ep:2d}: pred_J2=[{pj2[0]:.4f} → {pj2[-1]:.4f}] std={np.std(pj2):.4f}  "
              f"true_J2=[{tj2[0]:.4f} → {tj2[-1]:.4f}]")

    # Check all endpoints >= 1.45
    min_ep = min(ep_endpoints.values())
    all_ep_ok = min_ep >= 1.45
    print(f"\nMin ep endpoint J2: {min_ep:.4f}  {'PASS' if all_ep_ok else 'FAIL'}")

    # 4. STATE SENSITIVITY TEST
    print("\n" + "=" * 70)
    print("STATE SENSITIVITY TEST (same image, different J2)")
    print("=" * 70)

    anchor_img = images[0:1]
    anchor_state_norm = states_norm[0:1].clone()
    with torch.no_grad():
        base_pred_delta_norm = model(anchor_img, anchor_state_norm).squeeze(0).cpu().numpy()
    base_pred_delta = base_pred_delta_norm * delta_std + delta_mean

    print(f"Base delta J2: {base_pred_delta[1]:.6f}")
    print(f"Base abs action[J2]: {states_np[0, 1] + base_pred_delta[1]:.6f}")

    tests = {"J2 +0.01": (1, 0.01), "J2 +0.10": (1, 0.10),
             "J2 +0.50": (1, 0.50), "J2 +1.00": (1, 1.00),
             "J1 +0.50": (0, 0.50), "J3 +0.50": (2, 0.50)}

    all_sensitive = True
    for label, (dim, delta_val) in tests.items():
        mod_state = anchor_state_norm.clone()
        mod_state[0, dim] += delta_val
        with torch.no_grad():
            mod_pred_delta_norm = model(anchor_img, mod_state).squeeze(0).cpu().numpy()
        mod_pred_delta = mod_pred_delta_norm * delta_std + delta_mean
        d_delta = mod_pred_delta - base_pred_delta
        d_j2_delta = d_delta[1]
        flag = "OK" if abs(d_j2_delta) > 0.0001 else "FAIL"
        if abs(d_j2_delta) <= 0.0001:
            all_sensitive = False
        print(f"  {label:>12}:  Δdelta_J2={d_j2_delta:+8.6f}  |Δdelta|={np.linalg.norm(d_delta):.6f}  [{flag}]")

    # 5. IMAGE SENSITIVITY TEST
    print("\n" + "=" * 70)
    print("IMAGE SENSITIVITY TEST (same state, different ep images)")
    print("=" * 70)

    anchor_state_t = states_norm[0:1]
    img_sensitive = True
    prev_pred = None
    for ep_test in episode_list[:5]:
        ep_mask = ep_indices == ep_test
        if not ep_mask.any():
            continue
        ep_first_idx = np.where(ep_mask)[0][0]
        ep_img = images[ep_first_idx:ep_first_idx + 1]
        with torch.no_grad():
            ep_pred_delta_norm = model(ep_img, anchor_state_t).squeeze(0).cpu().numpy()
        ep_pred_delta = ep_pred_delta_norm * delta_std + delta_mean
        if prev_pred is not None:
            d = np.linalg.norm(ep_pred_delta - prev_pred)
            print(f"  ep{ep_test} vs previous:  Δdelta_J2={ep_pred_delta[1]-prev_pred[1]:+.6f}  |Δδ|={d:.6f}")
            if d < 0.0005:
                img_sensitive = False
        else:
            print(f"  ep{ep_test} (anchor):  delta_J2={ep_pred_delta[1]:+.6f}  δ={np.round(ep_pred_delta, 4)}")
        prev_pred = ep_pred_delta

    # 6. IMAGE MASK TEST
    print("\n" + "=" * 70)
    print("IMAGE MASK TEST (same state, real vs black vs noise)")
    print("=" * 70)

    img_real = images[0:1]
    img_black = torch.zeros_like(img_real)
    img_noise = torch.clamp(torch.randn_like(img_real) * 0.1 + 0.5, 0, 1)

    mask_preds = {}
    for label, img in [("real", img_real), ("black", img_black), ("noise", img_noise)]:
        with torch.no_grad():
            pn = model(img, anchor_state_t).squeeze(0).cpu().numpy()
        pr = pn * delta_std + delta_mean
        mask_preds[label] = pr
        print(f"  {label:>8}: delta_J2={pr[1]:+.6f}  δ={np.round(pr, 4)}")

    d_rb = np.linalg.norm(mask_preds["real"] - mask_preds["black"])
    d_rn = np.linalg.norm(mask_preds["real"] - mask_preds["noise"])
    print(f"  |real-black|={d_rb:.6f}  |real-noise|={d_rn:.6f}")
    mask_ok = max(d_rb, d_rn) > 0.001

    # 7. FIXED POINT CHECK
    print("\n" + "=" * 70)
    print("FIXED POINT CHECK (J2=0.9, 0.98, 1.12, 1.2)")
    print("=" * 70)

    mid_img = images[n_total // 2:n_total // 2 + 1]
    mid_state_norm = states_norm[n_total // 2:n_total // 2 + 1].clone()
    mid_state_raw = states_np[n_total // 2]

    print(f"  {'J2_in':>8}  {'base_δ[J2]':>12}  {'img_res[J2]':>12}  {'δ[J2]':>10}  {'action[J2]':>12}  {'gap':>10}  {'status'}")
    fp_ok = True
    for test_j2 in [0.0, 0.3, 0.6, 0.9, 0.98, 1.12, 1.2, 1.4, 1.6]:
        st_raw = mid_state_raw.copy()
        st_raw[1] = test_j2
        st_norm = (st_raw - state_mean) / state_std
        st_t = torch.from_numpy(st_norm).float().unsqueeze(0).to(device)
        with torch.no_grad():
            delta_norm, base_norm, img_res_norm = model.forward_with_internals(mid_img, st_t)
        delta_robot = delta_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        base_robot = base_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        img_res_robot = img_res_norm.squeeze(0).cpu().numpy() * delta_std + delta_mean
        action_j2 = test_j2 + delta_robot[1]
        gap = action_j2 - test_j2
        status = ""
        if test_j2 < PROGRESS_J2_THRESHOLD:
            if delta_robot[1] < 0.015:
                status = "STALL?"
                fp_ok = False
            else:
                status = "OK"
        print(f"  {test_j2:8.3f}  {base_robot[1]:12.4f}  {img_res_robot[1]:12.4f}  "
              f"{delta_robot[1]:10.4f}  {action_j2:12.4f}  {gap:+10.4f}  {status}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = 0
    total = 7
    checks = []

    checks.append(("All ep endpoint J2 >= 1.45", all_ep_ok, min_ep))
    passed += int(all_ep_ok)

    checks.append(("State sensitivity non-zero", all_sensitive, None))
    passed += int(all_sensitive)

    checks.append(("Image sensitivity non-zero", img_sensitive, None))
    passed += int(img_sensitive)

    checks.append(("Image mask OK (image matters)", mask_ok, None))
    passed += int(mask_ok)

    checks.append(("Fixed point OK (delta[J2] > 0.015 when J2 < 1.40)", fp_ok, None))
    passed += int(fp_ok)

    checks.append((f"improvement_ratio < 0.25 ({improvement_ratio:.4f})",
                   improvement_ratio < 0.25, improvement_ratio))
    passed += int(improvement_ratio < 0.25)

    print(f"\n  {passed}/{total} checks passed")
    for name, ok, val in checks:
        vstr = f" ({val:.4f})" if val is not None else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{vstr}")

    # ═══════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": {
            "state_dim": 7,
            "delta_dim": 6,
            "img_feat_dim": args.img_feat_dim,
            "state_feat_dim": args.state_feat_dim,
            "state_hidden": args.state_hidden,
            "action_hidden": args.action_hidden,
            "img_size": args.img_size,
            "lookahead_k": args.lookahead_k,
            "progress_loss_weight": args.progress_loss_weight,
            "use_global_img": False,
        },
        "norm_stats": norm_stats,
        "improvement_ratio": improvement_ratio,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "progress_losses": progress_losses,
        "grad_log": grad_log,
        "offline_checks": {
            "all_ep_endpoint_ok": all_ep_ok,
            "min_ep_endpoint": min_ep,
            "state_sensitive": all_sensitive,
            "img_sensitive": img_sensitive,
            "img_mask_ok": mask_ok,
            "fixed_point_ok": fp_ok,
            "improvement_ratio": improvement_ratio,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"\nCheckpoint saved to {output_path}")

    # Also save norm_stats as JSON for easy inspection
    norm_json_path = output_path.with_suffix(".json")
    with open(norm_json_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Norm stats saved to {norm_json_path}")

    deploy_ready = passed >= 6
    if deploy_ready:
        print("\nALL CRITICAL CHECKS PASSED — model is ready for real robot Test A.")
    else:
        print("\nSOME CHECKS FAILED — do NOT deploy on real robot until fixed.")


if __name__ == "__main__":
    main()
