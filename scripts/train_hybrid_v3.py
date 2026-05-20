#!/usr/bin/env python3
"""
Train Hybrid v3: state-conditioned policy with image residual head.
Key: LayerNorm + learnable gates + image_delta architecture.
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policies.state_conditioned_policy_v3 import StateConditionedPolicyV3

MIN_NORM_STD = 0.01


def load_dataset_cached(dataset_root, episodes, img_size=(120, 160), device="cuda"):
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


def compute_norm_stats(all_states, all_actions):
    state_mean = all_states.mean(axis=0)
    state_std = np.maximum(all_states.std(axis=0), MIN_NORM_STD)
    action_mean = all_actions.mean(axis=0)
    action_std = np.maximum(all_actions.std(axis=0), MIN_NORM_STD)
    return {
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_approach_20ep")
    parser.add_argument("--episodes", default="1,2,3,4,7,10,14,15,20,22,23,24,25,26")
    parser.add_argument("--img-feat-dim", type=int, default=256)
    parser.add_argument("--state-feat-dim", type=int, default=64)
    parser.add_argument("--state-hidden", type=int, default=128)
    parser.add_argument("--action-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/train/hybrid_v3.pt")
    parser.add_argument("--grad-log-every", type=int, default=250,
                        help="Log gradient norms every N steps")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    episode_list = [int(x.strip()) for x in args.episodes.split(",")]
    h, w = args.img_size, int(args.img_size * 4 / 3)
    img_size = (h, w)
    print(f"Image resize: {img_size}")

    # Load dataset
    images, states, actions, ep_indices, per_ep, eps_loaded = load_dataset_cached(
        args.dataset_root, episode_list, img_size=img_size
    )
    n_total = len(images)

    # Compute norm stats
    states_np = states.numpy()
    actions_np = actions.numpy()
    norm_stats = compute_norm_stats(states_np, actions_np)
    state_mean = np.array(norm_stats["state_mean"], dtype=np.float32)
    state_std = np.array(norm_stats["state_std"], dtype=np.float32)
    action_mean = np.array(norm_stats["action_mean"], dtype=np.float32)
    action_std = np.array(norm_stats["action_std"], dtype=np.float32)
    action_mean_t = torch.from_numpy(action_mean).to(device)
    action_std_t = torch.from_numpy(action_std).to(device)

    print(f"State mean:  {np.round(state_mean, 4)}")
    print(f"State std:   {np.round(state_std, 4)}")
    print(f"Action mean: {np.round(action_mean, 4)}")
    print(f"Action std:  {np.round(action_std, 4)}")

    # Normalize states and actions
    states_raw = states.clone()
    actions_raw = actions.clone()
    states = (states - torch.from_numpy(state_mean)) / torch.from_numpy(state_std)
    actions = (actions - torch.from_numpy(action_mean)) / torch.from_numpy(action_std)

    # Train/val split
    n_val = max(int(n_total * 0.2), 64)
    indices = torch.randperm(n_total)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    images = images.to(device)
    states = states.to(device)
    actions = actions.to(device)

    # Model
    model = StateConditionedPolicyV3(
        state_dim=7, action_dim=7,
        img_feat_dim=args.img_feat_dim,
        state_feat_dim=args.state_feat_dim,
        state_hidden=args.state_hidden,
        action_hidden=args.action_hidden,
        use_global_img=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"  img_gate init: {model.img_gate.item():.2f}")
    print(f"  state_gate init: {model.state_gate.item():.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"\nTraining {args.steps} steps, batch_size={args.batch_size}, lr={args.lr} ...")
    t0 = time.time()
    n_train = len(train_idx)

    val_img = images[val_idx]
    val_state = states[val_idx]
    val_act = actions[val_idx]

    train_losses, val_losses = [], []
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
        x_state = states[bidx]
        y = actions[bidx]

        pred = model(x_img, x_state)
        loss = loss_fn(pred, y)

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
                "img_gate_val": model.img_gate.item(),
                "state_gate_val": model.state_gate.item(),
            })

        if step == 0 or (step + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_img, val_state)
                val_loss = loss_fn(val_pred, val_act).item()
                train_loss = loss.item()
            train_losses.append((step + 1, train_loss))
            val_losses.append((step + 1, val_loss))

            gl = grad_log[-1] if grad_log else {}
            print(f"  step {step+1:5d}/{args.steps}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
                  f"  img_gate={gl.get('img_gate_val', 0):.3f}  state_gate={gl.get('state_gate_val', 0):.3f}"
                  f"  |g_img|={gl.get('img_enc_grad', 0):.3f}  |g_state|={gl.get('state_mlp_grad', 0):.3f}"
                  f"  |g_shead|={gl.get('state_head_grad', 0):.3f}  |g_idelta|={gl.get('img_delta_grad', 0):.3f}")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    # ── EVALUATION ──
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()
    dim_names = ["J1", "J2", "J3", "J4", "J5", "J6", "grip"]

    # Predict all frames
    all_preds_norm = []
    bs_eval = 64
    for i in range(0, n_total, bs_eval):
        with torch.no_grad():
            bp = model(images[i:i + bs_eval], states[i:i + bs_eval])
        all_preds_norm.append(bp.cpu().numpy())
    all_preds_norm = np.concatenate(all_preds_norm)
    all_preds = all_preds_norm * action_std + action_mean
    all_acts_np = actions_raw.numpy()

    # 1. Per-dim stats
    print(f"\n{'Dim':>6}  {'pred_mean':>10}  {'pred_std':>10}  {'true_mean':>10}  {'true_std':>10}  {'status'}")
    for d, name in enumerate(dim_names):
        ps = all_preds[:, d]
        ts = all_acts_np[:, d]
        ok = "COLLAPSED" if np.std(ps) < 0.0001 else "OK"
        print(f"{name:>6}  {np.mean(ps):10.6f}  {np.std(ps):10.6f}  {np.mean(ts):10.6f}  {np.std(ts):10.6f}  {ok}")

    # 2. improvement_ratio
    mse_per_dim = np.mean((all_preds - all_acts_np) ** 2, axis=0)
    mean_baseline = np.mean(all_acts_np, axis=0)
    mse_baseline = np.mean((all_acts_np - mean_baseline) ** 2, axis=0)
    print(f"\n{'Dim':>6}  {'model_mse':>12}  {'baseline_mse':>12}  {'ratio':>10}  {'status'}")
    for d, name in enumerate(dim_names):
        ratio = mse_per_dim[d] / mse_baseline[d] if mse_baseline[d] > 1e-10 else float("inf")
        flag = "*** COLLAPSED ***" if ratio > 0.9 else ""
        print(f"{name:>6}  {mse_per_dim[d]:12.6f}  {mse_baseline[d]:12.6f}  {ratio:10.6f}  {flag}")
    overall = np.mean(mse_per_dim[:6]) / np.mean(mse_baseline[:6])
    print(f"\nArm-only improvement_ratio: {overall:.6f}")

    # 3. Per-ep pred_J2 curves + endpoint check
    print("\n--- Per-Ep pred J2 ---")
    ep_endpoints = {}
    for ep in sorted(per_ep.keys()):
        ep_data = per_ep[ep]
        # Need normalized state for inference
        ep_states_raw = ep_data["states"]
        ep_states_norm = (ep_states_raw - torch.from_numpy(state_mean)) / torch.from_numpy(state_std)
        with torch.no_grad():
            ep_pred_norm = model(ep_data["images"].to(device), ep_states_norm.to(device)).cpu().numpy()
        ep_pred = ep_pred_norm * action_std + action_mean
        pj2 = ep_pred[:, 1]
        tj2 = ep_data["actions"].numpy()[:, 1]
        ep_endpoints[ep] = float(pj2[-1])
        print(f"Ep{ep:2d}: pred_J2=[{pj2[0]:.4f}→{pj2[-1]:.4f}] std={np.std(pj2):.4f}  "
              f"true_J2=[{tj2[0]:.4f}→{tj2[-1]:.4f}]")

    # Check all endpoints >= 1.45
    min_ep = min(ep_endpoints.values())
    all_ep_ok = min_ep >= 1.45
    print(f"\nMin ep endpoint J2: {min_ep:.4f}  {'PASS' if all_ep_ok else 'FAIL — some episodes < 1.45'}")

    # 4. STATE SENSITIVITY TEST
    print("\n" + "=" * 70)
    print("STATE SENSITIVITY TEST")
    print("=" * 70)

    anchor_img = images[0:1]
    anchor_state = states[0:1].clone()
    with torch.no_grad():
        base_pred_norm = model(anchor_img, anchor_state).squeeze(0).cpu().numpy()
    base_pred = base_pred_norm * action_std + action_mean

    print(f"Anchor state (norm): {anchor_state.cpu().numpy().flatten().round(4)}")
    print(f"Base pred J2: {base_pred[1]:.6f}\n")

    tests = {"J2 +0.01": (1, 0.01), "J2 +0.10": (1, 0.10),
             "J2 +0.50": (1, 0.50), "J2 +1.00": (1, 1.00),
             "J1 +0.50": (0, 0.50), "J3 +0.50": (2, 0.50)}

    all_sensitive = True
    for label, (dim, delta) in tests.items():
        mod_state = anchor_state.clone()
        mod_state[0, dim] += delta  # delta in normalized space
        with torch.no_grad():
            mod_pred_norm = model(anchor_img, mod_state).squeeze(0).cpu().numpy()
        mod_pred = mod_pred_norm * action_std + action_mean
        d_pred = mod_pred - base_pred
        d_j2 = d_pred[1]
        flag = "OK" if abs(d_j2) > 0.001 else "FAIL"
        if abs(d_j2) <= 0.001:
            all_sensitive = False
        print(f"  {label:>12}:  Δpred_J2={d_j2:+8.6f}  |Δpred|={np.linalg.norm(d_pred):.6f}  [{flag}]")

    # 5. IMAGE SENSITIVITY TEST (same state, different images)
    print("\n" + "=" * 70)
    print("IMAGE SENSITIVITY TEST (same state, different images)")
    print("=" * 70)

    anchor_state_t = states[0:1]
    img_sensitive = True
    prev_pred = None
    for ep_test in episode_list[:5]:
        ep_mask = ep_indices == ep_test
        if not ep_mask.any():
            continue
        ep_first_idx = np.where(ep_mask)[0][0]
        ep_img = images[ep_first_idx:ep_first_idx + 1]
        with torch.no_grad():
            ep_pred_norm = model(ep_img, anchor_state_t).squeeze(0).cpu().numpy()
        ep_pred = ep_pred_norm * action_std + action_mean
        if prev_pred is not None:
            d = np.linalg.norm(ep_pred - prev_pred)
            print(f"  ep{ep_test} vs previous:  Δpred_J2={ep_pred[1]-prev_pred[1]:+.6f}  |Δpred|={d:.6f}")
            if d < 0.0005:
                img_sensitive = False
        else:
            print(f"  ep{ep_test} (anchor):  pred_J2={ep_pred[1]:+.6f}  pred={np.round(ep_pred, 4)}")
        prev_pred = ep_pred

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
        pr = pn * action_std + action_mean
        mask_preds[label] = pr
        print(f"  {label:>8}: pred_J2={pr[1]:+.6f}  pred={np.round(pr, 4)}")

    d_rb = np.linalg.norm(mask_preds["real"] - mask_preds["black"])
    d_rn = np.linalg.norm(mask_preds["real"] - mask_preds["noise"])
    print(f"  |real-black|={d_rb:.6f}  |real-noise|={d_rn:.6f}")
    mask_ok = max(d_rb, d_rn) > 0.001
    print(f"  {'PASS' if mask_ok else 'FAIL'}: image mask test")

    # 7. Feature balance test
    print("\n" + "=" * 70)
    print("FEATURE BALANCE")
    print("=" * 70)

    # Capture features
    if_stats = {}
    sf_stats = {}

    def hook_if(name):
        def fn(m, inp, out):
            if_stats[name] = out.detach()
        return fn

    def hook_sf(name):
        def fn(m, inp, out):
            sf_stats[name] = out.detach()
        return fn

    # Hook after LN (LN is applied in forward, so hook on state_head and image_delta_head input)
    h_if = model.image_encoder.fc.register_forward_hook(hook_if("img_feat"))
    h_sf = model.state_mlp.net[2].register_forward_hook(hook_sf("state_feat"))

    with torch.no_grad():
        _ = model(images[:32], states[:32])

    if_l2 = float(if_stats["img_feat"].norm(dim=1).mean())
    sf_l2 = float(sf_stats["state_feat"].norm(dim=1).mean())
    print(f"  img_feat L2 (pre-LN):   {if_l2:.3f}")
    print(f"  state_feat L2 (pre-LN): {sf_l2:.3f}")
    print(f"  ratio (state/img):      {sf_l2/if_l2:.2f}x")
    if sf_l2 > if_l2 * 3:
        print("  WARN: state_feat still dominates after architecture changes")
    else:
        print("  OK: features balanced")

    h_if.remove()
    h_sf.remove()

    # 8. Cross-ep trajectory uniqueness
    print("\n" + "=" * 70)
    print("CROSS-EPISODE TRAJECTORY UNIQUENESS")
    print("=" * 70)
    ep_medians = {}
    for ep in sorted(per_ep.keys()):
        ep_data = per_ep[ep]
        ep_states_norm = (ep_data["states"] - torch.from_numpy(state_mean)) / torch.from_numpy(state_std)
        with torch.no_grad():
            ep_pred_norm = model(ep_data["images"].to(device), ep_states_norm.to(device)).cpu().numpy()
        ep_pred = ep_pred_norm * action_std + action_mean
        ep_medians[ep] = np.median(ep_pred[:, 1])
    ep_vals = np.array(list(ep_medians.values()))
    ep_spread = np.std(ep_vals)
    print(f"Cross-ep median pred_J2 std: {ep_spread:.6f}")
    print(f"  {'PASS' if ep_spread > 0.01 else 'FAIL'}")

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model params: {n_params:,}")
    print(f"img_gate final: {model.img_gate.item():.3f}  state_gate final: {model.state_gate.item():.3f}")
    print(f"[{'PASS' if overall < 0.25 else 'WARN'}] improvement_ratio (arm) = {overall:.6f}")
    print(f"[{'PASS' if all_sensitive else 'FAIL'}] state sensitivity")
    print(f"[{'PASS' if img_sensitive else 'FAIL'}] image sensitivity (cross-ep)")
    print(f"[{'PASS' if mask_ok else 'FAIL'}] image mask test")
    print(f"[{'PASS' if all_ep_ok else 'FAIL'}] all episodes J2 >= 1.45 (min={min_ep:.4f})")
    print(f"[{'PASS' if ep_spread > 0.01 else 'FAIL'}] cross-ep uniqueness (std={ep_spread:.6f})")

    passes = (overall < 0.25 and all_sensitive and img_sensitive and mask_ok
              and all_ep_ok and ep_spread > 0.01)
    if passes:
        print("\nVERDICT: HYBRID V3 PASSES — ready for real robot smoke test.")
    else:
        print("\nVERDICT: HYBRID V3 FAILED some checks — review above.")
        if not img_sensitive and not mask_ok:
            print("  Image contribution is still zero. Try: larger img_feat_dim, or reduce state_feat_dim further.")

    # Save model
    if args.output:
        print(f"\nSaving model to {args.output}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "norm_stats": norm_stats,
            "improvement_ratio": overall,
            "state_sensitive": all_sensitive,
            "img_sensitive": img_sensitive,
            "mask_ok": mask_ok,
            "cross_ep_spread": float(ep_spread),
            "min_ep_endpoint_j2": min_ep,
            "img_gate_final": model.img_gate.item(),
            "state_gate_final": model.state_gate.item(),
        }, args.output)

    # Save norm stats
    norm_path = args.output.replace(".pt", "_norm_stats.json")
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Norm stats saved to {norm_path}")


if __name__ == "__main__":
    main()
