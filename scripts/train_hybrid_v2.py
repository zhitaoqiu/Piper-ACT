#!/usr/bin/env python3
"""
Train Hybrid v2: explicit state-conditioned policy WITH normalization.

v2 changes from v1:
- state/action normalization (zero mean, unit variance per dim)
- norm stats saved in checkpoint for deployment
- larger image size (default 160 vs 120)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policies.state_conditioned_policy import StateConditionedPolicy


def compute_norm_stats(dataset_root: str, episodes: list):
    """Compute per-dim mean/std from parquet data (state and action)."""
    import pyarrow.parquet as pq

    data_dir = Path(dataset_root) / "data"
    ep_set = set(episodes)
    all_states = []
    all_actions = []

    for pqf in sorted(data_dir.rglob("*.parquet")):
        t = pq.read_table(str(pqf))
        eps = t.column("episode_index").to_pylist()
        states = t.column("observation.state").to_pylist()
        actions = t.column("action").to_pylist()
        for ep, s, a in zip(eps, states, actions):
            if ep in ep_set:
                all_states.append(np.asarray(s, dtype=np.float32))
                all_actions.append(np.asarray(a, dtype=np.float32))

    all_states = np.stack(all_states)
    all_actions = np.stack(all_actions)

    return {
        "state_mean": all_states.mean(axis=0).tolist(),
        "state_std": np.clip(all_states.std(axis=0), 1e-6, None).tolist(),
        "action_mean": all_actions.mean(axis=0).tolist(),
        "action_std": np.clip(all_actions.std(axis=0), 1e-6, None).tolist(),
        "n_frames": len(all_states),
    }


def load_dataset_cached(dataset_root: str, episodes: list, img_size=(160, 213),
                        norm_stats=None):
    """Load all (image, state, action) tuples, with optional normalization."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torchvision.transforms.functional as TF

    ds = LeRobotDataset("piper/bottle_approach_20ep", root=dataset_root, episodes=episodes)
    print(f"Dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")

    all_images = []
    all_states = []
    all_actions = []
    all_ep_indices = []
    ep_to_frames = {}

    # Normalization arrays
    state_mean = np.array(norm_stats["state_mean"], dtype=np.float32) if norm_stats else None
    state_std = np.array(norm_stats["state_std"], dtype=np.float32) if norm_stats else None
    action_mean = np.array(norm_stats["action_mean"], dtype=np.float32) if norm_stats else None
    action_std = np.array(norm_stats["action_std"], dtype=np.float32) if norm_stats else None

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
        if norm_stats:
            s = (s - state_mean) / state_std

        a = item["action"]
        if hasattr(a, "numpy"):
            a = a.numpy()
        a = np.asarray(a, dtype=np.float32)
        if norm_stats:
            a = (a - action_mean) / action_std

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

    per_ep = {}
    for ep, frame_ids in ep_to_frames.items():
        per_ep[ep] = {
            "images": images[frame_ids],
            "states": states[frame_ids],
            "actions": actions[frame_ids],
            "raw_actions": None,  # We'll populate later if needed
        }

    print(f"Images: {images.shape}, States: {states.shape}, Actions: {actions.shape}")
    return images, states, actions, np.array(all_ep_indices), per_ep, episodes, norm_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_approach_20ep")
    parser.add_argument("--episodes", default="1,2,3,4,7,10,14,15,20,22,23,24,25,26")
    parser.add_argument("--img-feat-dim", type=int, default=256)
    parser.add_argument("--state-feat-dim", type=int, default=128)
    parser.add_argument("--state-hidden", type=int, default=128)
    parser.add_argument("--action-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--norm-stats", default=None,
                        help="Path to JSON stats file. If not provided, compute from data.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/train/hybrid_v2.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    episode_list = [int(x.strip()) for x in args.episodes.split(",")]

    h = args.img_size
    w = int(h * 4 / 3)
    img_size = (h, w)
    print(f"Image resize: {img_size}")

    # Load or compute norm stats
    if args.norm_stats and Path(args.norm_stats).exists():
        with open(args.norm_stats) as f:
            norm_stats = json.load(f)
        print(f"Loaded norm stats from {args.norm_stats}")
    else:
        print("Computing norm stats from dataset...")
        norm_stats = compute_norm_stats(args.dataset_root, episode_list)
        print(f"  state_mean: {np.round(norm_stats['state_mean'], 4)}")
        print(f"  state_std:  {np.round(norm_stats['state_std'], 4)}")
        print(f"  action_mean: {np.round(norm_stats['action_mean'], 4)}")
        print(f"  action_std:  {np.round(norm_stats['action_std'], 4)}")

    # Load data
    images_raw, states_raw, actions_raw, ep_indices, per_ep_raw, eps_loaded, _ = load_dataset_cached(
        args.dataset_root, episode_list, img_size=img_size, norm_stats=None
    )
    n_total = len(images_raw)

    # Create normalized versions
    state_mean_arr = np.array(norm_stats["state_mean"], dtype=np.float32)
    state_std_arr = np.array(norm_stats["state_std"], dtype=np.float32)
    action_mean_arr = np.array(norm_stats["action_mean"], dtype=np.float32)
    action_std_arr = np.array(norm_stats["action_std"], dtype=np.float32)

    states_norm = (states_raw - state_mean_arr) / state_std_arr
    actions_norm = (actions_raw - action_mean_arr) / action_std_arr

    # Build normalized per_ep
    per_ep_norm = {}
    for ep in per_ep_raw:
        per_ep_norm[ep] = {
            "images": per_ep_raw[ep]["images"],
            "states": (per_ep_raw[ep]["states"] - state_mean_arr) / state_std_arr,
            "actions": (per_ep_raw[ep]["actions"] - action_mean_arr) / action_std_arr,
        }

    # Train/val split
    n_val = max(int(n_total * 0.2), 64)
    indices = torch.randperm(n_total)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    images = images_raw.to(device)
    states = states_norm.to(device)
    actions = actions_norm.to(device)

    model = StateConditionedPolicy(
        state_dim=7, action_dim=7,
        img_feat_dim=args.img_feat_dim,
        state_feat_dim=args.state_feat_dim,
        state_hidden=args.state_hidden,
        action_hidden=args.action_hidden,
        use_global_img=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"\nTraining {args.steps} steps, batch_size={args.batch_size}, lr={args.lr} ...")
    t0 = time.time()
    n_train = len(train_idx)

    val_img = images[val_idx]
    val_state = states[val_idx]
    val_act = actions[val_idx]

    n_batch = max(1, n_train // args.batch_size)
    train_idx_shuf = train_idx.clone()

    for step in range(args.steps):
        model.train()
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

        if step == 0 or (step + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_img, val_state)
                val_loss = loss_fn(val_pred, val_act).item()
                train_loss = loss.item()
            print(f"  step {step+1:5d}/{args.steps}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    # ── EVALUATION (denormalize predictions to raw space) ──
    print("\n" + "=" * 70)
    print("EVALUATION (raw/unnormalized space)")
    print("=" * 70)

    model.eval()
    all_preds_norm = []
    bs_eval = 64
    for i in range(0, n_total, bs_eval):
        batch_img = images_raw[i:i + bs_eval].to(device)
        batch_state = states_norm[i:i + bs_eval].to(device)
        with torch.no_grad():
            bp = model(batch_img, batch_state)
        all_preds_norm.append(bp.cpu().numpy())
    all_preds_norm = np.concatenate(all_preds_norm)
    # Denormalize
    all_preds = all_preds_norm * action_std_arr + action_mean_arr
    all_acts_np = actions_raw.cpu().numpy()

    dim_names = ["J1", "J2", "J3", "J4", "J5", "J6", "grip"]

    print(f"\n{'Dim':>6}  {'pred_mean':>10}  {'pred_std':>10}  {'true_mean':>10}  {'true_std':>10}  {'status'}")
    for d, name in enumerate(dim_names):
        ps = all_preds[:, d]
        ts = all_acts_np[:, d]
        ok = "COLLAPSED" if np.std(ps) < 0.0001 else "OK"
        print(f"{name:>6}  {np.mean(ps):10.6f}  {np.std(ps):10.6f}  {np.mean(ts):10.6f}  {np.std(ts):10.6f}  {ok}")

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

    # Per-ep pred_J2 (denormalized)
    print("\n--- Per-Ep pred J2 (raw space) ---")
    ep_ends = {}
    for ep in sorted(per_ep_norm.keys()):
        ep_data_n = per_ep_norm[ep]
        with torch.no_grad():
            ep_pred_norm = model(ep_data_n["images"].to(device),
                                 ep_data_n["states"].to(device)).cpu().numpy()
        ep_pred = ep_pred_norm * action_std_arr + action_mean_arr
        pj2 = ep_pred[:, 1]
        tj2 = per_ep_raw[ep]["actions"].numpy()[:, 1]
        ep_ends[ep] = float(pj2[-1])
        print(f"Ep{ep:2d}: pred_J2=[{pj2[0]:.4f}→{pj2[-1]:.4f}] std={np.std(pj2):.4f}  "
              f"true_J2=[{tj2[0]:.4f}→{tj2[-1]:.4f}]  "
              f"end_ok={'YES' if pj2[-1] > 1.4 else 'NO'}")
    ends = np.array(list(ep_ends.values()))
    print(f"\nPer-ep pred_J2 endpoint: min={ends.min():.4f}  max={ends.max():.4f}  mean={ends.mean():.4f}")
    print(f"All episodes reach J2 >= 1.5: {'YES' if ends.min() > 1.5 else 'NO (min=' + str(ends.min()) + ')'}")

    # ── STATE SENSITIVITY ──
    print("\n" + "=" * 70)
    print("STATE SENSITIVITY (normalized input space)")
    print("=" * 70)

    # Pick first frame, normalize
    anchor_state_raw = states_raw[0:1].numpy().flatten()
    anchor_state_norm = (anchor_state_raw - state_mean_arr) / state_std_arr
    anchor_img = images_raw[0:1].to(device)

    anchor_s_t = torch.from_numpy(anchor_state_norm).float().unsqueeze(0).to(device)
    with torch.no_grad():
        base_pred_norm = model(anchor_img, anchor_s_t).squeeze(0).cpu().numpy()
    base_pred = base_pred_norm * action_std_arr + action_mean_arr
    print(f"Anchor state (raw):    {np.round(anchor_state_raw, 4)}")
    print(f"Anchor state (norm):   {np.round(anchor_state_norm, 4)}")
    print(f"Base pred (raw):       {np.round(base_pred, 6)}")
    print(f"Base pred J2: {base_pred[1]:.6f}\n")

    tests = {
        "J2 +0.01 raw": (1, 0.01),
        "J2 +0.10 raw": (1, 0.10),
        "J2 +0.50 raw": (1, 0.50),
        "J2 +1.00 raw": (1, 1.00),
    }

    all_sensitive = True
    for label, (dim, delta) in tests.items():
        mod_raw = anchor_state_raw.copy()
        mod_raw[dim] += delta
        mod_norm = (mod_raw - state_mean_arr) / state_std_arr
        mod_t = torch.from_numpy(mod_norm).float().unsqueeze(0).to(device)
        with torch.no_grad():
            mod_pred_norm = model(anchor_img, mod_t).squeeze(0).cpu().numpy()
        mod_pred = mod_pred_norm * action_std_arr + action_mean_arr
        d_j2 = float(mod_pred[1] - base_pred[1])
        flag = "OK" if abs(d_j2) > 0.001 else "FAIL"
        if abs(d_j2) <= 0.001:
            all_sensitive = False
        print(f"  {label:>15}:  Δpred_J2={d_j2:+8.6f}  |Δpred|={np.linalg.norm(mod_pred - base_pred):.6f}  [{flag}]")

    # ── IMAGE SENSITIVITY ──
    print("\n" + "=" * 70)
    print("IMAGE SENSITIVITY (same normalized state, different image)")
    print("=" * 70)
    img_changes = []
    for ep in episode_list[:5]:
        ep_mask = ep_indices == ep
        if not ep_mask.any():
            continue
        ep_first_idx = np.where(ep_mask)[0][0]
        ep_img = images_raw[ep_first_idx:ep_first_idx + 1].to(device)
        with torch.no_grad():
            ep_pred_norm = model(ep_img, anchor_s_t).squeeze(0).cpu().numpy()
        ep_pred = ep_pred_norm * action_std_arr + action_mean_arr
        d_pred = ep_pred - base_pred
        img_changes.append(float(np.linalg.norm(d_pred)))
        print(f"  ep{ep} vs anchor:  Δpred_J2={d_pred[1]:+.6f}  |Δpred|={np.linalg.norm(d_pred):.6f}")

    if img_changes:
        print(f"  Max |Δpred| from image: {max(img_changes):.6f}")

    # ── CROSS-EP UNIQUENESS ──
    print("\n" + "=" * 70)
    print("CROSS-EP TRAJECTORY UNIQUENESS")
    print("=" * 70)
    ep_medians = {}
    for ep in sorted(per_ep_norm.keys()):
        ep_data_n = per_ep_norm[ep]
        with torch.no_grad():
            ep_pred_norm = model(ep_data_n["images"].to(device),
                                 ep_data_n["states"].to(device)).cpu().numpy()
        ep_pred = ep_pred_norm * action_std_arr + action_mean_arr
        ep_medians[ep] = np.median(ep_pred[:, 1])
    ep_vals = np.array(list(ep_medians.values()))
    ep_spread = np.std(ep_vals)
    print(f"Cross-ep median pred_J2 std: {ep_spread:.6f}")
    print(f"[{'PASS' if ep_spread > 0.01 else 'FAIL'}] cross-ep uniqueness")

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model params: {n_params:,}")
    print(f"[{'PASS' if overall < 0.25 else 'WARN'}] improvement_ratio (arm) = {overall:.6f}")
    print(f"[{'PASS' if all_sensitive else 'FAIL'}] state sensitivity")
    print(f"[{'PASS' if ep_spread > 0.01 else 'FAIL'}] cross-ep uniqueness")
    print(f"[{'PASS' if ends.min() > 1.4 else 'WARN'}] all ep endpoints J2 >= 1.4 (min={ends.min():.4f})")

    passes = overall < 0.25 and all_sensitive and ep_spread > 0.01
    if passes:
        print("\nVERDICT: HYBRID V2 PASSES — ready for real robot smoke test.")
    else:
        print("\nVERDICT: HYBRID V2 FAILED")

    # Save model with norm stats
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Also save norm stats alongside
    norm_path = output_path.with_name(output_path.stem + "_norm_stats.json")
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\nSaving to {args.output}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "norm_stats": norm_stats,
        "improvement_ratio": overall,
        "state_sensitive": all_sensitive,
        "cross_ep_spread": float(ep_spread),
        "min_ep_endpoint_j2": float(ends.min()),
    }, args.output)
    print(f"Norm stats saved to {norm_path}")


if __name__ == "__main__":
    main()
