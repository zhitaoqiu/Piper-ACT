#!/usr/bin/env python3
"""
Train explicit state-conditioned hybrid policy on 14ep dataset.

Architecture:
  image_encoder(wrist_img) → img_feat (256D)
  state_mlp(observation.state) → state_feat (128D)
  concat → action_mlp → action (7D)

Unlike ACT, the state vector explicitly participates — no way to ignore it.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policies.state_conditioned_policy import StateConditionedPolicy


def load_dataset_cached(dataset_root: str, episodes: list, img_size=(120, 160),
                        device="cuda"):
    """Load all (image, state, action) tuples into GPU/CPU memory in one pass."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torchvision.transforms.functional as TF

    ds = LeRobotDataset("piper/bottle_approach_20ep", root=dataset_root, episodes=episodes)
    print(f"Dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")

    all_images = []
    all_states = []
    all_actions = []
    all_ep_indices = []
    ep_to_frames = {}

    print("Loading frames into memory...")
    t0 = time.time()
    for i in range(ds.num_frames):
        item = ds[i]
        ep_idx = int(item["episode_index"])
        if ep_idx not in episodes:
            continue

        # Wrist image: torchvision tensor (C, H, W) float32 [0,1]
        img = item["observation.images.wrist_rgb"]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        if img.dtype != torch.float32:
            img = img.float() / 255.0
        # Resize
        img = TF.resize(img, img_size, antialias=True)

        # State
        s = item["observation.state"]
        if hasattr(s, "numpy"):
            s = s.numpy()
        s = np.asarray(s, dtype=np.float32)

        # Action
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

    # Stack into tensors
    images = torch.stack(all_images)  # (N, 3, H, W) float32
    states = torch.from_numpy(np.stack(all_states)).float()  # (N, 7)
    actions = torch.from_numpy(np.stack(all_actions)).float()  # (N, 7)
    ep_indices_arr = np.array(all_ep_indices)

    # Build per-ep mapping
    per_ep = {}
    for ep, frame_ids in ep_to_frames.items():
        per_ep[ep] = {
            "images": images[frame_ids],
            "states": states[frame_ids],
            "actions": actions[frame_ids],
        }

    print(f"Images: {images.shape}, States: {states.shape}, Actions: {actions.shape}")
    return images, states, actions, ep_indices_arr, per_ep, episodes


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
    parser.add_argument("--img-size", type=int, default=120,
                        help="Resize wrist image to (img_size, img_size*4/3)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Save model checkpoint .pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    episode_list = [int(x.strip()) for x in args.episodes.split(",")]

    h = args.img_size
    w = int(h * 4 / 3)  # 640/480 = 4/3
    img_size = (h, w)
    print(f"Image resize: {img_size}")

    # Load dataset
    images, states, actions, ep_indices, per_ep, eps_loaded = load_dataset_cached(
        args.dataset_root, episode_list, img_size=img_size
    )
    n_total = len(images)

    # Train/val split
    n_val = max(int(n_total * 0.2), 64)
    indices = torch.randperm(n_total)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    # Move to device
    images = images.to(device)
    states = states.to(device)
    actions = actions.to(device)

    # Model
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
    n_val_batch = len(val_idx)

    val_img = images[val_idx]
    val_state = states[val_idx]
    val_act = actions[val_idx]

    train_losses = []
    val_losses = []

    for step in range(args.steps):
        model.train()
        # Reshuffle each epoch
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

        if step == 0 or (step + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_img, val_state)
                val_loss = loss_fn(val_pred, val_act).item()
                train_loss = loss.item()
            train_losses.append((step + 1, train_loss))
            val_losses.append((step + 1, val_loss))
            print(f"  step {step+1:5d}/{args.steps}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    # ── EVALUATION ──
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()
    # Predict all frames in batches to avoid OOM
    all_preds = []
    bs_eval = 64
    for i in range(0, n_total, bs_eval):
        batch_img = images[i:i + bs_eval]
        batch_state = states[i:i + bs_eval]
        with torch.no_grad():
            bp = model(batch_img, batch_state)
        all_preds.append(bp.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_acts_np = actions.cpu().numpy()

    dim_names = ["J1", "J2", "J3", "J4", "J5", "J6", "grip"]

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

    # 3. Per-ep pred_J2 curves
    print("\n--- Per-Ep pred J2 ---")
    for ep in sorted(per_ep.keys()):
        ep_data = per_ep[ep]
        with torch.no_grad():
            ep_pred = model(ep_data["images"].to(device), ep_data["states"].to(device)).cpu().numpy()
        pj2 = ep_pred[:, 1]
        tj2 = ep_data["actions"].cpu().numpy()[:, 1]
        print(f"Ep{ep:2d}: pred_J2=[{pj2[0]:.4f}→{pj2[-1]:.4f}] std={np.std(pj2):.4f}  "
              f"true_J2=[{tj2[0]:.4f}→{tj2[-1]:.4f}]")

    # 4. STATE SENSITIVITY TEST
    print("\n" + "=" * 70)
    print("STATE SENSITIVITY TEST (same image, different state)")
    print("=" * 70)

    anchor_img = images[0:1]  # (1, 3, H, W)
    anchor_state = states[0:1].clone()  # (1, 7)
    anchor_state_np = anchor_state.cpu().numpy().flatten()

    with torch.no_grad():
        base_pred = model(anchor_img, anchor_state).squeeze(0).cpu().numpy()
    print(f"Anchor state: {np.round(anchor_state_np, 4)}")
    print(f"Base pred:    {np.round(base_pred, 6)}")
    print(f"Base pred J2: {base_pred[1]:.6f}\n")

    tests = {
        "J2 +0.01": (1, 0.01),
        "J2 +0.10": (1, 0.10),
        "J2 +0.50": (1, 0.50),
        "J2 +1.00": (1, 1.00),
        "J1 +0.50": (0, 0.50),
        "J3 +0.50": (2, 0.50),
    }

    all_sensitive = True
    for label, (dim, delta) in tests.items():
        mod_state = anchor_state.clone()
        mod_state[0, dim] += delta
        with torch.no_grad():
            mod_pred = model(anchor_img, mod_state).squeeze(0).cpu().numpy()
        d_pred = mod_pred - base_pred
        d_j2 = d_pred[1]
        flag = "OK" if abs(d_j2) > 0.001 else "FAIL"
        if abs(d_j2) <= 0.001:
            all_sensitive = False
        print(f"  {label:>12}:  Δpred_J2={d_j2:+8.6f}  |Δpred|={np.linalg.norm(d_pred):.6f}  [{flag}]")

    # 5. IMAGE SENSITIVITY TEST (same state, different image)
    print("\n" + "=" * 70)
    print("IMAGE SENSITIVITY TEST (same state, different image)")
    print("=" * 70)
    for ep_test in episode_list[:3]:
        # Find first frame of this episode
        ep_mask = ep_indices == ep_test
        ep_first_idx = np.where(ep_mask)[0][0]
        ep_img = images[ep_first_idx:ep_first_idx + 1]
        with torch.no_grad():
            ep_pred = model(ep_img, anchor_state).squeeze(0).cpu().numpy()
        d_pred = ep_pred - base_pred
        print(f"  ep{ep_test} vs anchor:  Δpred_J2={d_pred[1]:+.6f}  |Δpred|={np.linalg.norm(d_pred):.6f}")

    # 6. Cross-ep trajectory uniqueness
    print("\n" + "=" * 70)
    print("CROSS-EPISODE TRAJECTORY UNIQUENESS")
    print("=" * 70)
    ep_medians = {}
    for ep in sorted(per_ep.keys()):
        ep_data = per_ep[ep]
        with torch.no_grad():
            ep_pred = model(ep_data["images"].to(device), ep_data["states"].to(device)).cpu().numpy()
        ep_medians[ep] = np.median(ep_pred[:, 1])
    ep_vals = np.array(list(ep_medians.values()))
    ep_spread = np.std(ep_vals)
    print(f"Per-ep median pred_J2: {', '.join(f'Ep{e}={v:.4f}' for e, v in sorted(ep_medians.items()))}")
    print(f"Cross-ep median pred_J2 std: {ep_spread:.6f}")
    if ep_spread < 0.01:
        print("FAIL: All episodes output nearly identical trajectory (average collapse)")
    else:
        print("OK: Different episodes have different trajectories")

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model params: {n_params:,}")
    print(f"[{'PASS' if overall < 0.25 else 'WARN'}] improvement_ratio (arm) = {overall:.6f} (target < 0.25)")
    print(f"[{'PASS' if all_sensitive else 'FAIL'}] state sensitivity (same image, J2+0.5 must change pred)")
    print(f"[{'PASS' if ep_spread > 0.01 else 'FAIL'}] cross-ep uniqueness (no average collapse)")

    passes = overall < 0.25 and all_sensitive and ep_spread > 0.01
    if passes:
        print("\nVERDICT: HYBRID POLICY PASSES — ready for sensitivity debug and real robot smoke test.")
    else:
        print("\nVERDICT: HYBRID POLICY FAILED — check model design or training hyperparams.")

    # Save model
    if args.output:
        print(f"\nSaving model to {args.output}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "improvement_ratio": overall,
            "state_sensitive": all_sensitive,
            "cross_ep_spread": float(ep_spread),
        }, args.output)


if __name__ == "__main__":
    main()
