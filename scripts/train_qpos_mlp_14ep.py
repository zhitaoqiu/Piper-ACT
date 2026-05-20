#!/usr/bin/env python3
"""
qpos-only MLP baseline: verify 14ep state->action is learnable WITHOUT vision.

If this fails: data alignment problem. Don't touch vision models.
If this succeeds: proceed to image+state hybrid.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class QPosMLP(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=7, num_layers=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_14ep_data(dataset_root: str, episodes: list, device: str = "cuda"):
    """Load observation.state and action from parquet files."""
    import pyarrow.parquet as pq

    data_dir = Path(dataset_root) / "data"
    all_states = []
    all_actions = []
    per_ep_states = {}
    per_ep_actions = {}

    for pqf in sorted(data_dir.rglob("*.parquet")):
        t = pq.read_table(str(pqf))
        ep_indices = t.column("episode_index").to_pylist()
        obs_states = t.column("observation.state").to_pylist()
        actions = t.column("action").to_pylist()

        for ep_idx, os_row, act_row in zip(ep_indices, obs_states, actions):
            if ep_idx not in episodes:
                continue
            os_arr = np.asarray(os_row, dtype=np.float32)
            act_arr = np.asarray(act_row, dtype=np.float32)
            all_states.append(os_arr)
            all_actions.append(act_arr)
            per_ep_states.setdefault(ep_idx, []).append(os_arr)
            per_ep_actions.setdefault(ep_idx, []).append(act_arr)

    all_states = np.stack(all_states)  # (N, 7)
    all_actions = np.stack(all_actions)  # (N, 7)
    for ep in per_ep_states:
        per_ep_states[ep] = np.stack(per_ep_states[ep])
        per_ep_actions[ep] = np.stack(per_ep_actions[ep])

    return all_states, all_actions, per_ep_states, per_ep_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_approach_20ep")
    parser.add_argument("--episodes", default="1,2,3,4,7,10,14,15,20,22,23,24,25,26")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    episode_list = [int(x.strip()) for x in args.episodes.split(",")]
    print(f"Loading {len(episode_list)} episodes from {args.dataset_root} ...")
    all_states, all_actions, per_ep_states, per_ep_actions = load_14ep_data(
        args.dataset_root, episode_list, str(device)
    )
    n_total = len(all_states)
    print(f"Total frames: {n_total}")

    # Train/val split: 80/20
    n_val = max(int(n_total * 0.2), 64)
    indices = np.random.permutation(n_total)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    X_train = torch.from_numpy(all_states[train_idx]).float().to(device)
    Y_train = torch.from_numpy(all_actions[train_idx]).float().to(device)
    X_val = torch.from_numpy(all_states[val_idx]).float()
    Y_val = torch.from_numpy(all_actions[val_idx]).float()

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    model = QPosMLP(input_dim=7, hidden_dim=args.hidden_dim, output_dim=7,
                    num_layers=args.num_layers).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"\nTraining {args.steps} steps, batch_size={args.batch_size}, lr={args.lr} ...")
    t0 = time.time()
    n_batches = len(X_train) // args.batch_size

    for step in range(args.steps):
        model.train()
        # Reshuffle each epoch
        if step % n_batches == 0:
            perm = torch.randperm(len(X_train), device=device)
            X_train = X_train[perm]
            Y_train = Y_train[perm]

        batch_start = (step % n_batches) * args.batch_size
        xb = X_train[batch_start:batch_start + args.batch_size]
        yb = Y_train[batch_start:batch_start + args.batch_size]

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(device)).cpu()
                val_loss = loss_fn(val_pred, Y_val).item()
                train_loss = loss.item()
            print(f"  step {step+1:5d}/{args.steps}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    # ── EVALUATION ──
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        all_preds = model(torch.from_numpy(all_states).float().to(device)).cpu().numpy()

    dim_names = ["J1", "J2", "J3", "J4", "J5", "J6", "grip"]

    # 1. Per-dim pred_std vs true_std
    print(f"\n{'Dim':>6}  {'pred_mean':>10}  {'pred_std':>10}  {'true_mean':>10}  {'true_std':>10}  {'status'}")
    for d, name in enumerate(dim_names):
        ps = all_preds[:, d]
        ts = all_actions[:, d]
        ok = "COLLAPSED" if np.std(ps) < 0.0001 else "OK"
        print(f"{name:>6}  {np.mean(ps):10.6f}  {np.std(ps):10.6f}  {np.mean(ts):10.6f}  {np.std(ts):10.6f}  {ok}")

    # 2. improvement_ratio
    mse_per_dim = np.mean((all_preds - all_actions) ** 2, axis=0)
    mean_baseline = np.mean(all_actions, axis=0)
    mse_baseline = np.mean((all_actions - mean_baseline) ** 2, axis=0)
    print(f"\n{'Dim':>6}  {'model_mse':>12}  {'baseline_mse':>12}  {'ratio':>10}  {'status'}")
    for d, name in enumerate(dim_names):
        ratio = mse_per_dim[d] / mse_baseline[d] if mse_baseline[d] > 1e-10 else float("inf")
        flag = "*** COLLAPSED ***" if ratio > 0.9 else ""
        print(f"{name:>6}  {mse_per_dim[d]:12.6f}  {mse_baseline[d]:12.6f}  {ratio:10.6f}  {flag}")
    overall = np.mean(mse_per_dim[:6]) / np.mean(mse_baseline[:6])
    print(f"\nArm-only improvement_ratio: {overall:.6f}")

    # 3. Per-ep pred_J2 curves
    print("\n--- Per-Ep pred J2 ---")
    for ep in sorted(per_ep_states.keys()):
        s = torch.from_numpy(per_ep_states[ep]).float().to(device)
        with torch.no_grad():
            ep_pred = model(s).cpu().numpy()
        pj2 = ep_pred[:, 1]
        tj2 = per_ep_actions[ep][:, 1]
        print(f"Ep{ep:2d}: pred_J2=[{pj2[0]:.4f}→{pj2[-1]:.4f}] std={np.std(pj2):.4f}  "
              f"true_J2=[{tj2[0]:.4f}→{tj2[-1]:.4f}]")

    # 4. STATE SENSITIVITY TEST
    print("\n" + "=" * 70)
    print("STATE SENSITIVITY TEST")
    print("=" * 70)

    # Pick first frame as anchor
    anchor_state = all_states[0].copy()
    anchor_t = torch.from_numpy(anchor_state).float().unsqueeze(0).to(device)

    with torch.no_grad():
        base_pred = model(anchor_t).squeeze(0).cpu().numpy()

    print(f"Anchor state: {anchor_state}")
    print(f"Base pred:    {np.round(base_pred, 6)}")
    print(f"Base pred J2: {base_pred[1]:.6f}\n")

    tests = {
        "J2 +0.01": (1, 0.01),
        "J2 +0.10": (1, 0.10),
        "J2 +0.50": (1, 0.50),
        "J2 +1.00": (1, 1.00),
        "J1 +0.50": (0, 0.50),
        "J3 +0.50": (2, 0.50),
        "J4 +0.50": (3, 0.50),
        "J5 +0.50": (4, 0.50),
        "J6 +0.50": (5, 0.50),
        "grip +0.05": (6, 0.05),
    }

    all_sensitive = True
    for label, (dim, delta) in tests.items():
        mod_state = anchor_state.copy()
        mod_state[dim] += delta
        mod_t = torch.from_numpy(mod_state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            mod_pred = model(mod_t).squeeze(0).cpu().numpy()
        d_pred = mod_pred - base_pred
        d_j2 = d_pred[1]
        flag = "OK" if abs(d_j2) > 0.001 else "FAIL"
        if abs(d_j2) <= 0.001:
            all_sensitive = False
        print(f"  {label:>12}:  Δpred_J2={d_j2:+8.6f}  |Δpred|={np.linalg.norm(d_pred):.6f}  [{flag}]")

    # 5. Check: different episodes have different trajectories
    print("\n" + "=" * 70)
    print("CROSS-EPISODE TRAJECTORY UNIQUENESS")
    print("=" * 70)
    ep_medians = {}
    for ep in sorted(per_ep_states.keys()):
        s = torch.from_numpy(per_ep_states[ep]).float().to(device)
        with torch.no_grad():
            ep_pred = model(s).cpu().numpy()
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
    print(f"[{'PASS' if overall < 0.25 else 'WARN'}] improvement_ratio (arm) = {overall:.6f} (target < 0.25)")
    print(f"[{'PASS' if all_sensitive else 'FAIL'}] state sensitivity (J2+0.5 must change pred)")
    print(f"[{'PASS' if ep_spread > 0.01 else 'FAIL'}] cross-ep uniqueness (no average collapse)")

    if overall < 0.25 and all_sensitive and ep_spread > 0.01:
        print("\nVERDICT: MLP PASSES — state->action is learnable from 14ep data.")
        print("Proceed to Step 2: image+state hybrid policy.")
    else:
        print("\nVERDICT: MLP FAILED — check data quality, state-action alignment, or feature engineering.")
        print("Do NOT proceed to vision models until this passes.")


if __name__ == "__main__":
    main()
