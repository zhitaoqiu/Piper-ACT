#!/usr/bin/env python3
"""
Diagnose image contribution in hybrid policy.
Tests: feature stats, image sensitivity, mask test, gate analysis.
"""
import sys, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from policies.state_conditioned_policy import StateConditionedPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
ckpt = torch.load("outputs/train/hybrid_v2.pt", map_location=device, weights_only=False)
model_args = ckpt["args"]
model = StateConditionedPolicy(
    state_dim=7, action_dim=7,
    img_feat_dim=model_args.get("img_feat_dim", 256),
    state_feat_dim=model_args.get("state_feat_dim", 128),
    state_hidden=model_args.get("state_hidden", 128),
    action_hidden=model_args.get("action_hidden", 256),
    use_global_img=model_args.get("use_global_img", False),
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

ns = ckpt["norm_stats"]
MIN_NORM_STD = 0.01
sm = np.array(ns["state_mean"], dtype=np.float32)
ss = np.maximum(np.array(ns["state_std"], dtype=np.float32), MIN_NORM_STD)
am = np.array(ns["action_mean"], dtype=np.float32)
as_ = np.maximum(np.array(ns["action_std"], dtype=np.float32), MIN_NORM_STD)

img_h = model_args.get("img_size", 160)
img_w = int(img_h * 4 / 3)
print(f"Model: img_size=({img_h},{img_w}) state_feat_dim={model_args.get('state_feat_dim',128)}")

# Load dataset frames from multiple episodes
import pyarrow.parquet as pq
import av

data_dir = Path("data/lerobot_dataset_approach_20ep/data")
pq_files = sorted(data_dir.rglob("*.parquet"))

# Collect first frame from episodes 1,2,3,7,10 (different positions)
target_eps = [1, 2, 3, 7, 10]
ep_images = {}  # ep -> (img_tensor, state_raw)
ep_states = {}

for pqf in pq_files:
    t = pq.read_table(str(pqf))
    ep_indices = t.column("episode_index").to_pylist()
    obs_states = t.column("observation.state").to_pylist()

    pf = Path(str(pqf))
    chunk = pf.parent.name
    stem = pf.stem
    video_path = Path(f"data/lerobot_dataset_approach_20ep/videos/observation.images.wrist_rgb/{chunk}/{stem}.mp4")
    if not video_path.exists():
        continue

    container = av.open(str(video_path))
    stream = container.streams.video[0]

    for row_idx, ep_idx in enumerate(ep_indices):
        if ep_idx in target_eps and ep_idx not in ep_images:
            state_raw = np.asarray(obs_states[row_idx], dtype=np.float32)
            container.seek(row_idx, stream=stream)
            for frame in container.decode(stream):
                img_raw = frame.to_ndarray(format="rgb24")
                break
            img_t = torch.from_numpy(img_raw).float() / 255.0
            img_t = img_t.permute(2, 0, 1)
            img_t = TF.resize(img_t, (img_h, img_w), antialias=True)
            ep_images[ep_idx] = img_t
            ep_states[ep_idx] = state_raw
            break
    container.close()
    if len(ep_images) >= len(target_eps):
        break

print(f"Loaded images from episodes: {sorted(ep_images.keys())}")

# ── Pick a fixed state (from ep 1) ──
fixed_state_raw = ep_states[1].copy()
fixed_state_norm = np.clip((fixed_state_raw - sm) / ss, -5.0, 5.0)
fixed_state_t = torch.from_numpy(fixed_state_norm).float().unsqueeze(0).to(device)
print(f"\nFixed state (ep1, raw):  {np.round(fixed_state_raw, 4)}")
print(f"Fixed state (norm):      {np.round(fixed_state_norm, 4)}")

# ── Hook to capture intermediate features ──
captured = {}

def hook_img_feat(name):
    def fn(module, input, output):
        captured[name] = output.detach()
    return fn

def hook_state_feat(name):
    def fn(module, input, output):
        captured[name] = output.detach()
    return fn

h1 = model.image_encoder.fc.register_forward_hook(hook_img_feat("img_feat"))
h2 = model.state_mlp.net[2].register_forward_hook(hook_state_feat("state_feat"))  # after 2nd Linear+ReLU

print("\n" + "=" * 80)
print("TEST 1: IMAGE FEATURE STATS (fixed state, different images)")
print("=" * 80)

base_pred = None
for ep in sorted(ep_images.keys()):
    img_t = ep_images[ep].unsqueeze(0).to(device)
    with torch.inference_mode():
        pred_norm = model(img_t, fixed_state_t)
    pred_robot = (pred_norm.squeeze(0).cpu().numpy() * as_) + am

    if_f = captured["img_feat"].squeeze(0)
    sf_f = captured["state_feat"].squeeze(0)

    print(f"\n  Ep{ep}:")
    print(f"    img_feat:  mean={float(if_f.mean()):+.4f}  std={float(if_f.std()):.4f}  L2={float(if_f.norm()):.3f}")
    print(f"    state_feat: mean={float(sf_f.mean()):+.4f}  std={float(sf_f.std()):.4f}  L2={float(sf_f.norm()):.3f}")
    print(f"    pred_action[J2]  = {float(pred_robot[1]):+.6f}")

    if base_pred is None:
        base_pred = pred_robot
    else:
        delta = pred_robot - base_pred
        print(f"    Δpred vs Ep{sorted(ep_images.keys())[0]}:  |Δ|={np.linalg.norm(delta):.6f}  ΔJ2={delta[1]:+.6f}")

# ── Compare img_feat variation ──
all_if = []
for ep in sorted(ep_images.keys()):
    img_t = ep_images[ep].unsqueeze(0).to(device)
    with torch.inference_mode():
        _ = model(img_t, fixed_state_t)
    all_if.append(captured["img_feat"].squeeze(0).cpu())

all_if_stack = torch.stack(all_if)
pairwise_dists = []
for i in range(len(all_if_stack)):
    for j in range(i+1, len(all_if_stack)):
        d = float((all_if_stack[i] - all_if_stack[j]).norm())
        pairwise_dists.append(d)
print(f"\n  img_feat pairwise distances: mean={np.mean(pairwise_dists):.4f}  min={np.min(pairwise_dists):.4f}  max={np.max(pairwise_dists):.4f}")
if np.mean(pairwise_dists) < 0.01:
    print("  FAIL: img_feat is nearly constant across different images (encoder collapsed)")
else:
    print("  OK: img_feat varies between images")

# ── Compare state_feat L2 vs img_feat L2 ──
sf_l2 = float(captured["state_feat"].squeeze(0).norm())
if_l2_mean = float(all_if_stack.norm(dim=1).mean())
print(f"\n  state_feat L2 norm: {sf_l2:.3f}")
print(f"  img_feat L2 norm (mean): {if_l2_mean:.3f}")
if sf_l2 > if_l2_mean * 2:
    print("  WARN: state_feat dominates in magnitude (L2 ratio > 2x)")
else:
    print("  OK: img_feat and state_feat magnitudes comparable")

print("\n" + "=" * 80)
print("TEST 2: IMAGE MASK TEST (same state, different image types)")
print("=" * 80)

state_t = fixed_state_t
img_real = ep_images[1].unsqueeze(0).to(device)
img_black = torch.zeros_like(img_real)
img_noise = torch.randn_like(img_real) * 0.1 + 0.5
img_noise = torch.clamp(img_noise, 0, 1)

print(f"\n  Fixed state[J2] (norm) = {fixed_state_norm[1]:.4f}")

for label, img in [("real", img_real), ("black", img_black), ("noise", img_noise)]:
    with torch.inference_mode():
        pn = model(img, state_t)
    pr = (pn.squeeze(0).cpu().numpy() * as_) + am
    print(f"  {label:>8}: pred_action[J2]={pr[1]:+.6f}  |pred|={np.linalg.norm(pr):.4f}  {np.round(pr, 4)}")

# Check if outputs differ
with torch.inference_mode():
    p_real = model(img_real, state_t)
    p_black = model(img_black, state_t)
    p_noise = model(img_noise, state_t)
d_rb = float((p_real - p_black).norm())
d_rn = float((p_real - p_noise).norm())
d_bn = float((p_black - p_noise).norm())
print(f"\n  |pred_real - pred_black| = {d_rb:.6f}")
print(f"  |pred_real - pred_noise| = {d_rn:.6f}")
print(f"  |pred_black - pred_noise| = {d_bn:.6f}")
if max(d_rb, d_rn, d_bn) < 0.001:
    print("  FAIL: Model ignores image entirely — all image types give identical output")
else:
    print("  OK: Model uses image information")

print("\n" + "=" * 80)
print("TEST 3: PER-IMAGE PREDICTION VARIANCE (teacher-forcing)")
print("=" * 80)

# For each episode, run teacher-forcing and plot the J2 curve difference
for ep in sorted(ep_images.keys())[:3]:
    img_t = ep_images[ep].unsqueeze(0).to(device)
    with torch.inference_mode():
        pn = model(img_t, state_t)
    pr = (pn.squeeze(0).cpu().numpy() * as_) + am
    print(f"  Ep{ep}: pred_J2={pr[1]:+.6f}  (all images tested with state from ep1)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  img_feat variation: {'PASS' if np.mean(pairwise_dists) > 0.01 else 'FAIL'}")
print(f"  image sensitivity:  {'PASS' if max(d_rb, d_rn, d_bn) > 0.001 else 'FAIL'}")
print(f"  feature balance:    {'PASS' if sf_l2 <= if_l2_mean * 2 else 'FAIL — state dominant'}")

h1.remove()
h2.remove()
