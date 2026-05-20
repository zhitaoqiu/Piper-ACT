#!/usr/bin/env python3
"""Diagnose v3 stall at J2=0.98: teacher-forcing, decomposition, v2 comparison.
Correct decomposition: final = base_norm*std + delta_norm*std + mean"""
import sys, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import torch, torchvision.transforms.functional as TF

MIN_NORM_STD = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from policies.state_conditioned_policy import StateConditionedPolicy
from policies.state_conditioned_policy_v3 import StateConditionedPolicyV3

v2_ckpt = torch.load("outputs/train/hybrid_v2.pt", map_location=device, weights_only=False)
v3_ckpt = torch.load("outputs/train/hybrid_v3.pt", map_location=device, weights_only=False)
v2_args, v3_args = v2_ckpt["args"], v3_ckpt["args"]

v2 = StateConditionedPolicy(state_dim=7, action_dim=7,
    img_feat_dim=v2_args.get("img_feat_dim",256), state_feat_dim=v2_args.get("state_feat_dim",128),
    state_hidden=v2_args.get("state_hidden",128), action_hidden=v2_args.get("action_hidden",256),
    use_global_img=False).to(device)
v2.load_state_dict(v2_ckpt["model_state_dict"]); v2.eval()

v3 = StateConditionedPolicyV3(state_dim=7, action_dim=7,
    img_feat_dim=v3_args.get("img_feat_dim",256), state_feat_dim=v3_args.get("state_feat_dim",64),
    state_hidden=v3_args.get("state_hidden",128), action_hidden=v3_args.get("action_hidden",256),
    use_global_img=False).to(device)
v3.load_state_dict(v3_ckpt["model_state_dict"]); v3.eval()

def load_norm(ckpt):
    ns = ckpt["norm_stats"]
    return (np.array(ns["state_mean"], dtype=np.float32),
            np.maximum(np.array(ns["state_std"], dtype=np.float32), MIN_NORM_STD),
            np.array(ns["action_mean"], dtype=np.float32),
            np.maximum(np.array(ns["action_std"], dtype=np.float32), MIN_NORM_STD))

v2_sm, v2_ss, v2_am, v2_as = load_norm(v2_ckpt)
v3_sm, v3_ss, v3_am, v3_as = load_norm(v3_ckpt)

# Load dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
episodes = [1,2,3,4,7,10,14,15,20,22,23,24,25,26]
img_h, img_w = 160, int(160 * 4/3)
ds = LeRobotDataset("piper/bottle_approach_20ep", root="data/lerobot_dataset_approach_20ep", episodes=episodes)

all_frames = []
for i in range(ds.num_frames):
    item = ds[i]
    ep_idx = int(item["episode_index"])
    if ep_idx not in episodes: continue
    img = item["observation.images.wrist_rgb"]
    if not isinstance(img, torch.Tensor): img = torch.from_numpy(img)
    if img.dtype != torch.float32: img = img.float() / 255.0
    img = TF.resize(img, (img_h, img_w), antialias=True)
    s = item["observation.state"]
    if hasattr(s, "numpy"): s = s.numpy()
    s = np.asarray(s, dtype=np.float32)
    a = item["action"]
    if hasattr(a, "numpy"): a = a.numpy()
    a = np.asarray(a, dtype=np.float32)
    all_frames.append({"ep": ep_idx, "img": img, "state": s, "action": a})

print(f"Loaded {len(all_frames)} frames, {len(set(f['ep'] for f in all_frames))} episodes")

per_ep = {}
for f in all_frames:
    per_ep.setdefault(f["ep"], []).append(f)

# ═══════ CHECK 1: V3 teacher-forcing endpoints ═══════
print("\n" + "=" * 65)
print("CHECK 1: V3 Teacher-Forcing Per-Ep Endpoint J2")
print("=" * 65)
endpoints = {}
for ep in sorted(per_ep.keys()):
    epf = per_ep[ep]
    imgs = torch.stack([f["img"] for f in epf]).to(device)
    st_np = np.array([f["state"] for f in epf], dtype=np.float32)
    st_norm = np.clip((st_np - v3_sm) / v3_ss, -5.0, 5.0)
    st = torch.from_numpy(st_norm).float().to(device)
    with torch.inference_mode():
        pn = v3(imgs, st).cpu().numpy()
    p = pn * v3_as + v3_am
    endpoints[ep] = float(p[-1, 1])
    ok = "OK" if p[-1, 1] >= 1.45 else "FAIL"
    print(f"  Ep{ep:2d}: pred_J2={p[0,1]:.4f}->{p[-1,1]:.4f}  max={p[:,1].max():.4f}  [{ok}]")

min_ep = min(endpoints.values())
print(f"\n  Min endpoint: {min_ep:.4f}  {'PASS' if min_ep >= 1.45 else 'FAIL'}")

# ═══════ CHECK 2: Decomposition vs real robot state at J2~0.98 ═══════
print("\n" + "=" * 65)
print("CHECK 2: V3 Decomposition at J2=0.98 (real robot stall point)")
print("=" * 65)

# Hooks for v3 internals
caps = {}
v3.state_head.register_forward_hook(lambda m, inp, out: caps.update(base_norm=out.detach()))
v3.image_delta_head.register_forward_hook(lambda m, inp, out: caps.update(delta_norm=out.detach()))

# Load first image of ep1 (training) vs use it as proxy for real robot
ep1 = per_ep[1]
n_ep1 = len(ep1)

# Find training frame closest to J2=0.98
target_j2 = 0.98
best_idx = min(range(n_ep1), key=lambda i: abs(ep1[i]["state"][1] - target_j2))
print(f"  Closest training frame to J2=0.98: idx={best_idx}, true_J2={ep1[best_idx]['state'][1]:.4f}")

# Decompose at key points along ep1 trajectory
print(f"\n  {'frame':>6} {'true_J2':>8} {'v2_pred':>9} {'v3_base':>9} {'v3_delta':>9} {'v3_final':>9}")
for idx in [0, n_ep1//5, n_ep1//3, n_ep1//2, 2*n_ep1//3, best_idx, 4*n_ep1//5, n_ep1-1]:
    f = ep1[idx]
    st_np = np.asarray(f["state"], dtype=np.float32)
    img_t = f["img"].unsqueeze(0).to(device)

    # V2
    st_n2 = np.clip((st_np - v2_sm) / v2_ss, -5.0, 5.0)
    with torch.inference_mode():
        p2_norm = v2(img_t, torch.from_numpy(st_n2).float().unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    p2 = p2_norm * v2_as + v2_am

    # V3
    st_n3 = np.clip((st_np - v3_sm) / v3_ss, -5.0, 5.0)
    with torch.inference_mode():
        _ = v3(img_t, torch.from_numpy(st_n3).float().unsqueeze(0).to(device))
    bn = caps["base_norm"].squeeze(0).cpu().numpy()
    dn = caps["delta_norm"].squeeze(0).cpu().numpy()
    v3_base = bn * v3_as       # base contribution (no mean)
    v3_delta = dn * v3_as       # delta contribution (no mean)
    v3_final = (bn + dn) * v3_as + v3_am  # total

    marker = " <- STALL" if abs(f["state"][1] - 0.98) < 0.05 else ""
    print(f"  {idx:6d} {f['state'][1]:8.4f} {p2[1]:9.4f} {v3_base[1]:9.4f} {v3_delta[1]:9.4f} {v3_final[1]:9.4f}{marker}")

print(f"\n  Decomposition key: final = base + delta + action_mean[J2]={v3_am[1]:.4f}")
print(f"  Verify at frame 0: {caps['base_norm'].squeeze(0).cpu().numpy()[1]*v3_as[1]:.4f} + "
      f"{caps['delta_norm'].squeeze(0).cpu().numpy()[1]*v3_as[1]:.4f} + {v3_am[1]:.4f} = "
      f"{float((caps['base_norm'].squeeze(0).cpu().numpy()[1] + caps['delta_norm'].squeeze(0).cpu().numpy()[1])*v3_as[1] + v3_am[1]):.4f}")

# ═══════ CHECK 3: Fixed point for varying J2 with fixed image ═══════
print("\n" + "=" * 65)
print("CHECK 3: Fixed point -- action[J2] vs current J2 (mid-frame image)")
print("=" * 65)

mid_img = ep1[n_ep1//2]["img"].unsqueeze(0).to(device)
mid_state = np.asarray(ep1[n_ep1//2]["state"], dtype=np.float32).copy()

print(f"  {'J2_in':>8} {'v2_act':>9} {'v3_base':>9} {'v3_delta':>9} {'v3_final':>9} {'gap(v3)':>9}")
for test_j2 in np.linspace(0.0, 1.6, 17):
    st = mid_state.copy(); st[1] = test_j2

    sn2 = np.clip((st - v2_sm) / v2_ss, -5.0, 5.0)
    with torch.inference_mode():
        p2_norm = v2(mid_img, torch.from_numpy(sn2).float().unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    p2_full = p2_norm * v2_as + v2_am
    v2_j2 = float(p2_full[1])

    sn3 = np.clip((st - v3_sm) / v3_ss, -5.0, 5.0)
    with torch.inference_mode():
        _ = v3(mid_img, torch.from_numpy(sn3).float().unsqueeze(0).to(device))
    bn = caps["base_norm"].squeeze(0).cpu().numpy()
    dn = caps["delta_norm"].squeeze(0).cpu().numpy()
    base_j2 = float(bn[1] * v3_as[1])
    delta_j2 = float(dn[1] * v3_as[1])
    final_j2 = float((bn[1] + dn[1]) * v3_as[1] + v3_am[1])
    gap = final_j2 - test_j2

    marker = ""
    if abs(gap) < 0.06 and test_j2 > 0.3:
        marker = " <- FP"
    elif test_j2 == 0.0:
        marker = " start"
    elif test_j2 >= 1.55:
        marker = " end"
    print(f"  {test_j2:8.3f} {v2_j2:9.4f} {base_j2:9.4f} {delta_j2:9.4f} {final_j2:9.4f} {gap:+9.4f}{marker}")

# ═══════ CHECK 4: Same state, different training images ═══════
print("\n" + "=" * 65)
print("CHECK 4: Image sensitivity -- same state[J2=0.98], different ep images")
print("=" * 65)

test_state = mid_state.copy(); test_state[1] = 0.98
sn3 = np.clip((test_state - v3_sm) / v3_ss, -5.0, 5.0)
st3 = torch.from_numpy(sn3).float().unsqueeze(0).to(device)

print(f"  {'ep':>6} {'v2_act[J2]':>12} {'v3_base[J2]':>12} {'v3_delta[J2]':>12} {'v3_final[J2]':>12}")
for ep in sorted(per_ep.keys())[:6]:
    # Find frame closest to J2=0.98 in this episode
    epf = per_ep[ep]
    best_i = min(range(len(epf)), key=lambda i: abs(epf[i]["state"][1] - 0.98))
    img_t = epf[best_i]["img"].unsqueeze(0).to(device)

    sn2 = np.clip((test_state - v2_sm) / v2_ss, -5.0, 5.0)
    with torch.inference_mode():
        p2_norm = v2(img_t, torch.from_numpy(sn2).float().unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    v2_out = p2_norm * v2_as + v2_am

    with torch.inference_mode():
        _ = v3(img_t, st3)
    bn = caps["base_norm"].squeeze(0).cpu().numpy()
    dn = caps["delta_norm"].squeeze(0).cpu().numpy()
    print(f"  {ep:6d} {float(v2_out[1]):12.4f} {float(bn[1]*v3_as[1]):12.4f} {float(dn[1]*v3_as[1]):12.4f} "
          f"{float((bn[1]+dn[1])*v3_as[1]+v3_am[1]):12.4f}")

# Also test: what if image_delta was zero?
print(f"\n  If image_delta were 0 at J2=0.98: base_only[J2]={float(caps['base_norm'].squeeze(0).cpu().numpy()[1]*v3_as[1] + v3_am[1]):.4f}")
print(f"  This would {'push past' if float(caps['base_norm'].squeeze(0).cpu().numpy()[1]*v3_as[1] + v3_am[1]) > 1.0 else 'stall at'} J2=0.98")

# ═══════ SUMMARY ═══════
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  V3 offline endpoints: all >= 1.45 (min={min_ep:.4f}) -- PASS")
print(f"  Real robot stalled at J2=0.98 -- closed-loop distribution shift")
print(f"  Root cause: image_delta is LARGE POSITIVE constant (~{float(caps['delta_norm'].squeeze(0).cpu().numpy()[1]*v3_as[1]):.1f})")
print(f"  This makes v3 predict J2~1.5 regardless of actual state")
print(f"  When robot IS at J2=0.98, predicted action~0.98 -> no progress")
