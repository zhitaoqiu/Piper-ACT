#!/usr/bin/env python3
"""
Verify deploy.py inference matches offline eval exactly.
Compares the deploy pipeline (normalize → model → denormalize) against
a known-good offline prediction.
"""
import sys, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision.transforms.functional as TF
from policies.state_conditioned_policy import StateConditionedPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load checkpoint exactly as deploy.py does ──
ckpt = torch.load("outputs/train/hybrid_v2.pt", map_location=device, weights_only=False)
model_args = ckpt["args"]
policy = StateConditionedPolicy(
    state_dim=7, action_dim=7,
    img_feat_dim=model_args.get("img_feat_dim", 256),
    state_feat_dim=model_args.get("state_feat_dim", 128),
    state_hidden=model_args.get("state_hidden", 128),
    action_hidden=model_args.get("action_hidden", 256),
    use_global_img=model_args.get("use_global_img", False),
).to(device)
policy.load_state_dict(ckpt["model_state_dict"])
policy.eval()

MIN_NORM_STD = 0.01  # must match deploy.py
norm_stats = ckpt["norm_stats"]
state_mean = np.array(norm_stats["state_mean"], dtype=np.float32)
state_std = np.maximum(np.array(norm_stats["state_std"], dtype=np.float32), MIN_NORM_STD)
action_mean = np.array(norm_stats["action_mean"], dtype=np.float32)
action_std = np.maximum(np.array(norm_stats["action_std"], dtype=np.float32), MIN_NORM_STD)

img_h = model_args.get("img_size", 160)
img_w = int(img_h * 4 / 3)
print(f"Image resize: ({img_h}, {img_w})")
print(f"State mean:  {np.round(state_mean, 4)}")
print(f"State std:   {np.round(state_std, 4)}")
print(f"Action mean: {np.round(action_mean, 4)}")
print(f"Action std:  {np.round(action_std, 4)}")

# ── Load a sample dataset frame (episode 1, first frame) ──
import pyarrow.parquet as pq
import av

data_dir = Path("data/lerobot_dataset_approach_20ep/data")
pq_files = sorted(data_dir.rglob("*.parquet"))
t = pq.read_table(str(pq_files[0]))
ep_indices = t.column("episode_index").to_pylist()
obs_states = t.column("observation.state").to_pylist()
actions = t.column("action").to_pylist()

# Find first frame of episode 1
for i, ep_idx in enumerate(ep_indices):
    if ep_idx == 1:
        state_raw = np.asarray(obs_states[i], dtype=np.float32)
        true_action = np.asarray(actions[i], dtype=np.float32)
        row_idx = i
        pq_file = str(pq_files[0])
        break

print(f"\nEpisode 1, frame 0, row_idx={row_idx}")
print(f"Raw state:      {np.round(state_raw, 4)}")
print(f"True action:    {np.round(true_action, 4)}")

# Load image
pf = Path(pq_file)
chunk = pf.parent.name
stem = pf.stem
video_rel = f"videos/observation.images.wrist_rgb/{chunk}/{stem}.mp4"
video_path = Path("data/lerobot_dataset_approach_20ep") / video_rel

container = av.open(str(video_path))
stream = container.streams.video[0]
container.seek(row_idx, stream=stream)
for frame in container.decode(stream):
    img_raw = frame.to_ndarray(format="rgb24")
    break
container.close()
print(f"Image shape: {img_raw.shape}, dtype={img_raw.dtype}, range=[{img_raw.min()}, {img_raw.max()}]")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: deploy.py inference path (normalize → model → denormalize)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: deploy.py inference path (normalize -> model -> denormalize)")
print("=" * 70)

img_t = torch.from_numpy(img_raw).float() / 255.0
img_t = img_t.permute(2, 0, 1)
img_t = TF.resize(img_t, (img_h, img_w), antialias=True)
img_t = img_t.to(device)

state_norm = np.clip((state_raw - state_mean) / state_std, -5.0, 5.0)
state_t = torch.from_numpy(state_norm).float().to(device)

with torch.inference_mode():
    pred_norm = policy(img_t.unsqueeze(0), state_t.unsqueeze(0))
pred_norm_np = pred_norm.squeeze(0).cpu().numpy()
pred_robot = pred_norm_np * action_std + action_mean

print(f"  state_norm            = {np.round(state_norm, 4)}")
print(f"  state_norm[J2]        = {state_norm[1]:.6f}")
print(f"  model_output_norm     = {np.round(pred_norm_np, 6)}")
print(f"  model_output_norm[J2] = {pred_norm_np[1]:.6f}")
print(f"  action_after_unnorm   = {np.round(pred_robot, 6)}")
print(f"  action_after_unnorm[J2] = {pred_robot[1]:.6f}")
print(f"  true_action           = {np.round(true_action, 6)}")
print(f"  true_action[J2]       = {true_action[1]:.6f}")
err = pred_robot - true_action
print(f"  prediction_error      = {np.round(err, 6)}")
print(f"  |pred_error|          = {np.linalg.norm(err):.6f}")

# Sanity checks
j2_ok = -0.1 <= pred_robot[1] <= 1.8
print(f"\n  Sanity: action_after_unnorm[J2]={pred_robot[1]:.4f} in [-0.1, 1.8]? {'YES' if j2_ok else 'NO — WOULD HALT ON ROBOT'}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: NO normalization (raw state → model → raw output)
# This is what debug_policy_sensitivity.py does — it's WRONG for normalized model
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: NO normalization (raw state -> model -> raw output)")
print("This is what debug_policy_sensitivity.py currently does")
print("=" * 70)

with torch.inference_mode():
    pred_raw = policy(img_t.unsqueeze(0), torch.from_numpy(state_raw).float().unsqueeze(0).to(device))
pred_raw_np = pred_raw.squeeze(0).cpu().numpy()
print(f"  raw_state[J2] = {state_raw[1]:.6f}")
print(f"  pred_no_norm  = {np.round(pred_raw_np, 6)}")
print(f"  pred_no_norm[J2] = {pred_raw_np[1]:.6f}")
print(f"  (Model was trained with norm, so raw-state input gives garbage)")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Simulate robot starting position (J2 ≈ 0.0)
# The robot always starts from a neutral pose with J2 around 0
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: Simulate real robot start pose (J2 ~ 0.0)")
print("=" * 70)

robot_start_state = state_raw.copy()
robot_start_state[1] = 0.0  # J2 starts at ~0 on real robot
robot_start_state[6] = 0.08  # gripper open

state_norm_robot = np.clip((robot_start_state - state_mean) / state_std, -5.0, 5.0)
state_t_robot = torch.from_numpy(state_norm_robot).float().to(device)

with torch.inference_mode():
    pred_norm_robot = policy(img_t.unsqueeze(0), state_t_robot.unsqueeze(0))
pred_norm_robot_np = pred_norm_robot.squeeze(0).cpu().numpy()
pred_robot_action = pred_norm_robot_np * action_std + action_mean

print(f"  robot_start_state     = {np.round(robot_start_state, 4)}")
print(f"  state_norm[J2]        = {state_norm_robot[1]:.6f}  (training range roughly [-0.9, 0.9])")
print(f"  model_output_norm[J2] = {pred_norm_robot_np[1]:.6f}")
print(f"  action_after_unnorm[J2] = {pred_robot_action[1]:.6f}")
print(f"  action_after_unnorm   = {np.round(pred_robot_action, 6)}")

j2_ok_robot = -0.1 <= pred_robot_action[1] <= 1.8
norm_in_range = -3.0 <= pred_norm_robot_np[1] <= 3.0
print(f"\n  Sanity: action[J2]={pred_robot_action[1]:.4f} in [-0.1, 1.8]? {'YES' if j2_ok_robot else 'NO — WOULD HALT ON ROBOT'}")
print(f"  Norm output in reasonable range [-3,3]? {'YES' if norm_in_range else 'NO — model extrapolating wildly'}")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if j2_ok and j2_ok_robot:
    print("ALL CHECKS PASSED — deploy.py normalization path is correct.")
    print("Ready for real robot smoke test.")
else:
    print("ISSUES FOUND:")
    if not j2_ok:
        print("  - Training frame prediction is out of bounds (normalization bug!)")
    if not j2_ok_robot:
        print("  - Robot start pose gives out-of-bounds prediction")
        print("  - Model cannot handle J2=0.0 (out of training distribution)")
        print("  - Need to either: use a warm-up phase to get J2 into training range,")
        print("    or retrain with data that includes the starting pose")
