#!/usr/bin/env python3
"""
Debug script: test any policy for state sensitivity, image sensitivity, and
cross-episode trajectory uniqueness.

Usage:
  # Hybrid model:
  python3 scripts/debug_policy_sensitivity.py \\
    --checkpt outputs/train/hybrid_state_cond_14ep.pt \\
    --model-type hybrid

  # ACT model (from pretrained dir):
  python3 scripts/debug_policy_sensitivity.py \\
    --checkpt outputs/train/piper_bottle_approach_tiny_20ep/checkpoints/010000/pretrained_model \\
    --model-type act
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_act_model(checkpt, device):
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
        policy_cfg=policy.config, pretrained_path=checkpt,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    def predict(obs_dict, wrist_img, global_img=None):
        # Build observation in ACT format
        obs = {}
        s = np.asarray(obs_dict["observation.state"], dtype=np.float32)
        obs["observation.state"] = torch.from_numpy(s).unsqueeze(0).to(device)
        if wrist_img is not None:
            t = torch.from_numpy(np.asarray(wrist_img, dtype=np.float32))
            t = t.permute(2, 0, 1).unsqueeze(0).to(device)
            obs["observation.images.wrist_rgb"] = t
        if global_img is not None:
            t = torch.from_numpy(np.asarray(global_img, dtype=np.float32))
            t = t.permute(2, 0, 1).unsqueeze(0).to(device)
            obs["observation.images.global_rgb"] = t
        obs_batch = preprocessor(obs)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(obs_batch)
        raw = chunk[:, 0, :] if chunk.dim() == 3 else chunk
        pred = postprocessor(raw).squeeze(0).cpu().numpy()
        return pred

    return policy, predict


def load_hybrid_model(checkpt, device):
    import torchvision.transforms.functional as TF

    MIN_NORM_STD = 0.01

    ckpt = torch.load(checkpt, map_location=device, weights_only=False)
    model_args = ckpt["args"]

    # Auto-detect v3 (has img_gate_final or state_feat_dim=64) vs v2
    is_v3 = ckpt.get("img_gate_final") is not None or model_args.get("state_feat_dim") == 64
    if is_v3:
        from policies.state_conditioned_policy_v3 import StateConditionedPolicyV3
        model = StateConditionedPolicyV3(
            state_dim=7, action_dim=7,
            img_feat_dim=model_args.get("img_feat_dim", 256),
            state_feat_dim=model_args.get("state_feat_dim", 64),
            state_hidden=model_args.get("state_hidden", 128),
            action_hidden=model_args.get("action_hidden", 256),
            use_global_img=model_args.get("use_global_img", False),
        ).to(device)
    else:
        from policies.state_conditioned_policy import StateConditionedPolicy
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

    img_size_h = model_args.get("img_size", 120)
    img_size_w = int(img_size_h * 4 / 3)

    # Load normalization stats if present
    norm_stats = ckpt.get("norm_stats", None)
    if norm_stats:
        state_mean = np.array(norm_stats["state_mean"], dtype=np.float32)
        state_std = np.maximum(np.array(norm_stats["state_std"], dtype=np.float32), MIN_NORM_STD)
        action_mean = np.array(norm_stats["action_mean"], dtype=np.float32)
        action_std = np.maximum(np.array(norm_stats["action_std"], dtype=np.float32), MIN_NORM_STD)
        vlabel = "v3" if is_v3 else "v2"
        extra = ""
        if is_v3:
            extra = f"  img_gate={ckpt.get('img_gate_final', '?'):.2f}  state_gate={ckpt.get('state_gate_final', '?'):.2f}"
        print(f"  Hybrid {vlabel} loaded with normalization (img_size={img_size_h}){extra}")
    else:
        state_mean = state_std = action_mean = action_std = None
        print(f"  Hybrid model loaded WITHOUT normalization (img_size={img_size_h})")

    def predict(obs_dict, wrist_img, global_img=None):
        s = np.asarray(obs_dict["observation.state"], dtype=np.float32)
        if state_mean is not None:
            s = np.clip((s - state_mean) / state_std, -5.0, 5.0)
        s_t = torch.from_numpy(s).float().unsqueeze(0).to(device)

        if wrist_img is not None:
            img = torch.from_numpy(np.asarray(wrist_img, dtype=np.float32))
            img = img.permute(2, 0, 1)  # (C, H, W)
            img = TF.resize(img, (img_size_h, img_size_w), antialias=True)
            img = img.unsqueeze(0).to(device)
        else:
            img = torch.zeros(1, 3, img_size_h, img_size_w, device=device)
        with torch.no_grad():
            pred = model(img, s_t).squeeze(0).cpu().numpy()

        if action_mean is not None:
            pred = pred * action_std + action_mean
        return pred

    return model, predict


def load_dataset_frames(dataset_root, episodes):
    import pyarrow.parquet as pq

    data_dir = Path(dataset_root) / "data"
    frames = []

    for pqf in sorted(data_dir.rglob("*.parquet")):
        t = pq.read_table(str(pqf))
        ep_indices = t.column("episode_index").to_pylist()
        obs_states = t.column("observation.state").to_pylist()
        actions = t.column("action").to_pylist()

        for i, (ep_idx, os_row, act_row) in enumerate(zip(ep_indices, obs_states, actions)):
            if ep_idx not in episodes:
                continue
            frames.append({
                "episode": ep_idx,
                "state": np.asarray(os_row, dtype=np.float32),
                "action": np.asarray(act_row, dtype=np.float32),
                "parquet_file": str(pqf),
                "row_idx": i,
            })

    return frames


def load_frame_image(parquet_file, row_idx, dataset_root, img_size=None):
    """Load a single image frame from the video file."""
    import av
    from pathlib import Path
    import torchvision.transforms.functional as TF

    # Map parquet file to video file
    pf = Path(parquet_file)
    chunk = pf.parent.name  # chunk-000
    stem = pf.stem  # file-000
    video_rel = f"videos/observation.images.wrist_rgb/{chunk}/{stem}.mp4"
    video_path = Path(dataset_root) / video_rel

    if not video_path.exists():
        return None

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    # Seek to specific frame
    container.seek(row_idx, stream=stream)
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        img = img.astype(np.float32) / 255.0
        if img_size is not None:
            img_t = torch.from_numpy(img).permute(2, 0, 1)
            img_t = TF.resize(img_t, img_size, antialias=True)
            img = img_t.permute(1, 2, 0).numpy()
        break
    container.close()
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", required=True)
    parser.add_argument("--model-type", choices=("act", "hybrid"), default="hybrid")
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_approach_20ep")
    parser.add_argument("--episodes", default="1,2,3,4,7,10,14,15,20,22,23,24,25,26")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=120)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpt}")

    # Load model
    if args.model_type == "act":
        policy, predict_fn = load_act_model(args.checkpt, device)
    else:
        policy, predict_fn = load_hybrid_model(args.checkpt, device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Params: {n_params:,}")

    # Load dataset frames (state + action metadata only)
    episode_list = [int(x.strip()) for x in args.episodes.split(",")]
    frames = load_dataset_frames(args.dataset_root, episode_list)
    print(f"Loaded {len(frames)} frames from {len(set(f['episode'] for f in frames))} episodes")

    h = args.img_size
    w = int(h * 4 / 3)
    img_size = (h, w)

    # Group by episode
    per_ep = {}
    for f in frames:
        per_ep.setdefault(f["episode"], []).append(f)

    # ── TEST 1: State sensitivity (same image) ──
    print("\n" + "=" * 70)
    print("TEST 1: STATE SENSITIVITY (same image, different state)")
    print("=" * 70)

    anchor = frames[0]
    anchor_img = load_frame_image(anchor["parquet_file"], anchor["row_idx"],
                                  args.dataset_root, img_size=img_size)
    if anchor_img is None:
        print("WARNING: Could not load images from video. Using zero image (CNN-only test).")
        anchor_img = np.zeros((h, w, 3), dtype=np.float32)

    anchor_state = anchor["state"].copy()
    obs_base = {"observation.state": anchor_state}

    base_pred = predict_fn(obs_base, anchor_img)
    print(f"Anchor ep={anchor['episode']} state[J2]={anchor_state[1]:.6f}")
    print(f"Base pred:  {np.round(base_pred, 6)}")
    print(f"Base pred J2: {base_pred[1]:.6f}\n")

    tests = {
        "J2 +0.01 rad": (1, 0.01),
        "J2 +0.10 rad": (1, 0.10),
        "J2 +0.50 rad": (1, 0.50),
        "J2 +1.00 rad": (1, 1.00),
        "J1 +0.50 rad": (0, 0.50),
        "J3 +0.50 rad": (2, 0.50),
    }

    all_state_sensitive = True
    for label, (dim, delta) in tests.items():
        obs_mod = {"observation.state": anchor_state.copy()}
        obs_mod["observation.state"][dim] += delta
        mod_pred = predict_fn(obs_mod, anchor_img)
        d_pred = mod_pred - base_pred
        d_j2 = float(d_pred[1])
        flag = "OK" if abs(d_j2) > 0.001 else "FAIL"
        if abs(d_j2) <= 0.001:
            all_state_sensitive = False
        print(f"  {label:>15}:  Δpred_J2={d_j2:+8.6f}  |Δpred|={np.linalg.norm(d_pred):.6f}  [{flag}]")

    # ── TEST 2: Image sensitivity (same state) ──
    print("\n" + "=" * 70)
    print("TEST 2: IMAGE SENSITIVITY (same state, different image)")
    print("=" * 70)

    base_state = anchor_state.copy()
    img_changes = []
    for ep in sorted(per_ep.keys())[:5]:
        ep_frame = per_ep[ep][0]
        ep_img = load_frame_image(ep_frame["parquet_file"], ep_frame["row_idx"],
                                  args.dataset_root, img_size=img_size)
        if ep_img is None:
            print(f"  ep{ep}: SKIP (no image)")
            continue
        obs = {"observation.state": base_state}
        ep_pred = predict_fn(obs, ep_img)
        d_pred = ep_pred - base_pred
        img_changes.append(float(np.linalg.norm(d_pred)))
        print(f"  ep{ep} image, base state:  Δpred_J2={d_pred[1]:+.6f}  |Δpred|={np.linalg.norm(d_pred):.6f}")

    if img_changes:
        max_img_change = max(img_changes)
        print(f"  Max |Δpred| from image change: {max_img_change:.6f}")
        if max_img_change < 0.001:
            print("  WARN: Image has negligible effect on prediction (state-dominant)")
        else:
            print("  OK: Image affects prediction")

    # ── TEST 3: Per-ep teacher-forcing curves ──
    print("\n" + "=" * 70)
    print("TEST 3: PER-EPISODE TEACHER-FORCING J2 CURVES")
    print("=" * 70)

    per_ep_medians = {}
    per_ep_first_pred_j2 = {}

    for ep in sorted(per_ep.keys()):
        ep_frames = per_ep[ep]
        # Use first, middle, last frame for quick check
        sample_indices = [0, len(ep_frames)//2, len(ep_frames)-1]
        pred_j2_vals = []
        true_j2_vals = []
        for idx in sample_indices:
            f = ep_frames[idx]
            img = load_frame_image(f["parquet_file"], f["row_idx"],
                                   args.dataset_root, img_size=img_size)
            if img is None:
                continue
            obs = {"observation.state": f["state"].copy()}
            pred = predict_fn(obs, img)
            pred_j2_vals.append(float(pred[1]))
            true_j2_vals.append(float(f["action"][1]))

        if pred_j2_vals:
            pred_line = ", ".join(f"{v:.4f}" for v in pred_j2_vals)
            true_line = ", ".join(f"{v:.4f}" for v in true_j2_vals)
            print(f"  Ep{ep:2d}: pred_J2=[{pred_line}]  true_J2=[{true_line}]")
            per_ep_medians[ep] = np.median(pred_j2_vals)
            per_ep_first_pred_j2[ep] = pred_j2_vals[0]

    # Check cross-ep uniqueness
    if per_ep_medians:
        vals = np.array(list(per_ep_medians.values()))
        spread = np.std(vals)
        print(f"\nCross-ep median pred_J2 std: {spread:.6f}")
        if spread < 0.01:
            print("FAIL: All episodes output nearly identical J2 (average collapse)")
        else:
            print("OK: Different episodes have different J2 predictions")

    # Check first-frame uniqueness (all first frames should be different)
    if per_ep_first_pred_j2:
        first_vals = np.array(list(per_ep_first_pred_j2.values()))
        first_spread = np.std(first_vals)
        print(f"Cross-ep first-frame pred_J2 std: {first_spread:.6f}")
        if first_spread < 0.001:
            print("FAIL: All episodes start with identical first-frame prediction (image ignored)")
        else:
            print("OK: Different start images produce different first-frame predictions")

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"[{'PASS' if all_state_sensitive else 'FAIL'}] State sensitivity (J2+0.5 must change pred_J2)")
    print(f"[{'PASS' if first_spread > 0.001 else 'FAIL'}] Cross-ep first-frame uniqueness (different images → different preds)")
    print(f"[{'PASS' if spread > 0.01 else 'FAIL'}] Cross-ep trajectory uniqueness (no average collapse)")

    deploy_ready = all_state_sensitive and first_spread > 0.001 and spread > 0.01
    if deploy_ready:
        print("\nALL TESTS PASSED — model is ready for real robot smoke test.")
    else:
        print("\nSOME TESTS FAILED — do NOT deploy on real robot.")


if __name__ == "__main__":
    main()
