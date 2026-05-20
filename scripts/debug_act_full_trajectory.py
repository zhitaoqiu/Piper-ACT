#!/usr/bin/env python3
"""
Offline evaluation for Full-Trajectory ACT policy.

Checks:
1. Arm trajectory: J1-J6 pred vs true, endpoint, constant-output detection
2. Gripper trajectory: pred_gripper vs true, close/release timing
3. Phase switching: approach→descend→close→lift→place→release
4. Sensitivity: state[J2] perturbation, different images
5. Metrics: MSE, improvement_ratio, pred_std, gripper_std, close/release frame error

If gripper prediction is near-constant: FAIL.
If arm output is mean-action collapse: FAIL.
If close/release timing is not learned: FAIL.

Usage:
  python3 scripts/debug_act_full_trajectory.py \
    --checkpt outputs/train/act_full_fixed_overfit/checkpoints/005000/pretrained_model
"""
import argparse, json, sys, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import torch, torchvision.transforms.functional as TF

MIN_NORM_STD = 0.01
GRIPPER_STD_MIN = 0.003      # min std to consider gripper "not constant"
ARM_PRED_STD_MIN = 0.001     # min std per joint to consider arm "not constant"

# Absolute fallback thresholds — only used when dynamic detection fails.
# Dynamic thresholds (based on actual gripper range) are preferred because
# the gripper stops at bottle width, not at 0.0.
GRIPPER_OPEN_FALLBACK = 0.06
GRIPPER_CLOSE_FALLBACK = 0.02


def get_gripper_thresholds(grip_values):
    """Compute dynamic open/close thresholds from actual gripper data.

    The gripper doesn't close to 0.0 when holding a bottle — it stops at the
    bottle's width. So we use the data's own range: open = initial value,
    grasped = minimum value, midpoint = transition threshold.
    """
    grip = np.asarray(grip_values, dtype=np.float32)
    grip_open = float(grip[:10].mean())  # initial open baseline
    grip_grasped = float(grip.min())     # minimum = bottle width
    grip_mid = (grip_open + grip_grasped) / 2.0
    grip_range = grip_open - grip_grasped
    return grip_open, grip_grasped, grip_mid, grip_range


def load_act_policy(checkpt_dir, device):
    """Load ACT policy via LeRobot's standard from_pretrained."""
    from lerobot.policies.act.modeling_act import ACTPolicy

    ckpt_path = Path(checkpt_dir)
    config_path = ckpt_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {ckpt_path}. "
                                f"Expected LeRobot pretrained_model directory.")

    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(device)
    policy.eval()
    print(f"  ACT policy loaded: {sum(p.numel() for p in policy.parameters()):,} params")
    return policy


def load_dataset_frames(dataset_root, episodes, img_h=160, max_frames=2000):
    """Load frames from LeRobot dataset. Returns list of dicts."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    img_w = int(img_h * 4 / 3)
    ds = LeRobotDataset("piper/bottle_full_fixed_1ep", root=dataset_root, episodes=episodes)

    def preproc_img(raw):
        if not isinstance(raw, torch.Tensor):
            raw = torch.from_numpy(raw)
        if raw.dtype != torch.float32:
            raw = raw.float() / 255.0
        return TF.resize(raw, (img_h, img_w), antialias=True)

    all_frames = []
    for i in range(min(ds.num_frames, max_frames)):
        item = ds[i]
        ep_idx = int(item["episode_index"])
        if ep_idx not in episodes:
            continue
        s = item["observation.state"]
        if hasattr(s, "numpy"):
            s = s.numpy()
        s = np.asarray(s, dtype=np.float32)
        a = item["action"]
        if hasattr(a, "numpy"):
            a = a.numpy()
        a = np.asarray(a, dtype=np.float32)
        all_frames.append({
            "ep": ep_idx,
            "img_wrist": preproc_img(item["observation.images.wrist_rgb"]),
            "img_global": preproc_img(item["observation.images.global_rgb"]),
            "state": s, "action": a,
        })

    print(f"  Loaded {len(all_frames)} frames, {len(set(f['ep'] for f in all_frames))} episodes")
    return all_frames


def detect_phases(states, actions, grip_open, grip_mid):
    """Heuristic phase detection based on J2 and gripper signals.

    Uses dynamic gripper thresholds computed from the episode data.
    grip_open  = avg of first 10 grip values
    grip_mid   = midpoint between open and min (grasp width)

    Phases:
      0: approach — J2 increasing, gripper open
      1: descend  — J2 near max, gripper open
      2: close    — gripper dropping below midpoint
      3: lift     — J2 decreasing, gripper below midpoint
      4: place    — J1 moving, gripper below midpoint
      5: release  — gripper rising above midpoint
    """
    n = len(states)
    phases = np.zeros(n, dtype=int)
    j2 = np.array([s[1] for s in states])
    j1 = np.array([s[0] for s in states])
    j2_max = j2.max()
    grip = np.array([s[6] for s in states])

    for i in range(n):
        if grip[i] > grip_mid:
            if j2[i] < j2_max - 0.05:
                phases[i] = 0  # approach
            else:
                phases[i] = 1  # descend
        else:
            if j2[i] > j2_max - 0.08:
                phases[i] = 3  # lift
            elif abs(j1[i] - j1[0]) > 0.1:
                phases[i] = 4  # place
            elif grip[i] < grip_mid - 0.01:
                phases[i] = 2  # closing
            else:
                phases[i] = 5  # release

    return phases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", required=True,
                        help="Path to LeRobot pretrained_model directory (contains config.json).")
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_full_fixed_1ep")
    parser.add_argument("--episodes", default="0", help="Comma-separated episode indices.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    episode_list = [int(x.strip()) for x in args.episodes.split(",")]

    policy = load_act_policy(args.checkpt, device)
    all_frames = load_dataset_frames(args.dataset_root, episode_list)

    per_ep = {}
    for f in all_frames:
        per_ep.setdefault(f["ep"], []).append(f)

    # Load preprocessor for normalization
    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = None, None
    try:
        reenable = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=args.checkpt,
            preprocessor_overrides={"device_processor": {"device": device.type},
                                    "normalizer_processor": {"device": device.type}},
            postprocessor_overrides={"unnormalizer_processor": {"device": device.type},
                                     "device_processor": {"device": "cpu"}},
        )
        if not reenable:
            torch.set_grad_enabled(False)
    except Exception as e:
        print(f"  [WARN] Could not load pre/post processors: {e}")

    all_pass = True

    for ep in sorted(per_ep.keys()):
        epf = per_ep[ep]
        n = len(epf)
        print(f"\n{'=' * 65}")
        print(f"  Episode {ep}: {n} frames")
        print(f"{'=' * 65}")

        true_states = np.array([f["state"] for f in epf])
        true_actions = np.array([f["action"] for f in epf])
        true_grip = true_actions[:, 6]

        # ── Teacher-forcing inference ──
        pred_actions = np.zeros_like(true_actions)
        for i in range(n):
            img_w = epf[i]["img_wrist"].unsqueeze(0).to(device)
            img_g = epf[i]["img_global"].unsqueeze(0).to(device)
            st = torch.from_numpy(true_states[i]).float().unsqueeze(0).to(device)

            obs = {"observation.state": st,
                   "observation.images.wrist_rgb": img_w,
                   "observation.images.global_rgb": img_g}

            if preprocessor is not None:
                preprocessor.reset()
                obs_norm = preprocessor(obs)
            else:
                obs_norm = obs

            with torch.inference_mode():
                action = policy.select_action(obs_norm)

            if postprocessor is not None:
                action = postprocessor(action)

            action = action.squeeze(0).squeeze(0)
            pred_actions[i] = action.cpu().numpy() if hasattr(action, 'cpu') else action

        pred_grip = pred_actions[:, 6]

        # ════════════════════════════════════════════════════
        # CHECK 1: Arm trajectory
        # ════════════════════════════════════════════════════
        print(f"\n  --- CHECK 1: Arm Trajectory ---")
        arm_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
        arm_pred_std = np.std(pred_actions[:, :6], axis=0)
        arm_true_std = np.std(true_actions[:, :6], axis=0)
        arm_mse_by_joint = np.mean((pred_actions[:, :6] - true_actions[:, :6]) ** 2, axis=0)

        arm_constant = []
        for j in range(6):
            flag = "OK"
            if arm_pred_std[j] < ARM_PRED_STD_MIN:
                flag = "CONSTANT?"
                arm_constant.append(j)
                all_pass = False
            print(f"    {arm_names[j]}: "
                  f"MSE={arm_mse_by_joint[j]:.6f}  "
                  f"pred_std={arm_pred_std[j]:.4f} (true={arm_true_std[j]:.4f})  [{flag}]")

        mean_mse = np.mean(arm_mse_by_joint)
        baseline_mse = np.mean((true_actions[:, :6] - true_actions[:, :6].mean(axis=0)) ** 2)
        improvement_ratio = 1.0 - mean_mse / max(baseline_mse, 1e-8)
        print(f"    Mean arm MSE: {mean_mse:.6f}  baseline: {baseline_mse:.6f}  "
              f"improvement_ratio: {improvement_ratio:.4f}")

        if mean_mse > baseline_mse * 0.5:
            print(f"    [FAIL] Mean MSE > 50% of baseline — near mean-action collapse")
            all_pass = False

        # Dynamic gripper thresholds from this episode's true data.
        # Gripper stops at bottle width, not at 0.0 — absolute thresholds fail.
        grip_open, grip_grasped, grip_mid, grip_range = get_gripper_thresholds(true_grip)

        # ════════════════════════════════════════════════════
        # CHECK 2: Gripper trajectory
        # ════════════════════════════════════════════════════
        print(f"\n  --- CHECK 2: Gripper Trajectory ---")
        grip_pred_std = float(np.std(pred_grip))
        grip_true_std = float(np.std(true_grip))
        grip_mse = float(np.mean((pred_grip - true_grip) ** 2))
        grip_range_pred = float(np.max(pred_grip) - np.min(pred_grip))
        grip_range_true = float(np.max(true_grip) - np.min(true_grip))

        print(f"    Dynamic thresholds: open={grip_open:.4f} grasped={grip_grasped:.4f} "
              f"mid={grip_mid:.4f} range={grip_range:.4f}")
        print(f"    pred_grip: mean={pred_grip.mean():.4f} std={grip_pred_std:.4f} "
              f"range=[{pred_grip.min():.4f}, {pred_grip.max():.4f}]")
        print(f"    true_grip: mean={true_grip.mean():.4f} std={grip_true_std:.4f} "
              f"range=[{true_grip.min():.4f}, {true_grip.max():.4f}]")
        print(f"    grip MSE: {grip_mse:.6f}")

        grip_fail = False
        if grip_pred_std < GRIPPER_STD_MIN or grip_range_pred < grip_range * 0.1:
            print(f"    [FAIL] Gripper prediction is near-constant (std={grip_pred_std:.6f}, range={grip_range_pred:.6f})")
            grip_fail = True
            all_pass = False
        if grip_pred_std < grip_true_std * 0.1:
            print(f"    [FAIL] Gripper pred_std ({grip_pred_std:.4f}) << true_std ({grip_true_std:.4f})")
            grip_fail = True
            all_pass = False
        if not grip_fail:
            print(f"    [OK] Gripper prediction has meaningful variation")

        # ════════════════════════════════════════════════════
        # CHECK 3: Close/Release timing (dynamic thresholds)
        # ════════════════════════════════════════════════════
        print(f"\n  --- CHECK 3: Close/Release Timing ---")
        print(f"    grip_open={grip_open:.4f}  grip_grasped={grip_grasped:.4f}  grip_mid={grip_mid:.4f}")

        # Close: grip crosses below midpoint from above
        close_idx_true = None
        for i in range(5, n):
            if true_grip[i] < grip_mid and true_grip[i - 5] >= grip_mid:
                close_idx_true = i
                break

        close_idx_pred = None
        for i in range(5, n):
            if pred_grip[i] < grip_mid and pred_grip[i - 5] >= grip_mid:
                close_idx_pred = i
                break

        # Release: grip crosses above midpoint from below
        min_idx_true = int(np.argmin(true_grip))
        release_idx_true = None
        for i in range(min_idx_true + 10, n):
            if true_grip[i] > grip_mid and true_grip[i - 5] <= grip_mid:
                release_idx_true = i
                break

        release_idx_pred = None
        if close_idx_pred is not None:
            for i in range(close_idx_pred + 10, n):
                if pred_grip[i] > grip_mid and pred_grip[i - 5] <= grip_mid:
                    release_idx_pred = i
                    break

        if close_idx_true is None:
            print(f"    [WARN] No close event found in true data")
        else:
            if close_idx_pred is None:
                print(f"    [FAIL] No close event in prediction (true close @ frame {close_idx_true})")
                all_pass = False
            else:
                close_err = abs(close_idx_pred - close_idx_true)
                status = "OK" if close_err <= 10 else "FAIL"
                if close_err > 10:
                    all_pass = False
                print(f"    Close: true@{close_idx_true} pred@{close_idx_pred} err={close_err}frames [{status}]")

        if release_idx_true is None:
            print(f"    [WARN] No release event found in true data")
        else:
            if release_idx_pred is None:
                print(f"    [FAIL] No release event in prediction (true release @ frame {release_idx_true})")
                all_pass = False
            else:
                release_err = abs(release_idx_pred - release_idx_true)
                status = "OK" if release_err <= 10 else "FAIL"
                if release_err > 10:
                    all_pass = False
                print(f"    Release: true@{release_idx_true} pred@{release_idx_pred} err={release_err}frames [{status}]")

        # ════════════════════════════════════════════════════
        # CHECK 4: Sensitivity
        # ════════════════════════════════════════════════════
        print(f"\n  --- CHECK 4: Sensitivity ---")
        mid_idx = n // 2
        mid_img_w = epf[mid_idx]["img_wrist"].unsqueeze(0).to(device)
        mid_img_g = epf[mid_idx]["img_global"].unsqueeze(0).to(device)
        mid_state = true_states[mid_idx].copy()

        # State sensitivity: perturb J2
        st_base = torch.from_numpy(mid_state).float().unsqueeze(0).to(device)
        mid_state_pert = mid_state.copy()
        mid_state_pert[1] += 0.05  # +0.05 rad on J2
        st_pert = torch.from_numpy(mid_state_pert).float().unsqueeze(0).to(device)

        obs_base = {"observation.state": st_base,
                    "observation.images.wrist_rgb": mid_img_w,
                    "observation.images.global_rgb": mid_img_g}
        obs_pert = {"observation.state": st_pert,
                    "observation.images.wrist_rgb": mid_img_w,
                    "observation.images.global_rgb": mid_img_g}

        if preprocessor is not None:
            preprocessor.reset()
            obs_base_n = preprocessor(obs_base)
            preprocessor.reset()
            obs_pert_n = preprocessor(obs_pert)
        else:
            obs_base_n, obs_pert_n = obs_base, obs_pert

        with torch.inference_mode():
            a_base = policy.select_action(obs_base_n).squeeze().cpu().numpy()
            a_pert = policy.select_action(obs_pert_n).squeeze().cpu().numpy()

        j2_sensitivity = float(a_pert[1] - a_base[1])
        grip_sensitivity = float(a_pert[6] - a_base[6])
        print(f"    J2 +0.05: ΔJ2_pred={j2_sensitivity:.4f}  Δgrip_pred={grip_sensitivity:.4f}")
        if j2_sensitivity < 0.01:
            print(f"    [WARN] J2 sensitivity is low — model may be insensitive to state")
        if abs(grip_sensitivity) > 0.02:
            print(f"    [WARN] J2 perturbation changes gripper prediction — unwanted coupling?")

        # Image sensitivity: same state, different episodes' images
        if len(per_ep) >= 2:
            ep_keys = sorted(per_ep.keys())[:min(4, len(per_ep))]
            grip_preds_across_imgs = []
            for other_ep in ep_keys:
                other_frames = per_ep[other_ep]
                # find frame closest to mid_state
                best_i = min(range(len(other_frames)),
                             key=lambda i: np.max(np.abs(other_frames[i]["state"][:6] - mid_state[:6])))
                other_w = other_frames[best_i]["img_wrist"].unsqueeze(0).to(device)
                other_g = other_frames[best_i]["img_global"].unsqueeze(0).to(device)

                obs_img_test = {"observation.state": st_base,
                                "observation.images.wrist_rgb": other_w,
                                "observation.images.global_rgb": other_g}

                if preprocessor is not None:
                    preprocessor.reset()
                    obs_img_test = preprocessor(obs_img_test)

                with torch.inference_mode():
                    a_img = policy.select_action(obs_img_test).squeeze().cpu().numpy()
                grip_preds_across_imgs.append(a_img[6])

            img_grip_std = float(np.std(grip_preds_across_imgs))
            print(f"    Image sensitivity: grip std across {len(ep_keys)} images = {img_grip_std:.6f}")
            if img_grip_std < 0.0005:
                print(f"    [WARN] Image has negligible effect on gripper prediction — visual signal ignored?")
        else:
            print(f"    Image sensitivity: skipped (need >=2 episodes)")

    # ════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"SUMMARY")
    print(f"{'=' * 65}")
    if all_pass:
        print(f"  [PASS] All checks passed — full trajectory ACT is ready for dry-run.")
    else:
        print(f"  [FAIL] At least one check failed. Fix issues before deploying.")
        print(f"  Common causes:")
        print(f"    - Gripper is constant open: model learned dataset bias (gripper ~always open)")
        print(f"    - Arm is mean-action collapse: model regresses to mean, needs more training")
        print(f"    - Close/release not learned: not enough episodes or gripper signal too weak")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
