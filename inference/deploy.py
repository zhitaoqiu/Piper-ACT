#!/usr/bin/env python3
"""
Deploy trained ACT or Hybrid policy on Piper arm for bottle grasping.

Usage:
  conda activate piper_act
  # ACT policy (baseline fixed position):
  python3 inference/deploy.py \
    --checkpt outputs/baselines/baseline_v1_tiny_act_fixed_position_6of6/pretrained_model \
    --test-mode A --debug-actions --replan-every-step

  # Hybrid state-conditioned policy (multi-position):
  python3 inference/deploy.py \
    --policy-type hybrid \
    --hybrid-checkpt outputs/train/hybrid_state_cond_14ep.pt \
    --test-mode A --debug-actions --debug-policy-io

Controls:
  SPACE  = run one approach attempt
  Q/ESC  = quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hardware.piper_wrapper import PiperRobot
from hardware.config_piper import PiperRobotConfig
from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices
from policies.state_conditioned_policy import StateConditionedPolicy
from policies.state_conditioned_policy_v3 import StateConditionedPolicyV3
from policies.hybrid_delta_policy import HybridDeltaPolicy

PIPER_GRIPPER_MAX_M = 0.101

# ── Approach-phase constants ──
GRIPPER_OPEN = 0.08          # gripper fully open (m)
GRIPPER_CLOSE = 0.0          # gripper fully closed (m)
# Per-joint max delta: J1-J3 arm joints get 0.03, J4-J6 wrist get 0.012
MAX_DELTA_PER_JOINT = np.array([0.03, 0.03, 0.03, 0.012, 0.012, 0.012], dtype=np.float32)
ACTION_SMOOTH_ALPHA = 0.5    # EMA smoothing factor
APPROACH_STEPS_DEFAULT = 200
WRIST_FREEZE_J2 = 1.45       # freeze J4-J6 when J2 exceeds this
READY_J2 = 1.50              # J2 threshold for ready_count
READY_COUNT_MIN = 5          # consecutive steps above READY_J2 to trigger stop
STAGNATION_STEPS = 20
STAGNATION_THRESHOLD = 0.0008  # rad — below this for N consecutive steps = stuck

# Norm safety: gripper std in training is ~0.0006 (near-constant), so a 0.02m
# gripper offset = 33σ in normalized space, causing state MLP to output garbage.
# Floor the std and clip normalized state to keep model in its operating regime.
MIN_NORM_STD = 0.01
NORM_STATE_CLIP = 5.0
UNNORM_ACTION_J2_MIN = -0.1   # halt if denormalized action[J2] below this
UNNORM_ACTION_J2_MAX = 1.8    # halt if denormalized action[J2] above this


def load_policy_processors(policy, checkpt: str, device: torch.device):
    """Load normalization pipelines saved with the trained policy."""
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {"device": device.type},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {"device": device.type},
        "device_processor": {"device": "cpu"},
    }
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpt,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


def prepare_observation(state, wrist_img, global_img, device, expected_state_dim=7, phase=0.0,
                        gripper_unit_scale=1.0):
    """Convert raw numpy data to inference-ready tensors (batched, on device).

    state: raw robot joint positions [j1..j6, gripper] in robot units.
    gripper_unit_scale: multiply state[6] by this before feeding to policy,
      so the policy sees values in its training distribution.
    """
    obs = {}
    state_arr = np.asarray(state, dtype=np.float32).copy()
    state_arr[6] *= gripper_unit_scale
    if expected_state_dim == len(state_arr) + 1:
        state_arr = np.concatenate([state_arr, np.asarray([phase], dtype=np.float32)])
    elif expected_state_dim != len(state_arr):
        raise ValueError(
            f"Policy expects observation.state dim {expected_state_dim}, "
            f"but robot provides {len(state_arr)} joints. Only dim 7 or 8-with-phase is supported."
        )
    obs["observation.state"] = torch.from_numpy(
        state_arr
    ).unsqueeze(0).to(device)

    if wrist_img is not None:
        t = torch.from_numpy(wrist_img).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(device)
        obs["observation.images.wrist_rgb"] = t

    if global_img is not None:
        t = torch.from_numpy(global_img).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(device)
        obs["observation.images.global_rgb"] = t

    return obs


def build_preview(wrist_frame, global_frame, text: str, color=(0, 255, 0)):
    preview = cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if global_frame is not None:
        g_preview = cv2.cvtColor(global_frame.rgb, cv2.COLOR_RGB2BGR)
        g_preview = cv2.resize(g_preview, (preview.shape[1], preview.shape[0]))
        preview = np.hstack([preview, g_preview])
    return preview


def should_quit(key: int) -> bool:
    return key in (27, ord('q'), ord('Q'))


def fmt_vec(values, precision=3):
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def policy_state_dim(policy) -> int:
    feature = policy.config.input_features.get("observation.state")
    if feature is None:
        return 0
    return int(feature.shape[0])


def max_abs_diff(cur, prev) -> float:
    if prev is None:
        return float("nan")
    return float(np.max(np.abs(np.asarray(cur, dtype=np.float32) - np.asarray(prev, dtype=np.float32))))


def interpolate_joint_path(start: np.ndarray, target: np.ndarray,
                           max_step_rad: float, max_step_gripper: float):
    """Generate intermediate joint targets from start to target (excluding start, including target)."""
    diff = np.asarray(target, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    arm_steps = int(np.ceil(np.max(np.abs(diff[:6])) / max_step_rad)) if max_step_rad > 0 else 1
    grip_steps = int(np.ceil(abs(diff[6]) / max_step_gripper)) if max_step_gripper > 0 else 1
    n_steps = max(arm_steps, grip_steps, 1)
    waypoints = []
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp = np.asarray(start, dtype=np.float32) + diff * alpha
        waypoints.append(interp)
    return waypoints


def select_policy_action(policy, postprocessor, normalized_obs, replan_every_step: bool,
                         is_hybrid=False):
    """Return (first_action, full_chunk). full_chunk is None when queue-based."""
    if is_hybrid:
        # normalized_obs is actually (wrist_img_tensor, state_tensor) for hybrid
        wrist_img, state_t = normalized_obs
        with torch.inference_mode():
            pred = policy(wrist_img.unsqueeze(0), state_t.unsqueeze(0))
        return pred, pred.squeeze(0).cpu().numpy()

    if replan_every_step:
        if hasattr(policy, "predict_action_chunk"):
            action_chunk = policy.predict_action_chunk(normalized_obs)
            action_chunk = postprocessor(action_chunk)
            return action_chunk[:, 0, :], action_chunk.squeeze(0).cpu().numpy()  # (chunk, 7)

        policy.reset()

    action = policy.select_action(normalized_obs)
    return postprocessor(action), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str, default=None,
                        help="Path to trained ACT checkpoint dir (required for --policy-type act).")
    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--velocity-pct", type=int, default=25)
    parser.add_argument("--hz", type=float, default=30.0,
                        help="Control loop frequency. Keep this equal to dataset fps.")
    parser.add_argument("--max-steps", type=int, default=APPROACH_STEPS_DEFAULT,
                        help="Maximum action steps for one grasp attempt.")
    parser.add_argument("--test-mode", choices=("A", "B", "C", "D"), default="A",
                        help="A: approach only. B: approach + close + lift. C: approach + descend. D: full grasp + place + release.")
    parser.add_argument("--descend-j2-delta", type=float, default=0.04,
                        help="J2 increment (rad) for descent phase in test mode C.")
    parser.add_argument("--place-j1-offset", type=float, default=0.30,
                        help="J1 offset (rad) to move bottle to side before release (test mode D).")
    parser.add_argument("--approach-steps", type=int, default=APPROACH_STEPS_DEFAULT,
                        help="Stop ACT after this many steps and begin handover to code control.")
    parser.add_argument("--action-smooth", type=float, default=ACTION_SMOOTH_ALPHA,
                        help="EMA smoothing factor for consecutive action predictions (0=disabled, 0.3=default).")
    parser.add_argument("--no-global", action="store_true",
                        help="Disable global camera")
    parser.add_argument("--global-camera", type=str, default="auto",
                        help="Global camera device ID or 'auto'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run policy and preview targets without sending robot commands.")
    parser.add_argument("--debug-actions", action="store_true",
                        help="Print raw/clipped/smoothed/sent action at every debug-every step.")
    parser.add_argument("--debug-every", type=int, default=10,
                        help="Print one debug line every N action steps.")
    parser.add_argument("--replan-every-step", action="store_true",
                        help="Recompute a fresh ACT action chunk every control step instead of consuming the action queue.")
    parser.add_argument("--action-mode", choices=("absolute", "delta"), default="absolute",
                        help="Interpret policy action as absolute joint target or state-relative delta waypoint.")
    parser.add_argument("--delta-scale", type=float, default=1.0,
                        help="Multiplier for model-predicted delta before adding to current state.")
    parser.add_argument("--arm-scale", type=float, default=None,
                        help="Scale for J1/J2/J3 (arm joints). Defaults to --delta-scale.")
    parser.add_argument("--wrist-scale", type=float, default=None,
                        help="Scale for J4/J5/J6 (wrist joints). Defaults to --delta-scale.")
    parser.add_argument("--gripper-scale", type=float, default=None,
                        help="Scale for gripper (dim 7). Defaults to --delta-scale.")
    parser.add_argument("--gripper-deadband", type=float, default=0.0,
                        help="Ignore raw gripper action below this absolute value.")
    parser.add_argument("--min-gripper-delta", type=float, default=0.0,
                        help="If abs(raw_gripper) > deadband but abs(scaled) < min, override to min_gripper_delta.")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable cv2 GUI windows; use terminal input (Enter=grasp, q=quit).")
    parser.add_argument("--gripper-unit-scale", type=float, default=1.0,
                        help="Scale robot gripper state before feeding to policy, "
                             "and inverse-scale the target back before sending to robot.")
    parser.add_argument("--training-gripper-min", type=float, default=None,
                        help="Min gripper value in training data, for OOD warning.")
    parser.add_argument("--training-gripper-max", type=float, default=None,
                        help="Max gripper value in training data, for OOD warning.")
    parser.add_argument("--no-return-to-start", action="store_true",
                        help="Disable automatic return to start_pose after trajectory.")
    parser.add_argument("--debug-policy-io", action="store_true",
                        help="Print policy input/output at every debug-every step: "
                        "raw robot_state[J2], obs.state[J2], normalized_state[J2], raw_action[J2].")
    parser.add_argument("--save-rollout", action="store_true",
                        help="Save rollout frames (image, state, action) to disk for offline analysis.")
    parser.add_argument("--policy-type", choices=("act", "hybrid", "hybrid-v3", "hybrid-v4-delta"), default="act",
                        help="Policy architecture: act (default), hybrid, hybrid-v3, or hybrid-v4-delta (lookahead delta).")
    parser.add_argument("--hybrid-checkpt", type=str, default=None,
                        help="Path to hybrid policy .pt checkpoint (for --policy-type hybrid).")
    parser.add_argument("--clamp-delta-j2-nonnegative", action="store_true",
                        help="[hybrid-v3 only] Clamp image_delta[J2] >= 0 before combining with base_action. "
                        "Prevents image_delta from cancelling state_head forward push.")
    parser.add_argument("--max-joint-delta", type=float, default=None,
                        help="Override per-joint max delta for all arm joints. 0=disabled. Default uses per-joint limits.")
    parser.add_argument("--wrist-freeze-j2", type=float, default=1.45,
                        help="J2 threshold to freeze J4-J6 wrist joints (default: 1.45).")
    parser.add_argument("--ready-j2", type=float, default=1.50,
                        help="J2 threshold for ready_count (default: 1.50).")
    parser.add_argument("--ready-count-min", type=int, default=5,
                        help="Consecutive steps above READY_J2 to trigger stop (default: 5).")
    parser.add_argument("--v4-j2-only", action="store_true",
                        help="[hybrid-v4-delta] Use v4 delta only for J2; other joints from v2 (ver A) or position-hold (ver B).")
    parser.add_argument("--v4-j2-only-ver", choices=("A", "B"), default="A",
                        help="Version A: J1/J3/J4/J5/J6 from v2 baseline. Version B: hold current position (default: A).")
    parser.add_argument("--v2-checkpt", type=str, default="outputs/train/hybrid_v2.pt",
                        help="Path to hybrid v2 checkpoint for --v4-j2-only --v4-j2-only-ver A.")
    args = parser.parse_args()

    # Apply command-line overrides to shared constants
    WRIST_FREEZE_J2 = args.wrist_freeze_j2
    READY_J2 = args.ready_j2
    READY_COUNT_MIN = args.ready_count_min

    # Validate checkpoint argument
    if args.policy_type in ("hybrid", "hybrid-v3", "hybrid-v4-delta") and not args.hybrid_checkpt:
        parser.error("--hybrid-checkpt is required when --policy-type hybrid, hybrid-v3, or hybrid-v4-delta")
    if args.policy_type == "act" and not args.checkpt:
        parser.error("--checkpt is required when --policy-type act (default)")

    # Build per-dimension scale array: [J1, J2, J3, J4, J5, J6, Grip]
    arm_s = args.arm_scale if args.arm_scale is not None else args.delta_scale
    wrist_s = args.wrist_scale if args.wrist_scale is not None else args.delta_scale
    grip_s = args.gripper_scale if args.gripper_scale is not None else args.delta_scale
    dim_scale = np.array([arm_s, arm_s, arm_s, wrist_s, wrist_s, wrist_s, grip_s], dtype=np.float32)

    # Build per-joint max delta: use override if provided, else per-joint defaults
    if args.max_joint_delta is not None and args.max_joint_delta > 0:
        max_delta = np.full(6, args.max_joint_delta, dtype=np.float32)
    else:
        max_delta = MAX_DELTA_PER_JOINT.copy()

    print("=" * 60)
    print("  Piper ACT Deployment — Bottle Approach (v0.7.0)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Load policy ---
    if args.policy_type in ("hybrid", "hybrid-v3", "hybrid-v4-delta"):
        hybrid_ckpt = args.hybrid_checkpt or "outputs/train/hybrid_state_cond_14ep.pt"
        print(f"\n[1/4] Loading {args.policy_type} policy from {hybrid_ckpt} ...")
        ckpt = torch.load(hybrid_ckpt, map_location=device, weights_only=False)
        model_args = ckpt["args"]
        is_v4_delta = (args.policy_type == "hybrid-v4-delta")

        if args.v4_j2_only and not is_v4_delta:
            parser.error("--v4-j2-only requires --policy-type hybrid-v4-delta")

        if is_v4_delta:
            policy = HybridDeltaPolicy(
                state_dim=model_args.get("state_dim", 7),
                delta_dim=model_args.get("delta_dim", 6),
                img_feat_dim=model_args.get("img_feat_dim", 256),
                state_feat_dim=model_args.get("state_feat_dim", 64),
                state_hidden=model_args.get("state_hidden", 128),
                action_hidden=model_args.get("action_hidden", 256),
                use_global_img=model_args.get("use_global_img", False),
            ).to(device)
        elif args.policy_type == "hybrid-v3":
            policy = StateConditionedPolicyV3(
                state_dim=7, action_dim=7,
                img_feat_dim=model_args.get("img_feat_dim", 256),
                state_feat_dim=model_args.get("state_feat_dim", 64),
                state_hidden=model_args.get("state_hidden", 128),
                action_hidden=model_args.get("action_hidden", 256),
                use_global_img=model_args.get("use_global_img", False),
            ).to(device)
        else:
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
        hybrid_img_h = model_args.get("img_size", 160)
        hybrid_img_w = int(hybrid_img_h * 4 / 3)
        chunk_size = 1
        n_action_steps = 1
        expected_state_dim = 7
        uses_phase = False
        preprocessor = None
        postprocessor = None
        # Load norm stats
        norm_stats = ckpt.get("norm_stats", None)
        _v3_internals = None  # initialized here for scoping; set to dict if v3 clamp enabled
        hybrid_delta_mean = hybrid_delta_std = None  # v4 only
        if norm_stats:
            hybrid_state_mean = np.array(norm_stats["state_mean"], dtype=np.float32)
            hybrid_state_std = np.array(norm_stats["state_std"], dtype=np.float32)
            hybrid_state_std = np.maximum(hybrid_state_std, MIN_NORM_STD)
            if is_v4_delta:
                # V4 uses delta normalization (6D arm delta), not action normalization
                hybrid_delta_mean = np.array(norm_stats["delta_mean"], dtype=np.float32)
                hybrid_delta_std = np.array(norm_stats["delta_std"], dtype=np.float32)
                hybrid_delta_std = np.maximum(hybrid_delta_std, MIN_NORM_STD)
                hybrid_action_mean = hybrid_action_std = None  # not used by v4
                k = model_args.get("lookahead_k", "?")
                extra = f"  K={k}  residual_scale={policy.residual_scale.tolist()}"
            else:
                hybrid_action_mean = np.array(norm_stats["action_mean"], dtype=np.float32)
                hybrid_action_std = np.array(norm_stats["action_std"], dtype=np.float32)
                hybrid_action_std = np.maximum(hybrid_action_std, MIN_NORM_STD)
                extra = ""
                if args.policy_type == "hybrid-v3":
                    extra = f"  img_gate={ckpt.get('img_gate_final','?'):.2f}  state_gate={ckpt.get('state_gate_final','?'):.2f}"
            print(f"  Policy loaded: {sum(p.numel() for p in policy.parameters()):,} params  "
                  f"img_size=({hybrid_img_h},{hybrid_img_w})  "
                  f"improvement_ratio={ckpt.get('improvement_ratio','?'):.4f}  "
                  f"norm=on{extra}")
            # ── V3 delta clamp: register hooks to capture base/delta ──
            if args.policy_type == "hybrid-v3" and args.clamp_delta_j2_nonnegative:
                _v3_internals = {}
                policy.state_head.register_forward_hook(
                    lambda m, inp, out: _v3_internals.update(base_norm=out.detach()))
                policy.image_delta_head.register_forward_hook(
                    lambda m, inp, out: _v3_internals.update(delta_norm=out.detach()))
                print(f"  V3 delta clamp: ON (image_delta[J2] >= 0 in normalized space)")
            else:
                _v3_internals = None

            # ── v4-j2-only: load v2 model for non-J2 joints ──
            v2_model = None
            v2_action_mean = v2_action_std = None
            if is_v4_delta and args.v4_j2_only and args.v4_j2_only_ver == "A":
                v2_ckpt_path = args.v2_checkpt
                print(f"  [v4-j2-only Ver A] Loading v2 baseline from {v2_ckpt_path} ...")
                v2_ckpt = torch.load(v2_ckpt_path, map_location=device, weights_only=False)
                v2_args = v2_ckpt["args"]
                v2_model = StateConditionedPolicy(
                    state_dim=7, action_dim=7,
                    img_feat_dim=v2_args.get("img_feat_dim", 256),
                    state_feat_dim=v2_args.get("state_feat_dim", 128),
                    state_hidden=v2_args.get("state_hidden", 128),
                    action_hidden=v2_args.get("action_hidden", 256),
                    use_global_img=v2_args.get("use_global_img", False),
                ).to(device)
                v2_model.load_state_dict(v2_ckpt["model_state_dict"])
                v2_model.eval()
                v2_ns = v2_ckpt.get("norm_stats", {})
                if v2_ns:
                    v2_action_mean = np.array(v2_ns["action_mean"], dtype=np.float32)
                    v2_action_std = np.maximum(np.array(v2_ns["action_std"], dtype=np.float32), MIN_NORM_STD)
                print(f"  V2 baseline loaded: {sum(p.numel() for p in v2_model.parameters()):,} params"
                      f"  improvement_ratio={v2_ckpt.get('improvement_ratio','?'):.4f}")
        else:
            hybrid_state_mean = hybrid_state_std = None
            hybrid_action_mean = hybrid_action_std = None
            print(f"  Policy loaded: {sum(p.numel() for p in policy.parameters()):,} params  "
                  f"img_size=({hybrid_img_h},{hybrid_img_w})  "
                  f"improvement_ratio={ckpt.get('improvement_ratio','?'):.4f}  "
                  f"norm=off")
    else:
        print(f"\n[1/4] Loading ACT policy from {args.checkpt} ...")
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(args.checkpt)
        policy.to(device)
        policy.eval()
        chunk_size = policy.config.chunk_size
        n_action_steps = policy.config.n_action_steps
        expected_state_dim = policy_state_dim(policy)
        uses_phase = expected_state_dim == 8
        hybrid_img_h = None
        hybrid_img_w = None
        print(
            f"  Policy loaded (chunk_size={chunk_size}, n_action_steps={n_action_steps}, "
            f"state_dim={expected_state_dim})."
        )

        # --- Load pre/post processors ---
        print("\n[2/4] Loading pre/post processors from checkpoint ...")
        preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
        print("  Processors ready.")

    # --- Connect robot ---
    step_n = 2 if args.policy_type in ("hybrid", "hybrid-v3") else 3
    total_n = 3 if args.policy_type in ("hybrid", "hybrid-v3") else 4
    print(f"\n[{step_n}/{total_n}] Connecting Piper ({args.can_port}) ...")
    robot = PiperRobot(can_port=args.can_port, disable_torque_on_disconnect=False)
    robot.connect()  # connect + enable in one call
    print("  Robot connected and enabled (disable_torque_on_disconnect=False).")

    # --- Init cameras ---
    step_n += 1
    print(f"\n[{step_n}/{total_n}] Initializing cameras ...")
    rs_serials = find_realsense_devices()
    wrist_serial = rs_serials[0] if rs_serials else ""
    wrist_cam = RealSenseCamera(serial=wrist_serial, width=640, height=480, fps=30,
                                enable_depth=False)

    global_cam = None
    if args.policy_type in ("hybrid", "hybrid-v3", "hybrid-v4-delta"):
        requires_global = False
    else:
        requires_global = "observation.images.global_rgb" in policy.config.input_features
    if args.no_global and requires_global:
        raise ValueError("This policy was trained with global_rgb, so --no-global cannot be used.")
    if not args.no_global:
        try:
            global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
        except IOError as e:
            if requires_global:
                raise
            print(f"  Global camera skipped: {e}")
    print("  Cameras ready.")

    # --- Gripper distribution check ---
    robot_state = robot.get_joint_positions()
    grip_raw = robot_state[6]
    grip_policy = grip_raw * args.gripper_unit_scale
    print(f"\n  Gripper state (robot units): {grip_raw:.6f}")
    if args.gripper_unit_scale != 1.0:
        print(f"  Gripper state (policy units): {grip_policy:.6f}  [×{args.gripper_unit_scale}]")
    if args.training_gripper_min is not None and args.training_gripper_max is not None:
        if grip_policy < args.training_gripper_min or grip_policy > args.training_gripper_max:
            print(f"  [WARN] Deployment gripper state ({grip_policy:.4f} in policy units)"
                  f" is outside training distribution"
                  f" [{args.training_gripper_min:.3f}, {args.training_gripper_max:.3f}].")

    print("\n" + "-" * 60)
    print("  SPACE = run approach    Q/ESC = quit")
    print(f"  TEST MODE: {args.test_mode}  |  APPROACH STEPS: {args.approach_steps}"
          f"  |  SMOOTH: α={args.action_smooth}")
    print(f"  Per-joint max_delta: J1-J3={max_delta[0]:.3f}  J4-J6={max_delta[3]:.3f} rad")
    print(f"  Wrist freeze @ J2 > {WRIST_FREEZE_J2:.2f}  |  Ready stop @ J2 > {READY_J2:.2f} ×{READY_COUNT_MIN}")
    print(f"  Gripper forced OPEN ({GRIPPER_OPEN:.3f} m) during ACT approach.")
    if args.test_mode == "A":
        print("  → Approach only — no close, no lift.")
    elif args.test_mode == "C":
        print(f"  → Approach + descend (J2 += {args.descend_j2_delta:.3f} rad) — no close.")
    elif args.test_mode == "D":
        print(f"  → Full grasp: approach + close + lift + place(J1+={args.place_j1_offset:.2f}) + release + return.")
    else:
        print(f"  → Approach + close ({GRIPPER_CLOSE:.3f} m) + lift (J3 -= 0.06).")
    if args.dry_run:
        print("  DRY RUN: robot commands will not be sent.")
    if args.replan_every_step:
        print("  REPLAN: policy will predict a fresh first action at every step.")
    if args.v4_j2_only:
        ver_label = "v2 baseline" if args.v4_j2_only_ver == "A" else "position-hold"
        print(f"  [v4-j2-only Ver {args.v4_j2_only_ver}] J2 from v4 delta, "
              f"J1/J3/J4/J5/J6 from {ver_label}.")
    print("-" * 60 + "\n")

    try:
        while True:
            # --- Live preview ---
            if args.no_gui:
                cmd = input("  Press ENTER to run approach, Q then ENTER to quit: ").strip().lower()
                if cmd == "q":
                    break
            else:
                wrist_frame = wrist_cam.read()
                global_frame = global_cam.read() if global_cam else None

                preview = build_preview(wrist_frame, global_frame, "READY - SPACE run")
                cv2.imshow("ACT Deployment", preview)

                key = cv2.waitKey(1) & 0xFF
                if should_quit(key):
                    break
                if key != ord(' '):
                    continue

            # ================================================================
            #  ACT APPROACH PHASE
            # ================================================================
            print(f"  >>> Approach attempt (test-mode={args.test_mode}, {args.approach_steps} steps) ...")

            # Setup rollout saving directory
            rollout_dir = None
            if args.save_rollout:
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                rollout_dir = Path(PROJECT_ROOT) / "logs" / "rollouts" / f"test_a_{ts}"
                rollout_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Saving rollout to {rollout_dir}")

            # Save start position for auto return
            start_robot_state = np.asarray(robot.get_joint_positions(), dtype=np.float32)

            # Reset action queue before new trajectory
            if args.policy_type == "act":
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()

            approach_steps = args.approach_steps
            last_smoothed = None
            last_state = None
            raw_actions = []
            paused = False
            user_quit = False
            stagnation_count = 0
            ready_count = 0
            stop_reason = "completed"

            for step in range(approach_steps):
                loop_start = time.time()

                # Capture fresh observation
                wrist_frame = wrist_cam.read()
                global_frame = global_cam.read() if global_cam else None
                robot_state = robot.get_joint_positions()
                phase = 0.0 if approach_steps <= 1 else min(1.0, step / float(approach_steps - 1))

                # Build observation
                wrist_img = wrist_frame.rgb
                global_img = global_frame.rgb if global_frame else None

                if args.policy_type in ("hybrid", "hybrid-v3", "hybrid-v4-delta"):
                    import torchvision.transforms.functional as TF
                    # ── Stage 0: prepare image ──
                    img_t = torch.from_numpy(wrist_img).float() / 255.0
                    img_t = img_t.permute(2, 0, 1)  # (C, H, W)
                    img_t = TF.resize(img_t, (hybrid_img_h, hybrid_img_w), antialias=True)
                    img_t = img_t.to(device)

                    # ── Stage 1: capture raw robot state ──
                    robot_state_arr_hy = np.asarray(robot_state, dtype=np.float32)

                    # ── Stage 2: normalize state ──
                    if hybrid_state_mean is not None:
                        state_norm = (robot_state_arr_hy - hybrid_state_mean) / hybrid_state_std
                        state_norm = np.clip(state_norm, -NORM_STATE_CLIP, NORM_STATE_CLIP)
                    else:
                        state_norm = robot_state_arr_hy.copy()
                    state_t = torch.from_numpy(state_norm).float().to(device)

                    # ── Stage 3: model inference ──
                    with torch.inference_mode():
                        action, _ = select_policy_action(
                            policy, None, (img_t, state_t), True, is_hybrid=True
                        )
                    model_output_norm = action.squeeze(0).cpu().numpy() if action.dim() == 2 else action.cpu().numpy()

                    # ── Stage 3b (v3 only): clamp image_delta[J2] >= 0 ──
                    v3_base_norm = None
                    v3_delta_norm_raw = None
                    if _v3_internals is not None:
                        v3_base_norm = _v3_internals["base_norm"].squeeze(0).cpu().numpy().copy()
                        v3_delta_norm_raw = _v3_internals["delta_norm"].squeeze(0).cpu().numpy().copy()
                        v3_delta_norm_clamped = v3_delta_norm_raw.copy()
                        if v3_delta_norm_clamped[1] < 0.0:
                            v3_delta_norm_clamped[1] = 0.0
                        final_norm_clamped = v3_base_norm + v3_delta_norm_clamped
                        model_output_norm_before = model_output_norm.copy()
                        model_output_norm = final_norm_clamped

                    # ── Stage 4: denormalize ──
                    base_robot_j2 = delta_robot_raw_j2 = delta_robot_clamped_j2 = None
                    final_before_robot_j2 = final_after_robot_j2 = None
                    v4_delta_robot = None
                    v4_base_delta_robot = None
                    v4_img_res_robot = None
                    if is_v4_delta:
                        # V4: output is 6D normalized delta
                        v4_delta_robot = model_output_norm * hybrid_delta_std + hybrid_delta_mean
                        model_action = np.zeros(7, dtype=np.float32)

                        if args.v4_j2_only:
                            # ── v4-j2-only: only J2 from v4 delta ──
                            model_action[1] = robot_state_arr_hy[1] + v4_delta_robot[1]
                            if args.v4_j2_only_ver == "A":
                                # Version A: J1/J3/J4/J5/J6 from v2 baseline
                                with torch.inference_mode():
                                    v2_out_norm = v2_model(img_t.unsqueeze(0), state_t.unsqueeze(0))
                                v2_out = v2_out_norm.squeeze(0).cpu().numpy() * v2_action_std + v2_action_mean
                                model_action[0] = v2_out[0]   # J1
                                model_action[2] = v2_out[2]   # J3
                                model_action[3] = v2_out[3]   # J4
                                model_action[4] = v2_out[4]   # J5
                                model_action[5] = v2_out[5]   # J6
                            else:
                                # Version B: J1/J3/J4/J5/J6 hold current position
                                model_action[0] = robot_state_arr_hy[0]   # J1
                                model_action[2] = robot_state_arr_hy[2]   # J3
                                model_action[3] = robot_state_arr_hy[3]   # J4
                                model_action[4] = robot_state_arr_hy[4]   # J5
                                model_action[5] = robot_state_arr_hy[5]   # J6
                            model_action[6] = GRIPPER_OPEN
                        else:
                            # Standard v4: all 6 arm joints from delta
                            model_action[:6] = robot_state_arr_hy[:6] + v4_delta_robot
                            model_action[6] = GRIPPER_OPEN

                        if args.debug_policy_io and (
                            step == 0 or step == approach_steps - 1 or (step + 1) % args.debug_every == 0
                        ):
                            # Get decomposition for logging
                            with torch.inference_mode():
                                _, base_d, img_r = policy.forward_with_internals(
                                    img_t.unsqueeze(0), state_t.unsqueeze(0))
                            v4_base_delta_robot = base_d.squeeze(0).cpu().numpy() * hybrid_delta_std + hybrid_delta_mean
                            v4_img_res_robot = img_r.squeeze(0).cpu().numpy() * hybrid_delta_std + hybrid_delta_mean
                    elif hybrid_action_mean is not None:
                        model_action = model_output_norm * hybrid_action_std + hybrid_action_mean
                        if v3_base_norm is not None:
                            base_robot_j2 = float(v3_base_norm[1] * hybrid_action_std[1])
                            delta_robot_raw_j2 = float(v3_delta_norm_raw[1] * hybrid_action_std[1])
                            delta_robot_clamped_j2 = float(v3_delta_norm_clamped[1] * hybrid_action_std[1])
                            final_before_robot_j2 = float(model_output_norm_before[1] * hybrid_action_std[1] + hybrid_action_mean[1])
                            final_after_robot_j2 = float(model_action[1])
                    else:
                        model_action = model_output_norm
                    raw_actions.append(model_action.copy())

                    # ── Stage 5: sanity check ──
                    action_j2 = float(model_action[1])
                    if is_v4_delta:
                        # V4 sanity: check target J2 and delta J2
                        delta_j2 = float(v4_delta_robot[1])
                        if model_action[1] < robot_state_arr_hy[1] - 0.02:
                            print(f"\n  [HALT] v4 target[J2]={model_action[1]:.4f} < current_J2={robot_state_arr_hy[1]:.4f} - 0.02")
                            print(f"    pred_delta[J2] = {delta_j2:.4f}")
                            stop_reason = "action_sanity"
                            break
                        if model_action[1] > 1.8:
                            print(f"\n  [HALT] v4 target[J2]={model_action[1]:.4f} > 1.8")
                            print(f"    pred_delta[J2] = {delta_j2:.4f}")
                            stop_reason = "action_sanity"
                            break
                        if robot_state_arr_hy[1] < READY_J2 and delta_j2 < -0.01:
                            print(f"\n  [WARN] v4 pred_delta[J2]={delta_j2:.4f} < -0.01 while J2 < READY_J2 — unexpected negative delta")
                    elif action_j2 < UNNORM_ACTION_J2_MIN or action_j2 > UNNORM_ACTION_J2_MAX:
                        print(f"\n  [HALT] action_after_unnorm[J2]={action_j2:.4f} "
                              f"outside safe range [{UNNORM_ACTION_J2_MIN}, {UNNORM_ACTION_J2_MAX}]")
                        print(f"    model_output_norm[J2] = {model_output_norm[1]:.4f}")
                        print(f"    robot_state[J2] = {robot_state_arr_hy[1]:.4f}")
                        print(f"    state_norm[J2]  = {state_norm[1]:.4f}")
                        if v3_base_norm is not None:
                            print(f"    base_action[J2]        = {base_robot_j2:.4f}")
                            print(f"    raw_image_delta[J2]    = {delta_robot_raw_j2:.4f}")
                            print(f"    clamped_image_delta[J2]= {delta_robot_clamped_j2:.4f}")
                        print(f"    This indicates the model is extrapolating wildly (OOD input).")
                        print(f"    Check: gripper, camera, or robot starting pose.")
                        stop_reason = "action_sanity"
                        break

                    # ── Policy I/O debug for hybrid ──
                    if args.debug_policy_io and (
                        step == 0 or step == approach_steps - 1 or (step + 1) % args.debug_every == 0
                    ):
                        print(f"  [POLICY-IO] step={step+1:03d}/{approach_steps}")
                        print(f"    robot_state  (raw)     = {fmt_vec(robot_state_arr_hy)}")
                        print(f"    state_norm   (model in)= {fmt_vec(state_norm)}")
                        print(f"    model_output (norm)    = {fmt_vec(model_output_norm)}")
                        if is_v4_delta:
                            print(f"    pred_delta   (robot)   = {fmt_vec(v4_delta_robot)}")
                            if v4_base_delta_robot is not None:
                                print(f"    base_delta[J2]         = {v4_base_delta_robot[1]:.4f}")
                                print(f"    img_residual[J2]       = {v4_img_res_robot[1]:.4f}")
                            if args.v4_j2_only:
                                print(f"    target_arm   (robot)   = {fmt_vec(model_action[:6])}  [v4-j2-only ver {args.v4_j2_only_ver}]")
                                print(f"    J2 from v4 delta, J1/J3/J4/J5/J6 from {'v2' if args.v4_j2_only_ver == 'A' else 'position-hold'}")
                            else:
                                print(f"    target_arm   (robot)   = {fmt_vec(model_action[:6])}")
                            print(f"    target_grip  (forced)  = {model_action[6]:.3f}")
                            print(f"    current_qpos[J2]       = {robot_state_arr_hy[1]:.4f}")
                        else:
                            print(f"    action_unnorm(robot)   = {fmt_vec(model_action)}")
                            if v3_base_norm is not None:
                                print(f"    base_action[J2]        = {base_robot_j2:.4f}")
                                print(f"    raw_image_delta[J2]    = {delta_robot_raw_j2:.4f}")
                                print(f"    clamped_image_delta[J2]= {delta_robot_clamped_j2:.4f}")
                                print(f"    final_before_clamp[J2] = {final_before_robot_j2:.4f}")
                                print(f"    final_after_clamp[J2]  = {final_after_robot_j2:.4f}")
                                print(f"    current_qpos[J2]       = {robot_state_arr_hy[1]:.4f}")
                else:
                    obs = prepare_observation(
                        robot_state, wrist_img, global_img, device, expected_state_dim, phase,
                        gripper_unit_scale=args.gripper_unit_scale,
                    )

                    # Run inference
                    with torch.inference_mode():
                        normalized_obs = preprocessor(obs)
                        action, _ = select_policy_action(
                            policy, postprocessor, normalized_obs, args.replan_every_step
                        )

                    # action shape: (1, 7) -> (7,)
                    if action.dim() == 2:
                        action = action.squeeze(0)
                    model_action = action.cpu().numpy()
                    raw_actions.append(model_action.copy())

                    # ── Policy I/O debug for ACT ──
                    if args.debug_policy_io and (
                        step == 0 or step == approach_steps - 1 or (step + 1) % args.debug_every == 0
                    ):
                        nstate = normalized_obs["observation.state"].squeeze(0).cpu().numpy()
                        raw_obs_state = obs["observation.state"].squeeze(0).cpu().numpy()
                        robot_state_arr_tmp = np.asarray(robot_state, dtype=np.float32)
                        print(f"  [POLICY-IO] step={step+1:03d}/{approach_steps}")
                        print(f"    robot_state  (raw)     = {fmt_vec(robot_state_arr_tmp)}")
                        print(f"    obs.state    (raw)     = {fmt_vec(raw_obs_state)}")
                        print(f"    state_norm   (model in)= {fmt_vec(nstate)}")
                        print(f"    model_action (robot)   = {fmt_vec(model_action)}")

                robot_state_arr = np.asarray(robot_state, dtype=np.float32)

                # ── Compute raw_target from model output ──
                if args.action_mode == "delta":
                    scaled_action = model_action * dim_scale
                    policy_state_arr = robot_state_arr.copy()
                    policy_state_arr[6] *= args.gripper_unit_scale
                    policy_target = policy_state_arr + scaled_action
                    raw_target = policy_target.copy()
                    raw_target[6] /= args.gripper_unit_scale
                else:
                    raw_target = model_action.copy()

                # ── Step 1: per-joint independent delta clamp ──
                raw_delta = raw_target - robot_state_arr
                for j in range(6):
                    raw_delta[j] = np.clip(raw_delta[j], -max_delta[j], max_delta[j])
                clipped = robot_state_arr + raw_delta

                # Force gripper open throughout approach
                clipped[6] = GRIPPER_OPEN

                # ── Step 2: wrist freeze when J2 > WRIST_FREEZE_J2 ──
                wrist_frozen = False
                if robot_state_arr[1] > WRIST_FREEZE_J2:
                    clipped[3:6] = robot_state_arr[3:6]
                    wrist_frozen = True

                # ── Step 3: EMA smoothing ──
                alpha = args.action_smooth
                if last_smoothed is not None and alpha > 0:
                    smoothed_arm = alpha * clipped[:6] + (1.0 - alpha) * last_smoothed[:6]
                else:
                    smoothed_arm = clipped[:6].copy()
                sent_target = np.concatenate([smoothed_arm, [GRIPPER_OPEN]])

                # Safety clamp to joint limits
                sent_target[:6] = np.clip(sent_target[:6], -3.14, 3.14)
                sent_target[6] = np.clip(sent_target[6], 0.0, PIPER_GRIPPER_MAX_M)

                # ── Ready stop: J2 > READY_J2 for READY_COUNT_MIN consecutive steps after step 160 ──
                if robot_state_arr[1] > READY_J2 and step > 160:
                    ready_count += 1
                else:
                    ready_count = 0
                stop_act = (ready_count >= READY_COUNT_MIN) or (step + 1 >= approach_steps)

                # ── Logging ──
                if args.debug_actions and (
                    step == 0 or step == approach_steps - 1 or (step + 1) % args.debug_every == 0
                    or wrist_frozen or stop_act
                ):
                    print(f"  --- step {step+1:03d}/{approach_steps} ---")
                    print(f"    robot_state  : {fmt_vec(robot_state_arr)}")
                    print(f"    raw_action   : {fmt_vec(raw_target)}")
                    print(f"    clipped      : {fmt_vec(clipped)}")
                    print(f"    smoothed     : {fmt_vec(sent_target)}")
                    print(f"    sent_target  : {fmt_vec(sent_target)}")
                    sent_delta = sent_target - robot_state_arr
                    print(f"    delta (sent) : {fmt_vec(sent_delta)}")
                    print(f"    J2={robot_state_arr[1]:.4f}  wrist_frozen={wrist_frozen}"
                          f"  ready={ready_count}/{READY_COUNT_MIN}  stop_act={stop_act}")

                # ── Safety stop: joint limit violation ──
                if np.any(np.abs(sent_target[:6]) > 3.0):
                    print(f"\n  [STOP] Joint limit violation: target={fmt_vec(sent_target)}")
                    stop_reason = "joint_limit"
                    break

                # ── Safety stop: stagnation (state barely moves, not near end) ──
                state_diff = max_abs_diff(robot_state, last_state)
                near_end = step > approach_steps * 0.7
                if not near_end and last_state is not None and state_diff < STAGNATION_THRESHOLD:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                if stagnation_count >= STAGNATION_STEPS:
                    print(f"\n  [STOP] Stagnation: {STAGNATION_STEPS} consecutive steps"
                          f" with state_diff < {STAGNATION_THRESHOLD} before 70% progress")
                    print(f"    step={step+1}/{approach_steps}  state_diff={state_diff:.6f}")
                    stop_reason = "stagnation"
                    break

                # ── Ready stop: break after sending ──
                if stop_act:
                    if ready_count >= READY_COUNT_MIN:
                        stop_reason = "ready"
                    else:
                        stop_reason = "max_steps"
                    # Send the final target, then break
                    if not args.dry_run:
                        robot.set_joint_positions(sent_target.tolist(), velocity_pct=args.velocity_pct)
                    # Update last_smoothed before breaking
                    last_smoothed = smoothed_arm.copy()
                    last_state = np.asarray(robot_state, dtype=np.float32).copy()
                    print(f"\n  [STOP] Approach complete ({stop_reason})"
                          f"  J2={robot_state_arr[1]:.4f}  step={step+1}")
                    break

                # ── Save rollout data ──
                if rollout_dir is not None and (step % 5 == 0 or step < 5 or step >= approach_steps - 5):
                    np.savez_compressed(
                        rollout_dir / f"step_{step:04d}.npz",
                        robot_state=robot_state_arr.copy(),
                        raw_action=raw_target.copy(),
                        sent_target=sent_target.copy(),
                        step=step,
                    )
                    # Save wrist image every 20 steps (to save disk)
                    if step % 20 == 0:
                        cv2.imwrite(str(rollout_dir / f"wrist_{step:04d}.jpg"),
                                    cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR))

                # ── Send to robot ──
                if not args.dry_run:
                    robot.set_joint_positions(sent_target.tolist(), velocity_pct=args.velocity_pct)

                # Update preview + handle pause/quit
                if not args.no_gui:
                    label = f"PAUSED {step+1}/{approach_steps}" if paused else f"APPROACH {step+1}/{approach_steps}"
                    color = (0, 165, 255) if paused else (0, 0, 255)
                    preview = build_preview(wrist_frame, global_frame, label, color=color)
                    cv2.imshow("ACT Deployment", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if should_quit(key):
                        user_quit = True
                        stop_reason = "user_quit"
                        break
                    if key == ord(' '):
                        paused = not paused
                        if paused:
                            print("  ⏸  PAUSED — SPACE to resume, Q to quit")
                        else:
                            print("  ▶  RESUMED")

                elapsed = time.time() - loop_start
                step_time = 1.0 / args.hz
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)

                # ── Pause loop ──
                while paused:
                    if not args.no_gui:
                        preview = build_preview(wrist_frame, global_frame,
                                                f"PAUSED {step+1}/{approach_steps}", color=(0, 165, 255))
                        cv2.imshow("ACT Deployment", preview)
                    if last_smoothed is not None and not args.dry_run:
                        hold_pos = np.concatenate([last_smoothed, [GRIPPER_OPEN]])
                        robot.set_joint_positions(hold_pos.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                    if not args.no_gui:
                        key = cv2.waitKey(1) & 0xFF
                        if should_quit(key):
                            paused = False
                            user_quit = True
                            stop_reason = "user_quit"
                            break
                        if key == ord(' '):
                            paused = False
                            print("  ▶  RESUMED")
                    else:
                        import select
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            line = sys.stdin.readline().strip().lower()
                            if line == 'q':
                                paused = False
                                user_quit = True
                                stop_reason = "user_quit"
                                break
                            if line == '':
                                paused = False
                                print("  ▶  RESUMED")
                if user_quit:
                    break

                last_smoothed = smoothed_arm.copy()
                last_state = np.asarray(robot_state, dtype=np.float32).copy()

            # ── Print raw action stats ──
            if raw_actions:
                ra = np.array(raw_actions)
                jnames = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]
                print(f"\n  Raw action stats over {len(raw_actions)} steps:")
                print(f"  {'Dim':>6}  {'mean':>12}  {'abs_mean':>12}  {'min':>12}  {'max':>12}")
                for d in range(ra.shape[1]):
                    print(f"  {jnames[d]:>6}  {ra[:, d].mean():12.6f}  "
                          f"{np.abs(ra[:, d]).mean():12.6f}  "
                          f"{ra[:, d].min():12.6f}  {ra[:, d].max():12.6f}")

            print(f"\n  ACT approach finished ({stop_reason}, {len(raw_actions)} steps).")

            # ================================================================
            #  HANDOVER: hold position
            # ================================================================
            if not user_quit and not args.dry_run:
                print("  >>> Handover: hold position (0.3s) ...")
                cur = robot.get_joint_positions()
                hold_start = time.time()
                while time.time() - hold_start < 0.3:
                    robot.set_joint_positions(cur, velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Hold complete.")

            # ================================================================
            #  TEST MODE A: approach only — done
            # ================================================================
            if args.test_mode == "A":
                if not user_quit:
                    print("  [TEST-A] Approach only — gripper stays open, no close/lift.")
                    print("  [TEST-A] Check: is gripper aligned with bottle at pre-grasp position?")
                    final_state = robot.get_joint_positions()
                    print(f"  [TEST-A] Final J2 = {final_state[1]:.5f} rad")

            # ================================================================
            #  TEST MODE C: approach → descend 2-3cm → stop (no close)
            # ================================================================
            if args.test_mode == "C" and not user_quit:
                print(f"\n  >>> [TEST-C] Descending (J2 += {args.descend_j2_delta:.3f} rad) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                descend_pose = cur.copy()
                descend_pose[1] += args.descend_j2_delta
                descend_pose[1] = np.clip(descend_pose[1], -3.14, 3.14)
                descend_pose[6] = GRIPPER_OPEN
                descend_path = interpolate_joint_path(cur, descend_pose,
                                                      max_step_rad=0.015, max_step_gripper=0.002)
                for di, dt in enumerate(descend_path):
                    t_start = time.time()
                    dt[6] = GRIPPER_OPEN
                    if not args.dry_run:
                        robot.set_joint_positions(dt.tolist(), velocity_pct=args.velocity_pct)
                    if di == 0 or di == len(descend_path) - 1:
                        print(f"    descend {di+1:3d}/{len(descend_path):3d}  {fmt_vec(dt, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                # Hold at descended position
                hold_start = time.time()
                print("  Holding descended position for 0.5s ...")
                while time.time() - hold_start < 0.5:
                    if not args.dry_run:
                        robot.set_joint_positions(descend_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                final = robot.get_joint_positions()
                print(f"  Descent complete. Final J2 = {final[1]:.5f} rad")
                print("  [TEST-C] Check: is gripper at correct grasp depth?")

            # ================================================================
            #  TEST MODE B: close gripper + lift
            # ================================================================
            if args.test_mode == "B" and not user_quit:
                # --- Close gripper ---
                print(f"\n  >>> [TEST-B] Closing gripper to {GRIPPER_CLOSE:.3f} m ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                close_pose = cur.copy()
                close_pose[6] = GRIPPER_CLOSE
                close_path = interpolate_joint_path(cur, close_pose,
                                                    max_step_rad=0.02, max_step_gripper=0.002)
                for ci, ct in enumerate(close_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(ct.tolist(), velocity_pct=args.velocity_pct)
                    if ci == 0 or ci == len(close_path) - 1:
                        print(f"    close {ci+1:3d}/{len(close_path):3d}  grip={ct[6]:.4f}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                # Hold closed
                hold_start = time.time()
                print("  Holding close for 0.6s ...")
                while time.time() - hold_start < 0.6:
                    if not args.dry_run:
                        robot.set_joint_positions(close_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Gripper closed.")

                # --- Lift: J3 -= 0.06 rad ---
                print("\n  >>> [TEST-B] Lifting (J3 -= 0.06 rad) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                lift_pose = cur.copy()
                lift_pose[2] -= 0.06
                lift_pose[2] = np.clip(lift_pose[2], -3.14, 3.14)
                lift_pose[6] = GRIPPER_CLOSE  # keep gripper closed
                lift_path = interpolate_joint_path(cur, lift_pose,
                                                   max_step_rad=0.02, max_step_gripper=0.002)
                for li, lt in enumerate(lift_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(lt.tolist(), velocity_pct=args.velocity_pct)
                    if li == 0 or li == len(lift_path) - 1:
                        print(f"    lift {li+1:3d}/{len(lift_path):3d}  {fmt_vec(lt, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                # Hold lift
                hold_start = time.time()
                print("  Holding lift for 0.5s ...")
                while time.time() - hold_start < 0.5:
                    if not args.dry_run:
                        robot.set_joint_positions(lift_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Lift complete.")
                print("  [TEST-B] Verify: is bottle grasped and lifted?")

            # ================================================================
            #  TEST MODE D: full grasp → place → release → return
            # ================================================================
            if args.test_mode == "D" and not user_quit:
                # --- Close gripper (same as Test B) ---
                print(f"\n  >>> [TEST-D] Closing gripper to {GRIPPER_CLOSE:.3f} m ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                close_pose = cur.copy()
                close_pose[6] = GRIPPER_CLOSE
                close_path = interpolate_joint_path(cur, close_pose,
                                                    max_step_rad=0.02, max_step_gripper=0.002)
                for ci, ct in enumerate(close_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(ct.tolist(), velocity_pct=args.velocity_pct)
                    if ci == 0 or ci == len(close_path) - 1:
                        print(f"    close {ci+1:3d}/{len(close_path):3d}  grip={ct[6]:.4f}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                hold_start = time.time()
                print("  Holding close for 0.6s ...")
                while time.time() - hold_start < 0.6:
                    if not args.dry_run:
                        robot.set_joint_positions(close_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Gripper closed.")

                # --- Lift: J3 -= 0.06 ---
                print("\n  >>> [TEST-D] Lifting (J3 -= 0.06 rad) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                lift_pose = cur.copy()
                lift_pose[2] -= 0.06
                lift_pose[2] = np.clip(lift_pose[2], -3.14, 3.14)
                lift_pose[6] = GRIPPER_CLOSE
                lift_path = interpolate_joint_path(cur, lift_pose,
                                                   max_step_rad=0.02, max_step_gripper=0.002)
                for li, lt in enumerate(lift_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(lt.tolist(), velocity_pct=args.velocity_pct)
                    if li == 0 or li == len(lift_path) - 1:
                        print(f"    lift {li+1:3d}/{len(lift_path):3d}  {fmt_vec(lt, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                hold_start = time.time()
                print("  Holding lift for 0.5s ...")
                while time.time() - hold_start < 0.5:
                    if not args.dry_run:
                        robot.set_joint_positions(lift_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Lift complete.")

                # --- Place: J1 offset to move bottle to side ---
                print(f"\n  >>> [TEST-D] Moving to side (J1 += {args.place_j1_offset:.2f} rad) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                place_pose = cur.copy()
                place_pose[0] += args.place_j1_offset
                place_pose[0] = np.clip(place_pose[0], -3.14, 3.14)
                place_pose[6] = GRIPPER_CLOSE
                place_path = interpolate_joint_path(cur, place_pose,
                                                    max_step_rad=0.03, max_step_gripper=0.002)
                for pi, pt in enumerate(place_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(pt.tolist(), velocity_pct=args.velocity_pct)
                    if pi == 0 or pi == len(place_path) - 1 or (pi + 1) % 5 == 0:
                        print(f"    place {pi+1:3d}/{len(place_path):3d}  {fmt_vec(pt, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                print("  Moved to side.")

                # --- Release: open gripper ---
                print(f"\n  >>> [TEST-D] Releasing gripper (open to {GRIPPER_OPEN:.3f} m) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                release_pose = cur.copy()
                release_pose[6] = GRIPPER_OPEN
                release_path = interpolate_joint_path(cur, release_pose,
                                                      max_step_rad=0.02, max_step_gripper=0.004)
                for ri, rt in enumerate(release_path):
                    t_start = time.time()
                    if not args.dry_run:
                        robot.set_joint_positions(rt.tolist(), velocity_pct=args.velocity_pct)
                    if ri == 0 or ri == len(release_path) - 1:
                        print(f"    release {ri+1:3d}/{len(release_path):3d}  grip={rt[6]:.4f}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                print("  Gripper released.")

                # Dwell after release
                hold_start = time.time()
                print("  Holding release for 0.5s ...")
                while time.time() - hold_start < 0.5:
                    if not args.dry_run:
                        robot.set_joint_positions(release_pose.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  [TEST-D] Full grasp + place + release complete.")

            # ================================================================
            #  RETURN TO START
            # ================================================================
            auto_return = (not args.no_return_to_start and not user_quit and not args.dry_run)
            if auto_return:
                start_pose = start_robot_state.copy()
                print("\n  >>> Returning to start position ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                start_pose[6] = cur[6]  # preserve current gripper
                path = interpolate_joint_path(cur, start_pose, max_step_rad=0.03, max_step_gripper=0.004)
                for ri, rt in enumerate(path):
                    t_start = time.time()
                    rt_clamped = rt.copy()
                    rt_clamped[:6] = np.clip(rt_clamped[:6], -3.14, 3.14)
                    rt_clamped[6] = np.clip(rt_clamped[6], 0.0, PIPER_GRIPPER_MAX_M)
                    if not args.dry_run:
                        robot.set_joint_positions(rt_clamped.tolist(), velocity_pct=args.velocity_pct)
                    if ri == 0 or ri == len(path) - 1 or (ri + 1) % 10 == 0:
                        print(f"    return {ri+1:3d}/{len(path):3d}  {fmt_vec(rt_clamped, 3)}")
                    elapsed = time.time() - t_start
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                print("  Returned to start position.")

            print("  Trajectory complete.")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        # Emergency stop: hold current position, do NOT disable
        try:
            cur = robot.get_joint_positions()
            robot.set_joint_positions(cur, velocity_pct=50)
        except Exception:
            pass
        print("  Stopped. Arm stays ENABLED at current position.")
        wrist_cam.close()
        if global_cam:
            global_cam.close()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
