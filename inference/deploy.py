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
import json
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
DEFAULT_RECORDED_START_QPOS = np.array(
    [-0.07682, 0.00623, -0.00392, 0.00000, 0.33034, 0.02376, 0.09950],
    dtype=np.float32,
)

# ── Approach-phase constants ──
GRIPPER_OPEN = 0.08          # gripper fully open (m)
GRIPPER_CLOSE = 0.0          # gripper fully closed (m)
# Per-joint max delta: J1-J3 arm joints get 0.03, J4-J6 wrist get 0.012
MAX_DELTA_PER_JOINT = np.array([0.03, 0.03, 0.03, 0.012, 0.012, 0.012], dtype=np.float32)
ACTION_SMOOTH_ALPHA = 0.5    # EMA smoothing factor
APPROACH_STEPS_DEFAULT = 200
WRIST_FREEZE_J2 = 1.45       # freeze J4-J6 when J2 exceeds this
READY_J2 = 1.65              # J2 threshold for ready_count
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


def to_jsonable(value):
    """Convert numpy scalars/arrays to plain Python objects for debug JSON."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def load_recorded_start_qpos(checkpt: str | None):
    """Load the first training frame qpos from checkpoint train_config.json."""
    if not checkpt:
        return None, "missing --checkpt"

    train_config_path = Path(checkpt) / "train_config.json"
    if not train_config_path.exists():
        return None, f"missing {train_config_path}"

    try:
        train_config = json.loads(train_config_path.read_text())
        dataset_cfg = train_config.get("dataset", {})
        dataset_root = Path(dataset_cfg.get("root", ""))
        if not dataset_root.is_absolute():
            dataset_root = PROJECT_ROOT / dataset_root
        episodes = dataset_cfg.get("episodes")

        import pandas as pd

        parquet_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_paths:
            return None, f"no parquet files under {dataset_root / 'data'}"

        df = pd.concat([pd.read_parquet(path) for path in parquet_paths], ignore_index=True)
        if episodes:
            first_episode = int(episodes[0])
            df = df[df["episode_index"] == first_episode]
        sort_cols = [col for col in ("episode_index", "frame_index", "index") if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        if df.empty:
            return None, f"no rows found in {dataset_root}"
        qpos = np.asarray(df.iloc[0]["observation.state"], dtype=np.float32).reshape(-1)[:7]
        if qpos.shape[0] != 7:
            return None, f"recorded start has shape {qpos.shape}, expected 7"
        return qpos, f"{dataset_root} first training frame"
    except Exception as exc:
        return None, f"failed to load recorded start: {exc}"


def resolve_recorded_start_qpos(checkpt: str | None):
    """Load recorded start qpos, falling back to the fixed-overfit known start."""
    qpos, source = load_recorded_start_qpos(checkpt)
    if qpos is not None:
        return qpos, source
    return DEFAULT_RECORDED_START_QPOS.copy(), f"fallback fixed-overfit start ({source})"


def check_start_reset(current_qpos: np.ndarray, expected_qpos: np.ndarray,
                      qpos_tol: float, gripper_tol: float) -> bool:
    """Return True when all joints and gripper are close to the recorded start."""
    diff = np.abs(current_qpos - expected_qpos)
    arm_ok = bool(np.all(diff[:6] <= qpos_tol))
    gripper_ok = bool(diff[6] <= gripper_tol)
    if arm_ok and gripper_ok:
        print("  Reset guard: PASS")
        return True

    print("\n  [RESET-GUARD] Refusing to start rollout: current state is not clean.")
    print(f"    current qpos : {fmt_vec(current_qpos, 5)}")
    print(f"    expected qpos: {fmt_vec(expected_qpos, 5)}")
    print(f"    abs diff     : {fmt_vec(diff, 5)}")
    print(f"    arm tol={qpos_tol:.4f} rad  gripper tol={gripper_tol:.4f} m")
    print("    Please reset / return-to-start first, then run the rollout again.")
    return False


def move_robot_to_recorded_start(
    robot,
    target_qpos: np.ndarray,
    velocity_pct: int,
    hz: float,
    max_delta: np.ndarray,
    action_smooth: float,
    qpos_tol: float,
    gripper_tol: float,
    max_steps: int = 300,
    max_step_gripper: float = 0.004,
):
    """Safely move robot to recorded training start using deploy-style limits."""
    target_qpos = np.asarray(target_qpos, dtype=np.float32).copy()
    target_qpos[:6] = np.clip(target_qpos[:6], -3.14, 3.14)
    target_qpos[6] = np.clip(target_qpos[6], 0.0, PIPER_GRIPPER_MAX_M)

    print("\n  >>> Resetting to recorded training start ...")
    print(f"    target qpos: {fmt_vec(target_qpos, 5)}")
    print(f"    max_delta arm: {fmt_vec(max_delta, 3)}  max_step_gripper={max_step_gripper:.4f}")

    last_smoothed_arm = None
    final_qpos = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    for step in range(max_steps):
        cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
        diff = target_qpos - cur
        arm_max_diff = float(np.max(np.abs(diff[:6])))
        gripper_diff = float(abs(diff[6]))
        if arm_max_diff <= qpos_tol and gripper_diff <= gripper_tol:
            final_qpos = cur
            print(f"    reset reached tolerance at step {step}")
            break

        clipped_delta = diff.copy()
        for j in range(6):
            clipped_delta[j] = np.clip(clipped_delta[j], -max_delta[j], max_delta[j])
        clipped_delta[6] = np.clip(clipped_delta[6], -max_step_gripper, max_step_gripper)

        clipped_target = cur + clipped_delta
        if last_smoothed_arm is not None and action_smooth > 0:
            smoothed_arm = action_smooth * clipped_target[:6] + (1.0 - action_smooth) * last_smoothed_arm
        else:
            smoothed_arm = clipped_target[:6].copy()
        sent_target = np.concatenate([smoothed_arm, [clipped_target[6]]])
        sent_target[:6] = np.clip(sent_target[:6], -3.14, 3.14)
        sent_target[6] = np.clip(sent_target[6], 0.0, PIPER_GRIPPER_MAX_M)

        robot.set_joint_positions(sent_target.tolist(), velocity_pct=velocity_pct)
        last_smoothed_arm = smoothed_arm.copy()

        if step == 0 or (step + 1) % 10 == 0 or step == max_steps - 1:
            print(
                f"    reset {step+1:03d}/{max_steps}: "
                f"arm_max_diff={arm_max_diff:.5f}  gripper_diff={gripper_diff:.5f}  "
                f"sent={fmt_vec(sent_target, 4)}"
            )

        step_time = 1.0 / hz
        time.sleep(step_time)
    else:
        final_qpos = np.asarray(robot.get_joint_positions(), dtype=np.float32)
        print(f"    [WARN] Reset hit max_steps={max_steps} before reaching tolerance.")

    # Hold and let the arm settle on the exact target command.
    hold_start = time.time()
    while time.time() - hold_start < 0.4:
        robot.set_joint_positions(target_qpos.tolist(), velocity_pct=velocity_pct)
        time.sleep(1.0 / hz)

    final_qpos = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    diff = np.abs(final_qpos - target_qpos)
    print("\n  Reset validation:")
    print(f"    final qpos   : {fmt_vec(final_qpos, 5)}")
    print(f"    expected qpos: {fmt_vec(target_qpos, 5)}")
    print(f"    abs diff     : {fmt_vec(diff, 5)}")
    print(f"    arm max diff : {float(np.max(diff[:6])):.5f}")
    print(f"    gripper diff : {float(diff[6]):.5f}")
    passed = check_start_reset(final_qpos, target_qpos, qpos_tol, gripper_tol)
    return passed, final_qpos, diff


def open_gripper_to_value(
    robot,
    target_gripper: float,
    timeout: float = 5.0,
    tol: float = 0.01,
    hz: float = 50.0,
):
    """Command gripper to a target value and wait for feedback to converge.

    Returns (success, current_before, current_after, diff, elapsed).
    """
    current_before = float(robot.get_joint_positions()[6])
    print(f"\n  [GRIPPER-RESET] Opening gripper ...")
    print(f"    recorded_start_gripper = {target_gripper:.5f} m")
    print(f"    current_before         = {current_before:.5f} m")
    print(f"    target                 = {target_gripper:.5f} m")
    print(f"    timeout                = {timeout:.1f} s")
    print(f"    tol                    = {tol:.4f} m")

    target_gripper = float(np.clip(target_gripper, 0.0, PIPER_GRIPPER_MAX_M))
    start_time = time.time()
    step_time = 1.0 / hz

    # Send open command immediately (gripper-only, arm joints hold current)
    cur = robot.get_joint_positions()
    arm_hold = cur[:6]
    robot.set_joint_positions(list(arm_hold) + [float(target_gripper)], velocity_pct=30)

    current_after = current_before
    while time.time() - start_time < timeout:
        time.sleep(step_time)
        current_after = float(robot.get_joint_positions()[6])
        diff = abs(current_after - target_gripper)
        if diff <= tol:
            elapsed = time.time() - start_time
            print(f"    [GRIPPER-RESET] SUCCESS  current_after={current_after:.5f}  diff={diff:.5f}  elapsed={elapsed:.2f}s")
            return True, current_before, current_after, diff, elapsed

    elapsed = time.time() - start_time
    diff = abs(current_after - target_gripper)
    print(f"    [GRIPPER-RESET] TIMEOUT  current_after={current_after:.5f}  diff={diff:.5f}  timeout={elapsed:.2f}s")
    return False, current_before, current_after, diff, elapsed


def save_camera_png(frame, out_path: Path, camera_label: str) -> bool:
    if frame is None or getattr(frame, "rgb", None) is None:
        print(f"  [WARN] Cannot save {camera_label}: frame unavailable")
        return False
    try:
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(out_path), bgr)
    except Exception as exc:
        print(f"  [WARN] Cannot save {camera_label} to {out_path}: {exc}")
        return False
    if not ok:
        print(f"  [WARN] cv2.imwrite failed for {camera_label}: {out_path}")
        return False
    return True


def save_alignment_images(rollout_dir: Path, label: str, wrist_frame, global_frame) -> None:
    save_camera_png(wrist_frame, rollout_dir / f"{label}_realsense.png", "RealSense")
    if global_frame is None:
        print(f"  [WARN] Cannot save USB image for {label}: global camera unavailable")
    else:
        save_camera_png(global_frame, rollout_dir / f"{label}_usb.png", "USB camera")


def read_final_camera_frames(wrist_cam, global_cam):
    wrist_frame = None
    global_frame = None
    try:
        wrist_frame = wrist_cam.read()
    except Exception as exc:
        print(f"  [WARN] Final RealSense read failed: {exc}")
    if global_cam is not None:
        try:
            global_frame = global_cam.read()
        except Exception as exc:
            print(f"  [WARN] Final USB camera read failed: {exc}")
    return wrist_frame, global_frame


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


# ── Full-E2E Phase Detection (for act-full staged stops) ──

def get_gripper_phase(pred_grip: float, grip_open: float, grip_close: float,
                      close_detected: bool, release_detected: bool) -> str:
    """Classify current gripper prediction phase using dynamic midpoint."""
    grip_mid = (grip_open + grip_close) / 2.0
    if not close_detected:
        if pred_grip > grip_mid:
            return "open"
        else:
            return "closing"
    elif not release_detected:
        if pred_grip <= grip_mid:
            return "closed"
        else:
            return "releasing"
    else:
        return "released"


def check_phase_stop(stop_after: str, step: int,
                     grip_pred_history: list, grip_open: float, grip_close: float,
                     close_detected: bool, release_detected: bool,
                     close_step: int, release_step: int,
                     consecutive_close_count: int, consecutive_release_count: int,
                     lift_steps_after_close: int):
    """Determine whether to stop based on phase and stop_after mode.

    Returns (should_stop: bool, stop_reason: str).
    """
    grip_mid = (grip_open + grip_close) / 2.0

    if stop_after == "approach":
        # Stop BEFORE gripper closes: when grip first drops below grip_mid
        if not close_detected and len(grip_pred_history) >= 3:
            recent = grip_pred_history[-3:]
            if all(g < grip_mid for g in recent):
                return True, "approach_stop (gripper about to close)"

    elif stop_after == "close":
        # Stop after close detected + sustained for 3 frames
        if close_detected and consecutive_close_count >= 3:
            return True, "close_stop (close sustained 3 frames)"

    elif stop_after == "lift":
        # Stop after close + 30 more steps
        if close_detected and step - close_step >= 30:
            return True, "lift_stop (close + 30 steps)"

    elif stop_after == "release":
        # Stop after release detected + sustained for 3 frames
        if release_detected and consecutive_release_count >= 3:
            return True, "release_stop (release sustained 3 frames)"

    # "full": never early stop

    return False, ""


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
    parser.add_argument("--test-mode", choices=("A", "B", "C", "D", "full-e2e"), default="A",
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
    parser.add_argument("--no-auto-return", action="store_true",
                        help="Alias for --no-return-to-start, intended for alignment debugging.")
    parser.add_argument("--hold-after-stop", type=float, default=0.0,
                        help="Hold current pose for N seconds after approach/close stop when auto-return is disabled.")
    parser.add_argument("--debug-policy-io", action="store_true",
                        help="Print policy input/output at every debug-every step: "
                        "raw robot_state[J2], obs.state[J2], normalized_state[J2], raw_action[J2].")
    parser.add_argument("--save-rollout", action="store_true",
                        help="Save rollout frames (image, state, action) to disk for offline analysis.")
    parser.add_argument("--save-final-images", action="store_true",
                        help="Save approach alignment camera images and JSON to the rollout directory.")
    parser.add_argument("--debug-offline-policy-rollout-from-recorded-start", action="store_true",
                        help="[act-full debug] Run policy offline from recorded start qpos + first frame, "
                             "print predicted chunk sequences (J2, J3, grip), and exit. No robot connection.")
    parser.add_argument("--offline-debug-num-chunks", type=int, default=8,
                        help="Number of consecutive chunks for --debug-offline-policy-rollout-from-recorded-start (default: 8).")
    parser.add_argument("--debug-recorded-demo-gripper", action="store_true",
                        help="[debug] Inspect recorded dataset gripper trajectory and exit. No robot connection.")
    parser.add_argument("--enforce-start-reset", action="store_true",
                        help="Refuse rollout unless current qpos/gripper match the recorded training start.")
    parser.add_argument("--reset-to-recorded-start", action="store_true",
                        help="Move to the recorded training start, validate reset guard, then exit before rollout.")
    parser.add_argument("--start-qpos-tol", type=float, default=0.05,
                        help="Arm joint tolerance for --enforce-start-reset.")
    parser.add_argument("--start-gripper-tol", type=float, default=0.01,
                        help="Gripper tolerance for --enforce-start-reset.")
    parser.add_argument("--open-gripper-on-start", action="store_true",
                        help="Open gripper to a target value after robot connect, before policy rollout.")
    parser.add_argument("--gripper-start-open-value", type=float, default=None,
                        help="Target gripper opening value (m) for --open-gripper-on-start. "
                             "Default: recorded_start_gripper from checkpoint, or 0.09950.")
    parser.add_argument("--gripper-open-timeout", type=float, default=5.0,
                        help="Max seconds to wait for gripper to reach target (default: 5.0).")
    parser.add_argument("--gripper-open-tol", type=float, default=0.01,
                        help="Gripper feedback tolerance for open-gripper (default: 0.01 m).")
    parser.add_argument("--policy-type", choices=("act", "act-full", "hybrid", "hybrid-v3", "hybrid-v4-delta"), default="act",
                        help="Policy architecture: act, act-full (7D incl. gripper), hybrid, hybrid-v3, or hybrid-v4-delta.")
    parser.add_argument("--hybrid-checkpt", type=str, default=None,
                        help="Path to hybrid policy .pt checkpoint (for --policy-type hybrid).")
    parser.add_argument("--clamp-delta-j2-nonnegative", action="store_true",
                        help="[hybrid-v3 only] Clamp image_delta[J2] >= 0 before combining with base_action. "
                        "Prevents image_delta from cancelling state_head forward push.")
    parser.add_argument("--max-joint-delta", type=float, default=None,
                        help="Override per-joint max delta for all arm joints. 0=disabled. Default uses per-joint limits.")
    parser.add_argument("--wrist-freeze-j2", type=float, default=1.45,
                        help="J2 threshold to freeze J4-J6 wrist joints (default: 1.45).")
    parser.add_argument("--ready-j2", type=float, default=1.65,
                        help="J2 threshold for ready_count (default: 1.65).")
    parser.add_argument("--ready-count-min", type=int, default=5,
                        help="Consecutive steps above READY_J2 to trigger stop (default: 5).")
    parser.add_argument("--v4-j2-only", action="store_true",
                        help="[hybrid-v4-delta] Use v4 delta only for J2; other joints from v2 (ver A) or position-hold (ver B).")
    parser.add_argument("--v4-j2-only-ver", choices=("A", "B"), default="A",
                        help="Version A: J1/J3/J4/J5/J6 from v2 baseline. Version B: hold current position (default: A).")
    parser.add_argument("--v2-checkpt", type=str, default="outputs/train/hybrid_v2.pt",
                        help="Path to hybrid v2 checkpoint for --v4-j2-only --v4-j2-only-ver A.")
    parser.add_argument("--allow-real-full-e2e", action="store_true",
                        help="[act-full] Explicitly permit real-robot full trajectory execution. "
                             "Without this flag, act-full only runs in --dry-run mode.")
    parser.add_argument("--full-e2e-stop-after", choices=("approach", "close", "lift", "release", "full"),
                        default="approach",
                        help="[act-full + full-e2e] Stop trajectory after which phase. "
                             "approach: before gripper closes. close: after close detected. "
                             "lift: close + 30 steps. release: after release detected. "
                             "full: no early stop (complete trajectory). Default: approach.")
    parser.add_argument("--act-full-chunk-exec", choices=("normal", "hold_last_until_ready", "target_reached"),
                        default="normal",
                        help="[act-full + full-e2e] Chunk execution mode. "
                             "normal: standard replan on each inference. "
                             "hold_last_until_ready: after chunk consumed, hold last action as target "
                             "until J2 reaches READY_J2. "
                             "target_reached: advance chunk index only when robot reaches current target. "
                             "Default: normal.")
    parser.add_argument("--act-full-target-tol", type=float, default=0.04,
                        help="[act-full + target_reached] Max arm error (rad) to consider a target reached.")
    parser.add_argument("--act-full-target-max-hold", type=int, default=20,
                        help="[act-full + target_reached] Max control steps to hold same target before forcing advance.")
    parser.add_argument("--target-reached-ignore-frozen-wrist", action="store_true", default=True,
                        help="[act-full + target_reached] Exclude frozen wrist joints (J4-J6) from arm_error "
                             "when wrist_freeze is active. Prevents wrist freeze from blocking chunk_idx advance. "
                             "Default: True.")
    parser.add_argument("--gripper-close-detect-mode", choices=("absolute", "relative", "dataset"), default="relative",
                        help="[act-full] Gripper close detection mode. "
                             "absolute: grip < (open+close)/2. "
                             "relative: grip drops below recorded_start_gripper - close_drop. "
                             "dataset: thresholds derived from recorded demo. "
                             "Default: relative.")
    parser.add_argument("--gripper-close-drop", type=float, default=0.015,
                        help="[act-full + relative mode] Min grip drop from recorded start to trigger close onset (default: 0.015 m).")
    parser.add_argument("--gripper-close-onset-threshold", type=float, default=0.085,
                        help="[act-full] Absolute grip value below which close onset is triggered. "
                             "Default: 0.085 (recorded_start_gripper - 0.015).")
    parser.add_argument("--gripper-close-strong-threshold", type=float, default=0.055,
                        help="[act-full] Absolute grip value below which close is considered strong/complete. "
                             "Default: 0.055 (dataset min + 0.01).")
    args = parser.parse_args()

    if args.no_auto_return:
        args.no_return_to_start = True

    # Apply command-line overrides to shared constants
    WRIST_FREEZE_J2 = args.wrist_freeze_j2
    READY_J2 = args.ready_j2
    READY_COUNT_MIN = args.ready_count_min

    # Validate act-full safety: dry-run only unless explicitly allowed.
    # Offline debug mode does not use robot, so skip this check.
    is_act_full = (args.policy_type == "act-full")
    if (is_act_full and not args.dry_run and not args.allow_real_full_e2e
            and not args.reset_to_recorded_start
            and not args.debug_offline_policy_rollout_from_recorded_start):
        parser.error(
            "--policy-type act-full requires --dry-run for safety. "
            "To run on real robot, add --allow-real-full-e2e explicitly."
        )

    # Validate checkpoint argument
    if args.policy_type in ("hybrid", "hybrid-v3", "hybrid-v4-delta") and not args.hybrid_checkpt:
        parser.error("--hybrid-checkpt is required when --policy-type hybrid, hybrid-v3, or hybrid-v4-delta")
    if args.policy_type in ("act", "act-full") and not args.checkpt:
        parser.error("--checkpt is required when --policy-type act or act-full")
    if (args.enforce_start_reset or args.reset_to_recorded_start) and not args.checkpt:
        parser.error("--enforce-start-reset / --reset-to-recorded-start requires --checkpt")

    recorded_start_qpos = None
    recorded_start_source = ""
    if is_act_full or args.enforce_start_reset or args.reset_to_recorded_start:
        recorded_start_qpos, recorded_start_source = resolve_recorded_start_qpos(args.checkpt)
    act_full_gripper_open = GRIPPER_OPEN
    if is_act_full:
        if recorded_start_qpos is not None:
            act_full_gripper_open = float(
                np.clip(max(GRIPPER_OPEN, recorded_start_qpos[6]), 0.0, PIPER_GRIPPER_MAX_M)
            )
        else:
            act_full_gripper_open = PIPER_GRIPPER_MAX_M

    # ── Gripper close detection thresholds ──
    _rec_start_grip = float(recorded_start_qpos[6]) if recorded_start_qpos is not None else 0.09950
    _grip_mid_absolute = (act_full_gripper_open + GRIPPER_CLOSE) / 2.0

    if args.gripper_close_onset_threshold is not None:
        _close_onset_threshold = float(args.gripper_close_onset_threshold)
    elif args.gripper_close_detect_mode == "absolute":
        _close_onset_threshold = _grip_mid_absolute
    else:
        # relative or dataset: onset = recorded_start_grip - close_drop
        _close_onset_threshold = float(np.clip(_rec_start_grip - args.gripper_close_drop, 0.0, PIPER_GRIPPER_MAX_M))

    if args.gripper_close_strong_threshold is not None:
        _close_strong_threshold = float(args.gripper_close_strong_threshold)
    elif args.gripper_close_detect_mode == "absolute":
        _close_strong_threshold = _grip_mid_absolute
    else:
        # relative/dataset: strong = mid between onset and close
        _close_strong_threshold = max(_close_onset_threshold / 2.0, 0.02)

    _close_detect_onset_count = 3   # consecutive steps below onset to latch
    _close_detect_strong_count = 3  # consecutive steps below strong to confirm

    if is_act_full:
        print(f"  Close detect mode: {args.gripper_close_detect_mode}")
        print(f"    recorded_start_gripper = {_rec_start_grip:.5f}")
        print(f"    close_onset_threshold  = {_close_onset_threshold:.5f}")
        print(f"    close_strong_threshold = {_close_strong_threshold:.5f}")
        print(f"    absolute_midpoint      = {_grip_mid_absolute:.5f}")

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
    if is_act_full:
        print(f"\n[1/4] Loading ACT full-trajectory policy from {args.checkpt} ...")
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
        is_v4_delta = False
        force_gripper_open = False  # act-full: model controls gripper
        print(
            f"  ACT-full policy loaded: {sum(p.numel() for p in policy.parameters()):,} params  "
            f"chunk_size={chunk_size}  n_action_steps={n_action_steps}  "
            f"state_dim={expected_state_dim}"
        )

        # --- Load pre/post processors ---
        print("\n[2/4] Loading pre/post processors from checkpoint ...")
        preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
        print("  Processors ready.")
        hybrid_state_mean = hybrid_state_std = None
        hybrid_action_mean = hybrid_action_std = None
        hybrid_delta_mean = hybrid_delta_std = None
        _v3_internals = None
        v2_model = None
    elif args.policy_type in ("hybrid", "hybrid-v3", "hybrid-v4-delta"):
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
    # Set gripper control mode
    force_gripper_open = (args.policy_type != "act-full")

    # ── Dataset gripper inspection ──
    if args.debug_recorded_demo_gripper:
        if recorded_start_qpos is None:
            print("  [ERROR] Recorded start qpos unavailable; cannot inspect dataset.")
            return 1
        import pandas as pd
        from pathlib import Path as _Path
        dataset_root = _Path(recorded_start_source.split(" first training frame")[0]) if "first training frame" in recorded_start_source else None
        if dataset_root is None or not dataset_root.exists():
            print(f"  [ERROR] Cannot resolve dataset root from: {recorded_start_source}")
            return 1
        parquet_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_paths:
            print(f"  [ERROR] No parquet files under {dataset_root / 'data'}")
            return 1
        df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
        sort_cols = [col for col in ("episode_index", "frame_index", "index") if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        qpos_data = np.stack([np.asarray(r["observation.state"], dtype=np.float32).reshape(-1)[:7]
                              for _, r in df.iterrows()])
        action_data = np.stack([np.asarray(r["action"], dtype=np.float32).reshape(-1)[:7]
                               for _, r in df.iterrows()])
        n_frames = len(qpos_data)

        print("\n" + "=" * 60)
        print("  [DATASET GRIPPER INSPECTION]")
        print("=" * 60)
        print(f"  Dataset: {dataset_root}")
        print(f"  Frames: {n_frames}")
        print(f"  Qpos gripper:  min={float(np.min(qpos_data[:, 6])):.5f}  max={float(np.max(qpos_data[:, 6])):.5f}  range={float(np.max(qpos_data[:, 6]) - np.min(qpos_data[:, 6])):.5f}")
        print(f"  Action gripper: min={float(np.min(action_data[:, 6])):.5f}  max={float(np.max(action_data[:, 6])):.5f}")

        # Gripper trajectory analysis
        grip_vals = qpos_data[:, 6]
        grip_deltas = action_data[:, 6]
        grip_max = float(np.max(grip_vals))

        # Find first grip decrease > 0.01
        first_drop_idx = -1
        for i in range(1, n_frames):
            if grip_vals[i] < grip_max - 0.01:
                first_drop_idx = i
                break
        if first_drop_idx >= 0:
            print(f"  First grip drop:  frame={first_drop_idx}  grip={grip_vals[first_drop_idx]:.5f}  "
                  f"J2={qpos_data[first_drop_idx, 1]:.4f}  action_grip={grip_deltas[first_drop_idx]:.5f}")

        # Find minimum gripper value
        min_grip_idx = int(np.argmin(grip_vals))
        print(f"  Minimum grip:     frame={min_grip_idx}  grip={grip_vals[min_grip_idx]:.5f}  "
              f"J2={qpos_data[min_grip_idx, 1]:.4f}")

        # Gripper progression
        print(f"\n  Gripper progression (every 10% of frames):")
        for pct in range(0, 101, 10):
            idx = min(int(n_frames * pct / 100), n_frames - 1)
            print(f"    {pct:3d}%  frame={idx:4d}  grip={grip_vals[idx]:.5f}  J2={qpos_data[idx, 1]:.4f}  J3={qpos_data[idx, 2]:.4f}")

        # Close phase analysis
        grip_mid_open_close = (grip_max + float(np.min(grip_vals))) / 2.0
        grip_onset_val = grip_max - 0.015
        print(f"\n  Threshold analysis:")
        print(f"    Open gripper         : {grip_max:.5f}")
        print(f"    Closed gripper (min) : {float(np.min(grip_vals)):.5f}")
        print(f"    Midpoint             : {grip_mid_open_close:.5f}")
        print(f"    Onset (open - 0.015) : {grip_onset_val:.5f}")
        print(f"    Suggested close_onset_threshold : {grip_onset_val:.5f}  (open - 0.015)")
        print(f"    Suggested strong_close_threshold: {float(np.min(grip_vals)) + 0.01:.5f}  (min + 0.01)")
        print(f"    Suggested close_detect_midpoint  : {grip_mid_open_close:.5f}  (open+close)/2")

        # Find all frames where gripper < onset
        onset_frames = np.where(grip_vals < grip_onset_val)[0]
        if len(onset_frames) > 0:
            print(f"    First onset frame    : {onset_frames[0]}  grip={grip_vals[onset_frames[0]]:.5f}")
            print(f"    Sustained close from : frame {onset_frames[0]} to {onset_frames[-1]} ({len(onset_frames)} frames)")

        print("\n  Dataset inspection complete. Exiting without robot connection.")
        return 0

    # ── Offline policy rollout debug: run policy on recorded start frame, print chunk, exit ──
    if args.debug_offline_policy_rollout_from_recorded_start:
        if not is_act_full:
            print("  [ERROR] --debug-offline-policy-rollout-from-recorded-start requires --policy-type act-full")
            return 1
        if recorded_start_qpos is None:
            print("  [ERROR] Recorded start qpos unavailable; cannot run offline debug.")
            return 1
        num_chunks = max(1, args.offline_debug_num_chunks)
        print("\n" + "=" * 60)
        print(f"  [OFFLINE-DEBUG] Policy rollout from recorded start ({num_chunks} chunks)")
        print("=" * 60)
        print(f"  Recorded start qpos: {fmt_vec(recorded_start_qpos, 5)}")
        print(f"  Source: {recorded_start_source}")

        # Load first training frame images
        import pandas as pd
        from pathlib import Path as _Path
        dataset_root = _Path(recorded_start_source.split(" first training frame")[0]) if "first training frame" in recorded_start_source else None
        if dataset_root is None or not dataset_root.exists():
            print(f"  [ERROR] Cannot resolve dataset root from source: {recorded_start_source}")
            return 1
        parquet_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_paths:
            print(f"  [ERROR] No parquet files under {dataset_root / 'data'}")
            return 1
        df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
        sort_cols = [col for col in ("episode_index", "frame_index", "index") if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        first_row = df.iloc[0]
        first_qpos = np.asarray(first_row["observation.state"], dtype=np.float32).reshape(-1)[:7]
        print(f"  First frame qpos:    {fmt_vec(first_qpos, 5)}")

        # Also extract full recorded gripper trajectory for reference
        _rec_qpos = np.stack([np.asarray(r["observation.state"], dtype=np.float32).reshape(-1)[:7]
                              for _, r in df.iterrows()])
        _rec_grip = _rec_qpos[:, 6]
        _rec_grip_min = float(np.min(_rec_grip))
        _rec_grip_max = float(np.max(_rec_grip))
        print(f"  Recorded grip range: [{_rec_grip_min:.5f}, {_rec_grip_max:.5f}]")
        _rec_grip_drop_idx = int(np.argmax(_rec_grip < _rec_grip_max - 0.01)) if np.any(_rec_grip < _rec_grip_max - 0.01) else -1
        if _rec_grip_drop_idx >= 0:
            print(f"  Recorded grip first drop: frame={_rec_grip_drop_idx}  grip={_rec_grip[_rec_grip_drop_idx]:.5f}  J2={_rec_qpos[_rec_grip_drop_idx, 1]:.4f}")

        # Load first frame images (LeRobot stores images as MP4 videos, often AV1-encoded)
        import av
        wrist_img_np = None
        global_img_np = None
        video_root = dataset_root / "videos"
        for img_key, attr_name in [("observation.images.wrist_rgb", "wrist_img_np"),
                                    ("observation.images.global_rgb", "global_img_np")]:
            video_dir = video_root / img_key
            if video_dir.exists():
                mp4_paths = sorted(video_dir.glob("chunk-*/file-*.mp4"))
                if mp4_paths:
                    try:
                        container = av.open(str(mp4_paths[0]))
                        for frame in container.decode(video=0):
                            img = frame.to_ndarray(format="rgb24")
                            if attr_name == "wrist_img_np":
                                wrist_img_np = img
                            else:
                                global_img_np = img
                            break
                        container.close()
                        if (attr_name == "wrist_img_np" and wrist_img_np is not None) or \
                           (attr_name == "global_img_np" and global_img_np is not None):
                            print(f"  Loaded {img_key}: {mp4_paths[0].name} shape={img.shape}")
                        else:
                            print(f"  [WARN] No frames decoded from {mp4_paths[0]}")
                    except Exception as exc:
                        print(f"  [WARN] Failed to decode {mp4_paths[0]}: {exc}")
        if wrist_img_np is None:
            # Fallback: try image directory (PNG frames)
            img_root = dataset_root / "images"
            for img_key, attr_name in [("observation.images.wrist_rgb", "wrist_img_np"),
                                        ("observation.images.global_rgb", "global_img_np")]:
                img_dir = img_root / img_key
                if img_dir.exists():
                    png_paths = sorted(img_dir.glob("chunk-*/file-*.png"))
                    if png_paths:
                        frame = cv2.imread(str(png_paths[0]))
                        if frame is not None:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            if attr_name == "wrist_img_np":
                                wrist_img_np = frame_rgb
                            else:
                                global_img_np = frame_rgb
                            print(f"  Loaded {img_key}: {png_paths[0].name} shape={frame_rgb.shape}")
        if wrist_img_np is None:
            print("  [ERROR] No wrist image found (tried av, cv2, videos/ and images/ directories)")
            return 1

        # ── Multi-chunk rollout ──
        _current_qpos = first_qpos.copy()
        _global_min_grip = 1.0
        _global_min_chunk = -1
        _global_min_step = -1
        _global_min_j2 = 0.0
        _first_onset_chunk = -1
        _first_onset_step = -1
        _first_strong_chunk = -1
        _first_strong_step = -1
        _thresholds = [0.09, 0.08, 0.07, 0.06, 0.05]
        _onset_threshold = 0.09

        print(f"\n  {'Chunk':>6} | {'J2 min':>8} {'J2 max':>8} | {'Grip min':>10} {'Grip max':>10} | {'First grip':>10} {'Last grip':>10} | {'<0.09':>6} {'<0.08':>6} {'<0.07':>6} {'<0.06':>6} {'<0.05':>6}")
        print(f"  {'─'*6}─┼─{'─'*8}─{'─'*8}─┼─{'─'*10}─{'─'*10}─┼─{'─'*10}─{'─'*10}─┼─{'─'*6}─{'─'*6}─{'─'*6}─{'─'*6}─{'─'*6}")

        _step_offset = 0
        for ci in range(num_chunks):
            _phase = min(1.0, ci / max(1, num_chunks - 1)) if num_chunks > 1 else 0.0
            obs = prepare_observation(
                _current_qpos.tolist(), wrist_img_np, global_img_np, device, expected_state_dim, _phase,
                gripper_unit_scale=args.gripper_unit_scale,
            )
            with torch.inference_mode():
                normalized_obs = preprocessor(obs)
                action_chunk = policy.predict_action_chunk(normalized_obs)
                action_chunk = postprocessor(action_chunk)
                chunk = action_chunk.squeeze(0).cpu().numpy()

            _j2_min = float(np.min(chunk[:, 1]))
            _j2_max = float(np.max(chunk[:, 1]))
            _g_min = float(np.min(chunk[:, 6]))
            _g_max = float(np.max(chunk[:, 6]))
            _g_first = float(chunk[0, 6])
            _g_last = float(chunk[-1, 6])

            # Track global min
            if _g_min < _global_min_grip:
                _global_min_grip = _g_min
                _global_min_chunk = ci
                _global_min_step = _step_offset + int(np.argmin(chunk[:, 6]))
                _global_min_j2 = float(chunk[int(np.argmin(chunk[:, 6])), 1])

            # First onset (grip < 0.09)
            _onset_idx = np.where(chunk[:, 6] < _onset_threshold)[0]
            if len(_onset_idx) > 0 and _first_onset_chunk < 0:
                _first_onset_chunk = ci
                _first_onset_step = _step_offset + int(_onset_idx[0])

            # First strong close (grip < 0.05)
            _strong_idx = np.where(chunk[:, 6] < 0.05)[0]
            if len(_strong_idx) > 0 and _first_strong_chunk < 0:
                _first_strong_chunk = ci
                _first_strong_step = _step_offset + int(_strong_idx[0])

            # Threshold counts
            _thresh_hits = []
            for th in _thresholds:
                _hits = np.where(chunk[:, 6] < th)[0]
                _thresh_hits.append(str(len(_hits)) if len(_hits) == 0 else str(int(_hits[0])))

            print(f"  {ci:6d} | {_j2_min:8.3f} {_j2_max:8.3f} | {_g_min:10.5f} {_g_max:10.5f} | {_g_first:10.5f} {_g_last:10.5f} | {_thresh_hits[0]:>6} {_thresh_hits[1]:>6} {_thresh_hits[2]:>6} {_thresh_hits[3]:>6} {_thresh_hits[4]:>6}")

            # Print grip sequence for last few chunks (where close might happen)
            if ci >= num_chunks - 3:
                print(f"          J2  seq: {fmt_vec(chunk[:, 1], 3)}")
                print(f"          Grip seq: {fmt_vec(chunk[:, 6], 4)}")

            _current_qpos = chunk[-1].copy()
            _step_offset += len(chunk)

        # ── Summary ──
        print(f"\n  {'─'*50}")
        print(f"  [OFFLINE SUMMARY]")
        print(f"  Global min grip        : {_global_min_grip:.5f}  (chunk {_global_min_chunk}, step {_global_min_step}, J2={_global_min_j2:.4f})")
        print(f"  First onset (<{_onset_threshold:.2f}) : {'chunk ' + str(_first_onset_chunk) + ', step ' + str(_first_onset_step) if _first_onset_chunk >= 0 else 'NEVER'}")
        print(f"  First strong (<0.05)   : {'chunk ' + str(_first_strong_chunk) + ', step ' + str(_first_strong_step) if _first_strong_chunk >= 0 else 'NEVER'}")
        print(f"  Policy {'DOES' if _global_min_grip < 0.05 else 'DOES NOT'} predict strong close in {num_chunks} chunks")
        print(f"  Suggested close thresholds based on offline rollout:")
        if _global_min_grip < 0.05:
            print(f"    close_onset  = 0.09  (grip begins to drop)")
            print(f"    strong_close = 0.05")
        elif _global_min_grip < 0.08:
            print(f"    close_onset  = 0.09  (grip begins to drop)")
            print(f"    strong_close = {_global_min_grip + 0.01:.2f}  (near observed min + margin)")
        else:
            print(f"    [WARN] Policy barely predicts grip drop. Min grip = {_global_min_grip:.5f}")
            print(f"    Model may not have learned close phase. Check training data / checkpoint.")

        print("\n  Offline debug complete. Exiting without robot connection.")
        return 0

    step_n = 2 if args.policy_type in ("hybrid", "hybrid-v3") else 3
    total_n = 3 if args.policy_type in ("hybrid", "hybrid-v3") else 4
    print(f"\n[{step_n}/{total_n}] Connecting Piper ({args.can_port}) ...")
    robot = PiperRobot(can_port=args.can_port, disable_torque_on_disconnect=False)
    robot.connect()  # connect + enable in one call
    print("  Robot connected and enabled (disable_torque_on_disconnect=False).")

    # ── Gripper opening on start ──
    gripper_start_open_value = None
    if args.open_gripper_on_start or args.reset_to_recorded_start:
        if args.gripper_start_open_value is not None:
            gripper_start_open_value = args.gripper_start_open_value
        elif recorded_start_qpos is not None:
            gripper_start_open_value = float(recorded_start_qpos[6])
        else:
            gripper_start_open_value = 0.09950
        print(f"\n  Gripper start-open target: {gripper_start_open_value:.5f} m"
              f"  (source: {'--gripper-start-open-value' if args.gripper_start_open_value is not None else 'recorded_start_qpos[6]' if recorded_start_qpos is not None else 'default'})")

    if args.reset_to_recorded_start:
        if recorded_start_qpos is None:
            print("  [ERROR] Recorded start qpos is unavailable; cannot reset.")
            return 1
        try:
            # Step 1: Open gripper to recorded start value
            grip_ok, grip_before, grip_after, grip_diff, grip_elapsed = open_gripper_to_value(
                robot=robot,
                target_gripper=gripper_start_open_value,
                timeout=args.gripper_open_timeout,
                tol=args.gripper_open_tol,
                hz=args.hz,
            )
            if not grip_ok:
                print("  [RESET-GUARD] Gripper failed to reach recorded start value. Reset aborted.")
                return 1

            # Step 2: Move arm joints to recorded start
            passed, final_qpos, reset_diff = move_robot_to_recorded_start(
                robot=robot,
                target_qpos=recorded_start_qpos,
                velocity_pct=args.velocity_pct,
                hz=args.hz,
                max_delta=max_delta,
                action_smooth=args.action_smooth,
                qpos_tol=args.start_qpos_tol,
                gripper_tol=args.start_gripper_tol,
            )
            if not passed:
                print("  [RESET-GUARD] Arm joints failed to reach recorded start. Reset aborted.")
                return 1

            # Step 3: Re-verify gripper is still at target
            grip_final = float(robot.get_joint_positions()[6])
            grip_final_diff = abs(grip_final - gripper_start_open_value)
            grip_still_ok = grip_final_diff <= args.gripper_open_tol
            print(f"\n  [GRIPPER-RESET] Re-verify after arm move:")
            print(f"    gripper_current  = {grip_final:.5f}")
            print(f"    gripper_target   = {gripper_start_open_value:.5f}")
            print(f"    diff             = {grip_final_diff:.5f}")
            print(f"    pass             = {grip_still_ok}")
            if not grip_still_ok:
                print("  [RESET-GUARD] Gripper drifted during arm move. Reset incomplete.")
                return 1

            print("  Reset-only mode: no policy rollout or policy inference executed.")
            return 0
        finally:
            try:
                cur = robot.get_joint_positions()
                robot.set_joint_positions(cur, velocity_pct=args.velocity_pct)
            except Exception:
                pass
            print("  Reset-only complete. Arm stays ENABLED at current position.")

    # ── Open gripper on start (standalone, non-reset mode) ──
    if args.open_gripper_on_start and not args.reset_to_recorded_start:
        grip_ok, grip_before, grip_after, grip_diff, grip_elapsed = open_gripper_to_value(
            robot=robot,
            target_gripper=gripper_start_open_value,
            timeout=args.gripper_open_timeout,
            tol=args.gripper_open_tol,
            hz=args.hz,
        )
        if not grip_ok:
            print("  [RESET-GUARD] Gripper failed to reach target. Cannot start rollout with gripper closed.")
            print("  Waiting for the next SPACE after you resolve the gripper state.")
            # Don't exit — allow retry from the interactive loop
        else:
            print("  Gripper open complete. Ready for policy rollout.")

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
    if force_gripper_open:
        print(f"  Gripper forced OPEN ({GRIPPER_OPEN:.3f} m) during approach phase.")
    else:
        print(f"  Gripper controlled by policy — clamped to [{GRIPPER_CLOSE:.3f}, {act_full_gripper_open:.3f}] m.")
    if args.test_mode == "A":
        print("  → Approach only — no close, no lift.")
    elif args.test_mode == "C":
        print(f"  → Approach + descend (J2 += {args.descend_j2_delta:.3f} rad) — no close.")
    elif args.test_mode == "D":
        print(f"  → Full grasp: approach + close + lift + place(J1+={args.place_j1_offset:.2f}) + release + return.")
    elif args.test_mode == "full-e2e":
        stop_label = args.full_e2e_stop_after if is_act_full else "full"
        print(f"  → Full end-to-end: approach + descend + close + lift + place + release (model-driven).")
        if is_act_full:
            print(f"  → Stop after: {stop_label} (--full-e2e-stop-after)")
    else:
        print(f"  → Approach + close ({GRIPPER_CLOSE:.3f} m) + lift (J3 -= 0.06).")
    if args.dry_run:
        print("  DRY RUN: robot commands will not be sent.")
    if args.replan_every_step:
        print("  REPLAN: policy will predict a fresh first action at every step.")
    if is_act_full and args.test_mode == "full-e2e" and args.act_full_chunk_exec != "normal":
        print(f"  Chunk exec: {args.act_full_chunk_exec} (--act-full-chunk-exec)")
    if args.no_return_to_start:
        print("  AUTO RETURN: disabled; arm will stay enabled at the stop pose.")
    if args.hold_after_stop > 0:
        print(f"  HOLD AFTER STOP: {args.hold_after_stop:.1f}s")
    if args.save_final_images:
        print("  FINAL IMAGE DEBUG: enabled")
    if recorded_start_qpos is not None:
        print(f"  Reset guard expected start: {fmt_vec(recorded_start_qpos, 5)}")
        print(f"  Reset guard source: {recorded_start_source}")
    if args.open_gripper_on_start:
        print(f"  Gripper open-on-start: target={gripper_start_open_value:.5f} m"
              f"  timeout={args.gripper_open_timeout:.1f}s  tol={args.gripper_open_tol:.4f} m")
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

            # Save start position for auto return and reset validation.
            start_robot_state = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            if args.enforce_start_reset:
                if not check_start_reset(
                    start_robot_state,
                    recorded_start_qpos,
                    args.start_qpos_tol,
                    args.start_gripper_tol,
                ):
                    print("  Reset guard failed. Waiting for the next SPACE after you reset the arm.")
                    if args.no_gui:
                        time.sleep(0.5)
                    else:
                        cv2.waitKey(750)
                    continue

            # Setup rollout saving directory
            rollout_dir = None
            _csv_fh = None
            if args.save_rollout or args.save_final_images:
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                rollout_dir = Path(PROJECT_ROOT) / "logs" / "rollouts" / f"test_a_{ts}"
                rollout_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Saving rollout to {rollout_dir}")
                # Per-step CSV for close phase debug
                _csv_path = rollout_dir / "step_log.csv"
                try:
                    _csv_fh = open(str(_csv_path), "w", encoding="utf-8")
                    _csv_fh.write("step,chunk_id,chunk_idx,current_j1,current_j2,current_j3,current_j4,current_j5,current_j6,current_grip,"
                                  "raw_j1,raw_j2,raw_j3,raw_j4,raw_j5,raw_j6,raw_grip,"
                                  "sent_j1,sent_j2,sent_j3,sent_j4,sent_j5,sent_j6,sent_grip,"
                                  "gripper_pred,gripper_feedback,close_detected,gripper_phase,target_reached,arm_error_active,wrist_frozen\n")
                except OSError as e:
                    print(f"  [WARN] Could not open CSV for writing: {e}")
                    print(f"  [WARN] Per-step CSV logging disabled for this rollout.")
                    _csv_fh = None

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
            final_step = 0
            final_target = None
            final_arm_error = 0.0

            # ── Full-E2E phase tracking (act-full only) ──
            full_e2e_phase_tracking = (is_act_full and args.test_mode == "full-e2e")
            grip_pred_history = []       # last N gripper predictions for phase detection
            close_detected = False
            release_detected = False
            close_step = -1
            release_step = -1
            consecutive_close_count = 0
            consecutive_release_count = 0
            gripper_phase = "open"
            early_gripper_close_warned = False

            # ── Act-full chunk execution mode ──
            use_chunk_exec = (is_act_full and args.test_mode == "full-e2e"
                              and args.act_full_chunk_exec != "normal")
            hold_last_mode = (args.act_full_chunk_exec == "hold_last_until_ready")
            act_chunk = None          # full chunk from predict_action_chunk: (chunk_size, 7)
            chunk_idx = 0
            chunk_id = 0
            holding_last = False
            target_hold_count = 0   # steps spent on current target (target_reached mode)
            arm_error = 0.0         # max arm joint error to current target
            _active_dims = [0, 1, 2, 3, 4, 5]
            arm_error_all = 0.0
            arm_error_active = 0.0
            ignored_wrist_error = 0.0
            _wrist_is_frozen = False

            # ── Close phase debug tracking ──
            _csv_rows = []                        # per-step CSV data
            _global_min_grip_pred = 1.0           # track minimum gripper prediction
            _first_close_candidate_step = -1       # first step where grip_pred < grip_mid
            _total_chunks_generated = 0
            _chunk_history = []                   # (chunk_id, min_grip, max_grip, min_j2, max_j2, min_j3, max_j3)

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
                if args.save_final_images and rollout_dir is not None and step in (0, 50, 100):
                    save_alignment_images(rollout_dir, f"step_{step:03d}", wrist_frame, global_frame)

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

                    using_held_last = False  # overridden in chunk exec mode

                    if use_chunk_exec:
                        # ── Chunk execution mode (hold_last_until_ready / target_reached) ──
                        need_new_chunk = (
                            act_chunk is None
                            or (chunk_idx >= len(act_chunk) and not hold_last_mode)
                        )
                        if need_new_chunk:
                            with torch.inference_mode():
                                normalized_obs = preprocessor(obs)
                                action_chunk = policy.predict_action_chunk(normalized_obs)
                                action_chunk = postprocessor(action_chunk)
                                act_chunk = action_chunk.squeeze(0).cpu().numpy()  # (chunk_size, 7)
                            chunk_idx = 0
                            chunk_id += 1
                            holding_last = False
                            target_hold_count = 0
                            _total_chunks_generated += 1
                            _chunk_grip_min = float(np.min(act_chunk[:, 6]))
                            _chunk_grip_max = float(np.max(act_chunk[:, 6]))
                            _chunk_j2_min = float(np.min(act_chunk[:, 1]))
                            _chunk_j2_max = float(np.max(act_chunk[:, 1]))
                            _chunk_j3_min = float(np.min(act_chunk[:, 2]))
                            _chunk_j3_max = float(np.max(act_chunk[:, 2]))
                            _chunk_history.append((chunk_id, _chunk_grip_min, _chunk_grip_max,
                                                   _chunk_j2_min, _chunk_j2_max, _chunk_j3_min, _chunk_j3_max))
                            _close_thresh = (act_full_gripper_open + GRIPPER_CLOSE) / 2.0
                            _has_close_candidate = _chunk_grip_min < _close_thresh
                            print(f"  [CHUNK] id={chunk_id}  size={len(act_chunk)}"
                                  f"  J2=[{_chunk_j2_min:.4f}, {_chunk_j2_max:.4f}]"
                                  f"  first={act_chunk[0][1]:.4f}  last={act_chunk[-1][1]:.4f}")
                            print(f"          J3=[{_chunk_j3_min:.4f}, {_chunk_j3_max:.4f}]"
                                  f"  Grip=[{_chunk_grip_min:.4f}, {_chunk_grip_max:.4f}]"
                                  f"  first={act_chunk[0][6]:.4f}  last={act_chunk[-1][6]:.4f}"
                                  f"  close_candidate={_has_close_candidate}")

                        if hold_last_mode:
                            # ── hold_last_until_ready ──
                            if chunk_idx < len(act_chunk):
                                model_action = act_chunk[chunk_idx].copy()
                                chunk_idx += 1
                            else:
                                # Chunk consumed — hold last action as target.
                                # Robot tracks this under max_delta/smoothing until J2 reaches READY_J2.
                                model_action = act_chunk[-1].copy()
                                holding_last = True
                                using_held_last = True
                                grip_mid = (act_full_gripper_open + GRIPPER_CLOSE) / 2.0
                                if model_action[6] < grip_mid:
                                    print(f"\n  [SAFETY] hold_last: last_action gripper={model_action[6]:.4f}"
                                          f" < mid={grip_mid:.4f} — unsafe early close!")
                                    stop_reason = "unsafe_early_close_hold"
                                    break
                                arm_error = float(np.max(np.abs(
                                    np.asarray(robot_state, dtype=np.float32)[:6] - model_action[:6])))
                        else:
                            # ── target_reached ──
                            # Advance chunk_idx only when robot reaches current target
                            # (or timeout after target_max_hold steps).
                            current_target = act_chunk[min(chunk_idx, len(act_chunk) - 1)]
                            cur_state_arr = np.asarray(robot_state, dtype=np.float32)
                            arm_error_all = float(np.max(np.abs(cur_state_arr[:6] - current_target[:6])))
                            # Determine active dims: when wrist is frozen, J4-J6 can't move
                            # so they shouldn't block chunk_idx advance.
                            _wrist_is_frozen = (cur_state_arr[1] > WRIST_FREEZE_J2)
                            if _wrist_is_frozen and args.target_reached_ignore_frozen_wrist:
                                _active_dims = [0, 1, 2]
                            else:
                                _active_dims = [0, 1, 2, 3, 4, 5]
                            arm_error_active = float(np.max(np.abs(
                                cur_state_arr[_active_dims] - current_target[_active_dims])))
                            arm_error = arm_error_active
                            ignored_wrist_error = 0.0
                            if _wrist_is_frozen and args.target_reached_ignore_frozen_wrist:
                                ignored_wrist_error = float(np.max(np.abs(
                                    cur_state_arr[3:6] - current_target[3:6])))
                            target_reached_this_step = False

                            if arm_error < args.act_full_target_tol or target_hold_count >= args.act_full_target_max_hold:
                                target_reached_this_step = True
                                chunk_idx += 1
                                target_hold_count = 0
                            else:
                                target_hold_count += 1

                            if chunk_idx < len(act_chunk):
                                model_action = act_chunk[chunk_idx].copy()
                            else:
                                model_action = act_chunk[-1].copy()
                                holding_last = True
                    else:
                        # ── Normal mode ──
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
                        if use_chunk_exec:
                            mode_label = args.act_full_chunk_exec
                            print(f"    act_full_chunk_exec={mode_label}"
                                  f"  chunk_id={chunk_id}  chunk_idx={chunk_idx}")
                            print(f"    arm_error={arm_error:.4f}"
                                  f"  current_qpos[J2]={robot_state_arr_tmp[1]:.4f}"
                                  f"  target[J2]={model_action[1]:.4f}")
                            if mode_label == "hold_last_until_ready":
                                print(f"    using_held_last_action={using_held_last}"
                                      f"  ready_count={ready_count}")
                            else:
                                print(f"    target_reached={arm_error < args.act_full_target_tol}"
                                      f"  target_hold_count={target_hold_count}")
                                print(f"    wrist_frozen={_wrist_is_frozen}"
                                      f"  active_dims={_active_dims}")
                                print(f"    arm_error_all={arm_error_all:.4f}"
                                      f"  arm_error_active={arm_error_active:.4f}")
                                print(f"    ignored_wrist_error={ignored_wrist_error:.4f}")
                                print(f"    target[J4:J6]={fmt_vec(current_target[3:6], 3)}"
                                      f"  current[J4:J6]={fmt_vec(cur_state_arr[3:6], 3)}")
                            print(f"    gripper_pred={model_action[6]:.4f}"
                                  f"  holding_last={holding_last}")
                            if act_chunk is not None:
                                print(f"    held_last_action[J2]={act_chunk[-1][1]:.4f}")
                        if full_e2e_phase_tracking and close_detected:
                            print(f"    phase: close_step={close_step}  release_step={release_step}")

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

                # Force gripper open throughout approach (unless act-full)
                if force_gripper_open:
                    clipped[6] = GRIPPER_OPEN
                else:
                    clipped[6] = np.clip(clipped[6], GRIPPER_CLOSE, act_full_gripper_open)

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
                grip_val = GRIPPER_OPEN if force_gripper_open else np.clip(
                    clipped[6], GRIPPER_CLOSE, act_full_gripper_open
                )
                sent_target = np.concatenate([smoothed_arm, [grip_val]])

                # Safety clamp to joint limits
                sent_target[:6] = np.clip(sent_target[:6], -3.14, 3.14)
                sent_target[6] = np.clip(sent_target[6], 0.0, PIPER_GRIPPER_MAX_M)

                # Gripper close latch: once close is confirmed, prevent policy from
                # re-opening the gripper before release is detected. This counters the
                # policy's learned "return to open" behavior within a trajectory chunk.
                if full_e2e_phase_tracking and close_detected and not release_detected:
                    sent_target[6] = min(sent_target[6], _close_strong_threshold)

                final_step = step + 1
                final_target = sent_target.copy()
                if use_chunk_exec:
                    final_arm_error = float(arm_error)
                else:
                    final_arm_error = float(np.max(np.abs(sent_target[:6] - robot_state_arr[:6])))

                # ── Full-E2E phase tracking (act-full only) ──
                if full_e2e_phase_tracking:
                    pred_grip = float(sent_target[6])
                    grip_pred_history.append(pred_grip)
                    if len(grip_pred_history) > 20:
                        grip_pred_history.pop(0)

                    # Update close/release detection (two-stage: onset → strong)
                    # Stage 1: close onset — grip drops below onset threshold
                    # Stage 2: strong close — grip drops below strong threshold
                    # close_detected latches once onset is sustained, stays True while grip < onset
                    if not close_detected:
                        if pred_grip < _close_onset_threshold:
                            consecutive_close_count += 1
                            if consecutive_close_count >= _close_detect_onset_count:
                                close_detected = True
                                close_step = step
                                print(f"\n  [CLOSE] Onset detected at step {step+1}: grip={pred_grip:.5f} < onset={_close_onset_threshold:.5f}")
                        else:
                            consecutive_close_count = max(0, consecutive_close_count - 1)
                    elif not release_detected:
                        # Already closed — check for strong close and reopen
                        if pred_grip < _close_strong_threshold and consecutive_close_count < _close_detect_strong_count + _close_detect_onset_count:
                            consecutive_close_count += 1
                            if consecutive_close_count >= _close_detect_onset_count + _close_detect_strong_count:
                                print(f"  [CLOSE] Strong close at step {step+1}: grip={pred_grip:.5f} < strong={_close_strong_threshold:.5f}")
                        if pred_grip > _close_onset_threshold:
                            consecutive_release_count += 1
                            if consecutive_release_count >= _close_detect_onset_count and not release_detected:
                                release_detected = True
                                release_step = step
                                print(f"\n  [CLOSE] Release detected at step {step+1}: grip={pred_grip:.5f} > onset={_close_onset_threshold:.5f}")
                        else:
                            consecutive_release_count = max(0, consecutive_release_count - 1)

                    # Early gripper close safety check (first 30 steps)
                    if close_detected and step < 30 and not early_gripper_close_warned:
                        print(f"\n  [SAFETY] Early gripper close detected at step {step+1}!")
                        print(f"    gripper_pred={pred_grip:.4f}  onset={_close_onset_threshold:.4f}")
                        early_gripper_close_warned = True
                        stop_reason = "early_gripper_close"
                        break

                    gripper_phase = get_gripper_phase(
                        pred_grip, act_full_gripper_open, GRIPPER_CLOSE,
                        close_detected, release_detected,
                    )

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
                    if full_e2e_phase_tracking:
                        print(f"    grip_phase={gripper_phase}  close_detected={close_detected}"
                              f"  release_detected={release_detected}"
                              f"  stop_after={args.full_e2e_stop_after}")
                        if close_detected:
                            print(f"    close_step={close_step}  consec_close={consecutive_close_count}")
                        if release_detected:
                            print(f"    release_step={release_step}  consec_release={consecutive_release_count}")

                # ── Safety stop: joint limit violation ──
                if np.any(np.abs(sent_target[:6]) > 3.0):
                    print(f"\n  [STOP] Joint limit violation: target={fmt_vec(sent_target)}")
                    stop_reason = "joint_limit"
                    break

                # ── Safety stop: stagnation (state barely moves, not near end) ──
                # Skip in dry-run mode: robot isn't being commanded so state won't change.
                if not args.dry_run:
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

                # ── Full-E2E phase stop (act-full only) ──
                if full_e2e_phase_tracking:
                    should_phase_stop, phase_stop_reason = check_phase_stop(
                        args.full_e2e_stop_after, step,
                        grip_pred_history, act_full_gripper_open, GRIPPER_CLOSE,
                        close_detected, release_detected,
                        close_step, release_step,
                        consecutive_close_count, consecutive_release_count,
                        30,  # lift_steps_after_close
                    )
                    if should_phase_stop:
                        # Send final target then break
                        if not args.dry_run:
                            robot.set_joint_positions(sent_target.tolist(), velocity_pct=args.velocity_pct)
                        last_smoothed = smoothed_arm.copy()
                        last_state = np.asarray(robot_state, dtype=np.float32).copy()
                        stop_reason = phase_stop_reason
                        print(f"\n  [STOP] Phase stop: {phase_stop_reason}  step={step+1}")
                        break

                # ── Ready stop: break after sending ──
                # For full-e2e with stop_after beyond approach, don't break on ready.
                # Phase stop check handles the actual stop point.
                # Max-steps still breaks normally.
                _stop_due_to_ready = (ready_count >= READY_COUNT_MIN)
                _stop_due_to_max = (step + 1 >= approach_steps)
                _skip_ready_break = (
                    full_e2e_phase_tracking
                    and args.full_e2e_stop_after != "approach"
                    and _stop_due_to_ready
                    and not _stop_due_to_max
                )
                if stop_act and not _skip_ready_break:
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
                    if stop_reason == "ready":
                        print("  WARNING: J2-ready only means joint-space ready.")
                        print("  It does NOT guarantee visual alignment with the object.")
                    break
                elif _skip_ready_break:
                    # Log that we passed approach ready but continue for next phase
                    if ready_count == READY_COUNT_MIN:
                        print(f"\n  >>> Approach ready (J2={robot_state_arr[1]:.4f}),"
                              f" continuing to next phase (stop_after={args.full_e2e_stop_after}) ...")

                # ── Per-step CSV logging (close phase debug) ──
                if rollout_dir is not None:
                    _grip_mid = (act_full_gripper_open + GRIPPER_CLOSE) / 2.0
                    _grip_fb = float(robot_state_arr[6])
                    _pred_grip = float(sent_target[6])
                    # Update global tracking
                    _global_min_grip_pred = min(_global_min_grip_pred, _pred_grip)
                    if _pred_grip < _grip_mid and _first_close_candidate_step < 0:
                        _first_close_candidate_step = step
                    _csv_row = (
                        f"{step},{chunk_id},{chunk_idx},"
                        + ",".join(f"{robot_state_arr[j]:.6f}" for j in range(7))
                        + ","
                        + ",".join(f"{raw_target[j]:.6f}" for j in range(7))
                        + ","
                        + ",".join(f"{sent_target[j]:.6f}" for j in range(7))
                        + f",{_pred_grip:.6f},{_grip_fb:.6f},{close_detected},{gripper_phase},"
                        f"{arm_error < args.act_full_target_tol if use_chunk_exec and args.act_full_chunk_exec == 'target_reached' else 'N/A'},"
                        f"{arm_error_active:.6f},{_wrist_is_frozen}\n"
                    )
                    if _csv_fh is not None:
                        _csv_fh.write(_csv_row)
                        _csv_fh.flush()

                # ── Save rollout data ──
                if args.save_rollout and rollout_dir is not None and (
                    step % 5 == 0 or step < 5 or step >= approach_steps - 5
                ):
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
                        hold_grip = GRIPPER_OPEN if force_gripper_open else np.clip(
                            float(robot.get_joint_positions()[6]), GRIPPER_CLOSE, act_full_gripper_open
                        )
                        hold_pos = np.concatenate([last_smoothed, [hold_grip]])
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
            joint_ready = bool(ready_count >= READY_COUNT_MIN)
            if joint_ready:
                print("  WARNING: J2-ready only means joint-space ready.")
                print("  It does NOT guarantee visual alignment with the object.")

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

            if args.save_final_images and rollout_dir is not None:
                final_qpos_for_debug = None
                try:
                    final_qpos_for_debug = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                except Exception as exc:
                    print(f"  [WARN] Could not read final qpos for alignment debug: {exc}")
                final_wrist_frame, final_global_frame = read_final_camera_frames(wrist_cam, global_cam)
                save_alignment_images(rollout_dir, "final", final_wrist_frame, final_global_frame)

                debug_payload = {
                    "final_step": int(final_step),
                    "stop_reason": stop_reason,
                    "final_qpos": to_jsonable(final_qpos_for_debug),
                    "final_gripper": (
                        float(final_qpos_for_debug[6]) if final_qpos_for_debug is not None else None
                    ),
                    "final_chunk_id": int(chunk_id),
                    "final_chunk_idx": int(chunk_idx),
                    "final_target": to_jsonable(final_target),
                    "final_arm_error": float(final_arm_error),
                    "final_ready_count": int(ready_count),
                    "close_detected": bool(close_detected),
                    "release_detected": bool(release_detected),
                    "gripper_phase": gripper_phase,
                    "rollout_dir": str(rollout_dir),
                    "joint_ready": joint_ready,
                    "visual_alignment_required": True,
                    "user_note": "",
                }
                debug_path = rollout_dir / "approach_alignment_debug.json"
                debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
                print(f"  Saved approach alignment debug JSON: {debug_path}")
                print("  Please inspect final_realsense.png and final_usb.png:")
                print("    - Is the gripper center aligned with the bottle center?")
                print("    - Is the approach height correct?")
                print("    - Is the gripper orientation correct?")
                print("    - If not aligned, do not run close/full.")

            # ── Close phase debug summary ──
            if full_e2e_phase_tracking:
                _grip_mid_summary = (act_full_gripper_open + GRIPPER_CLOSE) / 2.0
                print(f"\n  {'─' * 40}")
                print(f"  [CLOSE-DEBUG] Rollout Summary")
                print(f"  {'─' * 40}")
                print(f"    Total steps            : {len(raw_actions)}")
                print(f"    Total chunks generated : {_total_chunks_generated}")
                print(f"    Final chunk_id         : {chunk_id}")
                print(f"    Final chunk_idx        : {chunk_idx}")
                print(f"    Global min gripper_pred: {_global_min_grip_pred:.5f}")
                print(f"    Global max gripper_pred: {float(np.max([r[6] for r in raw_actions])):.5f}" if raw_actions else "    No raw actions")
                print(f"    Close threshold (mid)  : {_grip_mid_summary:.5f}")
                print(f"    First close candidate  : step={_first_close_candidate_step}"
                      f"{' (NONE)' if _first_close_candidate_step < 0 else ''}")
                print(f"    close_detected         : {close_detected} (step={close_step})")
                print(f"    Final raw action       : {fmt_vec(raw_actions[-1], 5)}" if raw_actions else "    No raw actions")
                print(f"    Final target qpos      : {fmt_vec(final_target, 5)}" if final_target is not None else "    No final target")
                if _chunk_history:
                    print(f"    Chunk grip history:")
                    for cid, gmin, gmax, j2min, j2max, j3min, j3max in _chunk_history:
                        _has = gmin < _grip_mid_summary
                        print(f"      chunk {cid:2d}: grip=[{gmin:.4f},{gmax:.4f}]"
                              f"  J2=[{j2min:.3f},{j2max:.3f}]  J3=[{j3min:.3f},{j3max:.3f}]"
                              f"  close_candidate={_has}")
                print(f"  {'─' * 40}")

            # Close CSV file (always, regardless of phase tracking)
            if _csv_fh is not None:
                _csv_fh.close()
                _csv_fh = None

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
            #  TEST MODE full-e2e: model handles full trajectory, no scripted phases
            # ================================================================
            if args.test_mode == "full-e2e" and not user_quit:
                print(f"  [FULL-E2E] Trajectory finished — stop_reason={stop_reason}")
                print(f"  [FULL-E2E] stop_after={args.full_e2e_stop_after if is_act_full else 'full'}")
                final_state = robot.get_joint_positions()
                print(f"  [FULL-E2E] Final step={len(raw_actions)}  "
                      f"state: J2={final_state[1]:.5f} rad  grip={final_state[6]:.5f} m")
                # Phase summary
                if is_act_full:
                    print(f"  [FULL-E2E] Phase: close_detected={close_detected} (step={close_step})"
                          f"  release_detected={release_detected} (step={release_step})")
                    print(f"  [FULL-E2E] Final gripper_phase={gripper_phase}")
                    # Show last gripper prediction
                    if grip_pred_history:
                        last_grips = grip_pred_history[-5:]
                        print(f"  [FULL-E2E] Last 5 grip_preds: {[f'{g:.4f}' for g in last_grips]}")
                    # Early close warning
                    if early_gripper_close_warned:
                        print(f"  [FULL-E2E] [WARN] Early gripper close was detected!")
                # Gripper state assessment
                if final_state[6] > act_full_gripper_open - 0.01:
                    print(f"  [FULL-E2E] Gripper is OPEN ({final_state[6]:.4f} m).")
                elif final_state[6] < GRIPPER_CLOSE + 0.01:
                    print(f"  [FULL-E2E] Gripper is CLOSED ({final_state[6]:.4f} m) — holding object?")
                else:
                    print(f"  [FULL-E2E] Gripper is partially open ({final_state[6]:.4f} m).")

            # ================================================================
            #  TEST MODE C: approach → descend 2-3cm → stop (no close)
            # ================================================================
            if args.test_mode == "C" and not user_quit and not is_act_full:
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
            if args.test_mode == "B" and not user_quit and not is_act_full:
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
            if args.test_mode == "D" and not user_quit and not is_act_full:
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
            #  ALIGNMENT DEBUG HOLD
            # ================================================================
            alignment_debug_hold = (
                args.no_return_to_start
                and args.hold_after_stop > 0
                and not user_quit
                and not args.dry_run
                and is_act_full
                and args.test_mode == "full-e2e"
                and args.full_e2e_stop_after in ("approach", "close")
            )
            if alignment_debug_hold:
                print(f"\n  >>> Holding stop pose for visual inspection ({args.hold_after_stop:.1f}s) ...")
                cur = robot.get_joint_positions()
                hold_start = time.time()
                while time.time() - hold_start < args.hold_after_stop:
                    robot.set_joint_positions(cur, velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                print("  Hold-after-stop complete. Arm remains enabled at current pose.")

            # ================================================================
            #  RETURN TO START
            # ================================================================
            auto_return = (not args.no_return_to_start and not user_quit and not args.dry_run)
            if auto_return:
                if is_act_full and recorded_start_qpos is not None:
                    print("\n  >>> Returning to recorded training start ...")
                    move_robot_to_recorded_start(
                        robot=robot,
                        target_qpos=recorded_start_qpos,
                        velocity_pct=args.velocity_pct,
                        hz=args.hz,
                        max_delta=max_delta,
                        action_smooth=args.action_smooth,
                        qpos_tol=args.start_qpos_tol,
                        gripper_tol=args.start_gripper_tol,
                    )
                    print("  Returned to recorded training start.")
                else:
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
    sys.exit(main())
