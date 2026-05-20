#!/usr/bin/env python3
"""Deploy trained Diffusion Policy on Piper arm for bottle grasping.

USAGE:
  python3 inference/deploy_diffusion.py \
      --checkpt outputs/train/piper_bottle_grasp_diffusion_v1/checkpoints/50000/pretrained_model \
      --hz 30 --max-steps 200 --debug-actions
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
from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices

PIPER_GRIPPER_MAX_M = 0.101


def fmt_vec(values, precision=3):
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def max_abs_diff(cur, prev) -> float:
    if prev is None:
        return float("nan")
    return float(np.max(np.abs(np.asarray(cur, dtype=np.float32) - np.asarray(prev, dtype=np.float32))))


def interpolate_joint_path(start, target, max_step_rad, max_step_gripper):
    diff = np.asarray(target, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    arm_steps = int(np.ceil(np.max(np.abs(diff[:6])) / max_step_rad)) if max_step_rad > 0 else 1
    grip_steps = int(np.ceil(abs(diff[6]) / max_step_gripper)) if max_step_gripper > 0 else 1
    n_steps = max(arm_steps, grip_steps, 1)
    waypoints = []
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        waypoints.append(np.asarray(start, dtype=np.float32) + diff * alpha)
    return waypoints


def policy_state_dim(policy) -> int:
    feature = policy.config.input_features.get("observation.state")
    if feature is None:
        return 0
    return int(feature.shape[0])


def load_pre_post_processors(policy, checkpt, device):
    from lerobot.policies.factory import make_pre_post_processors
    pre_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {"device": device.type},
    }
    post_overrides = {
        "unnormalizer_processor": {"device": device.type},
        "device_processor": {"device": "cpu"},
    }
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpt,
        preprocessor_overrides=pre_overrides,
        postprocessor_overrides=post_overrides,
    )


def prepare_observation(state, wrist_img, global_img, device, expected_state_dim=7, gripper_unit_scale=1.0):
    obs = {}
    state_arr = np.asarray(state, dtype=np.float32).copy()
    state_arr[6] *= gripper_unit_scale
    # Diffusion Policy may need 2 observation steps — handled by preprocessor
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str, required=True)
    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--velocity-pct", type=int, default=25)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--global-camera", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-actions", action="store_true")
    parser.add_argument("--debug-every", type=int, default=10)
    # Gripper
    parser.add_argument("--gripper-unit-scale", type=float, default=1.0)
    parser.add_argument("--force-close-after-step", type=int, default=None)
    parser.add_argument("--force-close-target", type=float, default=0.0)
    # Hybrid / lift
    parser.add_argument("--hybrid-force-grasp", action="store_true")
    parser.add_argument("--lift-after-close", action="store_true")
    parser.add_argument("--waypoints", type=str, default=None)
    parser.add_argument("--lift-pose", type=float, nargs=7, default=None)
    parser.add_argument("--lift-hold-seconds", type=float, default=2.0)
    parser.add_argument("--close-hold-seconds", type=float, default=1.0)
    # Action smoothing
    parser.add_argument("--action-smoothing", type=float, default=0.0,
                        help="EMA weight for action smoothing (0=none, 0.5=moderate)")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--training-gripper-min", type=float, default=None)
    parser.add_argument("--training-gripper-max", type=float, default=None)
    args = parser.parse_args()

    # Waypoints
    waypoints = None
    if args.waypoints:
        import json as _json
        waypoints = _json.loads(Path(args.waypoints).read_text())

    print("=" * 60)
    print("  Piper Diffusion Policy Deployment — Bottle Grasp")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load policy
    print(f"\n[1/4] Loading Diffusion Policy from {args.checkpt} ...")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    policy = DiffusionPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    horizon = policy.config.horizon
    n_action_steps = policy.config.n_action_steps
    expected_state_dim = policy_state_dim(policy)
    print(f"  Policy (horizon={horizon}, n_action_steps={n_action_steps},"
          f" state_dim={expected_state_dim})")

    # Pre/post processors
    print("\n[2/4] Loading processors ...")
    preprocessor, postprocessor = load_pre_post_processors(policy, args.checkpt, device)
    print("  Processors ready.")

    # Robot
    print(f"\n[3/4] Connecting Piper ({args.can_port}) ...")
    robot = PiperRobot(can_port=args.can_port)
    robot.connect()  # connect + enable in one call
    print("  Robot connected.")

    # Cameras
    print("\n[4/4] Initializing cameras ...")
    rs_serials = find_realsense_devices()
    wrist_serial = rs_serials[0] if rs_serials else ""
    wrist_cam = RealSenseCamera(serial=wrist_serial, width=640, height=480, fps=30, enable_depth=False)
    global_cam = None
    requires_global = "observation.images.global_rgb" in policy.config.input_features
    if args.no_global and requires_global:
        raise ValueError("Policy requires global_rgb; --no-global not allowed.")
    if not args.no_global:
        try:
            global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
        except IOError as e:
            if requires_global:
                raise
            print(f"  Global camera skipped: {e}")
    print("  Cameras ready.")

    # Gripper OOD check
    robot_state = robot.get_joint_positions()
    grip_raw = robot_state[6]
    grip_policy = grip_raw * args.gripper_unit_scale
    print(f"\n  Gripper state (robot): {grip_raw:.6f}  (policy): {grip_policy:.6f}")
    if args.training_gripper_min is not None and args.training_gripper_max is not None:
        if grip_policy < args.training_gripper_min or grip_policy > args.training_gripper_max:
            print(f"  [WARN] Gripper outside training range"
                  f" [{args.training_gripper_min:.3f}, {args.training_gripper_max:.3f}].")

    if args.hybrid_force_grasp:
        print("  HYBRID MODE: code controls gripper + lift.")
        hybrid_open = waypoints.get("open_gripper", None) if waypoints else None
        hybrid_close = waypoints.get("close_gripper", args.force_close_target) if waypoints else args.force_close_target
        print(f"  HYBRID GRIPPER: open={hybrid_open if hybrid_open is not None else 'current'}"
              f"  close={hybrid_close:.6f}")
    if args.lift_after_close:
        print("  LIFT AFTER CLOSE: enabled.")

    print("\n" + "-" * 60)
    print("  SPACE = grasp    Q/ESC = quit")
    if args.dry_run:
        print("  DRY RUN")
    print("-" * 60 + "\n")

    action_buffer = None  # for action smoothing

    try:
        while True:
            if args.no_gui:
                cmd = input("  Press ENTER to run grasp, Q then ENTER to quit: ").strip().lower()
                if cmd == "q":
                    break
            else:
                wrist_frame = wrist_cam.read()
                global_frame = global_cam.read() if global_cam else None
                preview = cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(preview, "READY - SPACE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Diffusion Policy Deployment", preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                if key != ord(' '):
                    continue

            print("  >>> Grasp attempt ...")
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            action_buffer = None

            last_target = None
            last_state = None
            noop_count = 0

            for step in range(max(1, args.max_steps)):
                loop_start = time.time()

                wrist_frame = wrist_cam.read()
                global_frame = global_cam.read() if global_cam else None
                robot_state = robot.get_joint_positions()

                wrist_img = wrist_frame.rgb
                global_img = global_frame.rgb if global_frame else None
                obs = prepare_observation(robot_state, wrist_img, global_img, device,
                                          expected_state_dim, args.gripper_unit_scale)

                with torch.inference_mode():
                    normalized_obs = preprocessor(obs)
                    action = policy.select_action(normalized_obs)
                    action = postprocessor(action)

                if action.dim() == 3:
                    action = action[:, 0, :]
                if action.dim() == 2:
                    action = action.squeeze(0)
                model_action = action.cpu().numpy()
                robot_state_arr = np.asarray(robot_state, dtype=np.float32)

                # Action smoothing
                if args.action_smoothing > 0 and action_buffer is not None:
                    model_action = (args.action_smoothing * action_buffer
                                    + (1 - args.action_smoothing) * model_action)
                action_buffer = model_action.copy()

                target = robot_state_arr + model_action  # delta mode
                target[:6] = np.clip(target[:6], -3.14, 3.14)
                target[6] = np.clip(target[6], 0.0, PIPER_GRIPPER_MAX_M)

                if args.hybrid_force_grasp:
                    open_target = waypoints.get("open_gripper", robot_state_arr[6]) if waypoints else robot_state_arr[6]
                    target[6] = np.clip(open_target, 0.0, PIPER_GRIPPER_MAX_M)

                # Force close
                if (args.force_close_after_step is not None
                        and step >= args.force_close_after_step):
                    target[6] = np.clip(args.force_close_target, 0.0, PIPER_GRIPPER_MAX_M)

                delta = target - robot_state_arr
                max_arm_delta = float(np.max(np.abs(delta[:6])))
                gripper_delta = float(abs(delta[6]))

                if args.debug_actions and (
                    step == 0 or step == args.max_steps - 1
                    or step % max(1, args.debug_every) == 0
                ):
                    print(f"  step {step+1:03d}: max_arm_delta={max_arm_delta:.4f}  "
                          f"grip_delta={gripper_delta:.4f}  "
                          f"target_diff={max_abs_diff(target, last_target):.4f}")
                    print(f"    state : {fmt_vec(robot_state)}")
                    print(f"    target: {fmt_vec(target)}")

                if max_arm_delta < 0.002 and gripper_delta < 0.001:
                    noop_count += 1
                else:
                    noop_count = 0
                if noop_count >= 50:
                    print(f"  [WARN] No-op for {noop_count} steps; stopping.")
                    break

                if not args.dry_run:
                    robot.set_joint_positions(target.tolist(), velocity_pct=args.velocity_pct)

                if not args.no_gui:
                    preview = cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(preview, f"EXEC {step+1}/{args.max_steps}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Diffusion Policy Deployment", preview)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                        break

                elapsed = time.time() - loop_start
                s_time = 1.0 / args.hz
                if elapsed < s_time:
                    time.sleep(s_time - elapsed)
                last_target = target.copy()
                last_state = np.asarray(robot_state, dtype=np.float32).copy()

            if args.hybrid_force_grasp:
                close_target = waypoints.get("close_gripper", args.force_close_target) if waypoints else args.force_close_target
                close_target = float(np.clip(close_target, 0.0, PIPER_GRIPPER_MAX_M))
                print(f"\n  >>> Hybrid close gripper to {close_target:.6f} ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                close_pose = cur.copy()
                close_pose[6] = close_target
                close_path = interpolate_joint_path(cur, close_pose, 0.03, 0.002)
                for ci, ct in enumerate(close_path):
                    if not args.dry_run:
                        robot.set_joint_positions(ct.tolist(), velocity_pct=args.velocity_pct)
                    if ci == 0 or ci == len(close_path) - 1 or (ci + 1) % 10 == 0:
                        print(f"    close {ci+1:3d}/{len(close_path):3d}  grip={ct[6]:.4f}")
                    time.sleep(1.0 / args.hz)
                hold_start = time.time()
                while time.time() - hold_start < args.close_hold_seconds:
                    if not args.dry_run:
                        robot.set_joint_positions(close_pose.tolist(),
                                                  velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)

            # Lift after close
            if args.lift_after_close:
                print("\n  >>> Lift after close ...")
                if args.lift_pose:
                    lift_target = np.array(args.lift_pose, dtype=np.float32)
                elif waypoints and "lift_pose" in waypoints:
                    lift_target = np.array(waypoints["lift_pose"], dtype=np.float32)
                else:
                    lift_target = np.array(robot.get_joint_positions(), dtype=np.float32)
                    lift_target[2] -= 0.1
                lift_path = interpolate_joint_path(
                    np.asarray(robot.get_joint_positions(), dtype=np.float32),
                    lift_target, 0.03, 0.002)
                for li, lt in enumerate(lift_path):
                    lt[:6] = np.clip(lt[:6], -3.14, 3.14)
                    lt[6] = np.clip(lt[6], 0.0, PIPER_GRIPPER_MAX_M)
                    if not args.dry_run:
                        robot.set_joint_positions(lt.tolist(), velocity_pct=args.velocity_pct)
                    if li == 0 or li == len(lift_path) - 1 or (li + 1) % 20 == 0:
                        print(f"    lift {li+1:3d}/{len(lift_path):3d}  {fmt_vec(lt, 3)}")
                    time.sleep(1.0 / args.hz)
                hold_start = time.time()
                while time.time() - hold_start < args.lift_hold_seconds:
                    if not args.dry_run:
                        robot.set_joint_positions(lift_path[-1].tolist(),
                                                  velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)

            print("  Trajectory complete.")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        print("  Shutting down ...")
        robot.disable()
        robot.disconnect()
        wrist_cam.close()
        if global_cam:
            global_cam.close()
        if not args.no_gui:
            cv2.destroyAllWindows()
        print("  Done.")


if __name__ == "__main__":
    main()
