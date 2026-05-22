#!/usr/bin/env python3
"""Guarded adapter-v2 mirror recorder for Piper LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapter_v2.piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from adapter_v2.reset import reset_to_standard_start
from adapter_v2.schema import (
    GLOBAL_CAMERA_KEY,
    QposTolerance,
    STANDARD_START_QPOS,
    ZONE_ARM_TOLERANCE_RAD,
    ZONE_GRIPPER_OPEN_MIN_M,
    StartGuardMode,
    as_qpos,
)
from adapter_v2.start_pose import qpos_diff, start_pose_guard
from camera.rs_camera import USBCamera
from teleop import data_collector as collector

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "lerobot_dataset_piper_adapter_v2"
DEFAULT_DATASET_REPO_ID = "piper/adapter_v2"
DEFAULT_TASK = "Piper adapter v2 hardware mirror demonstration"
MIN_SAVE_FRAMES = 1


class RecordState(str, Enum):
    WAIT_FOR_START_GUARD = "WAIT_FOR_START_GUARD"
    START_GUARD_PASS = "START_GUARD_PASS"
    WAIT_FOR_USER_START = "WAIT_FOR_USER_START"
    RECORDING = "RECORDING"
    WAIT_FOR_USER_STOP = "WAIT_FOR_USER_STOP"
    SAVE_EPISODE = "SAVE_EPISODE"
    NEXT_EPISODE_START_GUARD = "NEXT_EPISODE_START_GUARD"


@dataclass(frozen=True)
class StartGuardResult:
    expected: np.ndarray
    current: np.ndarray
    diff: np.ndarray
    tolerance: QposTolerance
    passed: bool


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def parse_qpos(text: str, *, label: str) -> np.ndarray:
    if not text:
        return STANDARD_START_QPOS.copy()
    return as_qpos([float(value.strip()) for value in text.split(",")], label=label)


def load_start_pose_file(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        print(f"  [WARN] start pose file not found: {file_path}, using schema default.")
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        qpos = as_qpos(data["qpos"], label=f"start pose file {file_path}")
        print(f"  Loaded start pose from {file_path}: {fmt_qpos(qpos)}")
        return qpos
    except Exception as exc:
        print(f"  [WARN] failed to load start pose file {file_path}: {exc}")
        return None


def fmt_qpos(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values]


def check_start_guard(
    bus: PiperMotorsBusV2,
    expected: np.ndarray,
    *,
    mode: StartGuardMode = "zone",
    tolerance: QposTolerance = QposTolerance(),
) -> StartGuardResult:
    current = bus.read_qpos()
    diff = qpos_diff(current, expected)
    passed = start_pose_guard(current, expected, mode=mode, tolerance=tolerance)
    result = StartGuardResult(expected, current, diff, tolerance, passed)
    print()
    print(f"Start guard ({mode} mode):")
    print(f"  expected qpos: {fmt_qpos(result.expected)}")
    print(f"  current qpos : {fmt_qpos(result.current)}")
    print(f"  abs diff     : {fmt_qpos(result.diff)}")
    if mode == "zone":
        print(f"  zone J1-J3 tol: {ZONE_ARM_TOLERANCE_RAD[:3]}")
        print(f"  zone J4-J6 tol: {ZONE_ARM_TOLERANCE_RAD[3:]}")
        print(f"  zone gripper min open: {ZONE_GRIPPER_OPEN_MIN_M} m")
    else:
        print(f"  arm tol      : {result.tolerance.arm_rad:.5f} rad")
        print(f"  gripper tol  : {result.tolerance.gripper_m:.5f} m")
    if result.passed:
        print("  START GUARD PASS")
    else:
        print("  START GUARD FAIL")
        print("  Adjust the arm manually, then press C to re-check.")
        if mode == "zone":
            per_joint_ok = diff[:6] <= np.asarray(ZONE_ARM_TOLERANCE_RAD, dtype=np.float32)
            for i in range(6):
                status = "OK" if per_joint_ok[i] else "EXCEEDED"
                print(f"    j{i+1}: diff={float(diff[i]):.5f} rad (tol={ZONE_ARM_TOLERANCE_RAD[i]:.4f}) [{status}]")
            gripper_status = "OK" if float(current[6]) >= ZONE_GRIPPER_OPEN_MIN_M else "TOO CLOSED"
            print(f"    gripper: {float(current[6]):.5f} m (min={ZONE_GRIPPER_OPEN_MIN_M}) [{gripper_status}]")
    print()
    return result


def create_global_camera(device_id: str):
    print("\nInitializing adapter-v2 global camera ...")
    collector.print_camera_inventory()
    cam = USBCamera(
        device_id=device_id,
        width=collector.GLOBAL_WIDTH,
        height=collector.GLOBAL_HEIGHT,
        fps=collector.GLOBAL_FPS,
    )
    print(f"  Selected global camera: {cam.device_id}")
    return cam


def is_frame_black(frame, threshold: float = 5.0) -> bool:
    if frame is None or frame.rgb is None:
        return True
    try:
        gray_mean = float(frame.rgb.mean())
        return gray_mean < threshold
    except Exception:
        return True


def create_or_resume_dataset(args):
    dataset_cls = collector.load_lerobot_dataset_class()
    if dataset_cls is None:
        raise RuntimeError("LeRobotDataset is unavailable in this environment.")
    collector.CONTROL_RATE_HZ = args.fps
    dataset = collector.create_or_resume_dataset(
        dataset_cls,
        Path(args.dataset_root),
        args.dataset_repo_id,
        include_wrist=False,
    )
    dataset_fps = float(getattr(dataset, "fps", args.fps))
    if abs(dataset_fps - args.fps) > 1e-6:
        print(f"  [WARN] requested fps={args.fps}, existing dataset fps={dataset_fps}.")
    return dataset


def save_episode_start_metadata(
    dataset_root: Path,
    *,
    episode_id: int,
    result: StartGuardResult,
    fps: float,
    camera_key: str,
    operator_start_time: str,
    operator_stop_time: str,
    start_pose_file: str = "",
    start_guard_mode: str = "zone",
) -> Path:
    metadata = {
        "episode_id": int(episode_id),
        "expected_start_qpos": [float(value) for value in result.expected],
        "actual_start_qpos": [float(value) for value in result.current],
        "start_abs_diff": [float(value) for value in result.diff],
        "start_guard_pass": True,
        "fps": float(fps),
        "camera_key": camera_key,
        "operator_start_time": operator_start_time,
        "operator_stop_time": operator_stop_time,
        "start_pose_file": start_pose_file,
        "start_guard_mode": start_guard_mode,
    }
    metadata_dir = dataset_root / "meta" / "adapter_v2_episode_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"episode_{episode_id:06d}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def draw_status_preview(cv2, preview, *, state: RecordState, episode_id: int, dry_run: bool, guard_passed: bool = False):
    if preview is None:
        preview = np.zeros((collector.GLOBAL_HEIGHT, collector.GLOBAL_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            preview,
            "Waiting for global camera...",
            (40, collector.GLOBAL_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
    color = (0, 255, 0) if guard_passed else (0, 255, 255)
    lines = [
        f"adapter v2 {'DRY RUN' if dry_run else 'RECORD'}",
        f"episode {episode_id + 1}  state {state.value}",
        "",
    ]
    if state in (RecordState.WAIT_FOR_START_GUARD, RecordState.NEXT_EPISODE_START_GUARD):
        if guard_passed:
            lines[2] = "GUARD PASS - press SPACE to start recording"
        else:
            lines[2] = "C re-check guard  R reset  Q quit"
    elif state == RecordState.WAIT_FOR_USER_START:
        lines[2] = "GUARD PASS - press SPACE to start recording  C re-check  Q quit"
    elif state in (RecordState.WAIT_FOR_USER_STOP,):
        lines[2] = "SPACE/ENTER stop  (recording in progress)"
    elif state == RecordState.SAVE_EPISODE:
        lines[2] = "Saving episode..."
    else:
        lines[2] = "SPACE start/stop  C check  R reset  Q quit"
    for index, line in enumerate(lines):
        cv2.putText(
            preview,
            line,
            (12, 28 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return preview


def confirm_reset() -> bool:
    print()
    print("Reset request received.")
    print("  The recorder is not recording any frames during reset.")
    print("  Ensure the teaching hardware state is safe before commanding follower reset.")
    answer = input("  Type RESET to move to STANDARD_START_QPOS: ").strip()
    if answer != "RESET":
        print("  Reset cancelled.")
        return False
    return True


def run_confirmed_reset(bus: PiperMotorsBusV2, expected: np.ndarray) -> None:
    if not confirm_reset():
        return
    final_qpos = reset_to_standard_start(bus, expected, confirmed=True)
    print(f"  Reset final qpos: {fmt_qpos(final_qpos)}")
    print("  Arm stays ENABLED after reset_to_standard_start.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safe adapter-v2 Piper recorder with a required per-episode start guard."
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--camera-key", default=GLOBAL_CAMERA_KEY)
    parser.add_argument(
        "--expected-start-qpos",
        default="",
        help="Comma-separated [j1,j2,j3,j4,j5,j6,gripper]. Default uses adapter-v2 schema.",
    )
    parser.add_argument(
        "--start-guard-mode",
        choices=("strict", "zone"),
        default="zone",
        help="strict: scalar arm/gripper tolerance near STANDARD_START_QPOS. zone: per-joint tolerances J1-J3=0.08,J4-J6=0.12, gripper >= 0.09m (default).",
    )
    parser.add_argument("--arm-start-tol", type=float, default=0.05)
    parser.add_argument("--gripper-start-tol", type=float, default=0.010)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-start-guard",
        action="store_true",
        default=True,
        help="Always enabled. Recording cannot bypass the adapter-v2 start guard.",
    )
    parser.add_argument("--global-camera", default="auto")
    parser.add_argument(
        "--start-pose-file",
        default="",
        help="Path to a JSON file with 'qpos' key. Overrides --expected-start-qpos and schema default.",
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()
    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive.")
    if args.fps <= 0:
        parser.error("--fps must be positive.")
    if args.camera_key != GLOBAL_CAMERA_KEY:
        parser.error(f"--camera-key must stay {GLOBAL_CAMERA_KEY} for adapter v2 Stage 1.")
    if args.arm_start_tol <= 0 or args.gripper_start_tol <= 0:
        parser.error("start tolerances must be positive.")
    return args


def main() -> int:
    args = parse_args()
    start_pose_from_file = load_start_pose_file(args.start_pose_file)
    if start_pose_from_file is not None:
        expected_start = start_pose_from_file
    else:
        expected_start = parse_qpos(args.expected_start_qpos, label="--expected-start-qpos")
    tolerance = QposTolerance(args.arm_start_tol, args.gripper_start_tol)
    start_guard_mode: StartGuardMode = args.start_guard_mode
    cv2 = collector.ensure_opencv()
    collector.check_gui_environment()

    print("=" * 72)
    print("Adapter v2 Piper guarded recorder")
    print(f"  mode              : {'dry-run' if args.dry_run else 'record'}")
    print(f"  start guard mode  : {start_guard_mode}")
    print(f"  can_port          : {args.can_port}")
    print(f"  num_episodes      : {args.num_episodes}")
    print(f"  fps               : {args.fps:.3f}")
    print(f"  camera_key        : {args.camera_key}")
    print("  camera topology   : global camera only, no wrist camera")
    print("  state/action dim  : 7 [j1,j2,j3,j4,j5,j6,gripper]")
    print(f"  start pose source : {'file=' + args.start_pose_file if start_pose_from_file is not None else 'schema STANDARD_START_QPOS'}")
    print("  start guard       : REQUIRED for every episode")
    print("  reset recording   : forbidden")
    if start_guard_mode == "zone":
        print(f"  zone J1-J3 tol    : {ZONE_ARM_TOLERANCE_RAD[:3]}")
        print(f"  zone J4-J6 tol    : {ZONE_ARM_TOLERANCE_RAD[3:]}")
        print(f"  zone gripper min  : {ZONE_GRIPPER_OPEN_MIN_M} m")
    print("=" * 72)

    bus = PiperMotorsBusV2(PiperMotorsBusV2Config(can_port=args.can_port))
    global_cam = None
    dataset = None
    window_name = "Adapter v2 Recorder | Global"
    state = RecordState.WAIT_FOR_START_GUARD
    guard_result = None
    active_start_guard = None
    active_episode_id = 0
    dry_run_episodes = 0
    saved_episodes_this_run = 0
    prev_qpos = None
    operator_start_time = ""
    camera_error_log: dict[str, float] = {}
    global_frame = None
    request_guard_check = True
    guard_passed = False

    try:
        print("\n[1/2] Connecting Piper ...")
        bus.connect()
        print("  Connected.")
        initial_qpos = bus.read_qpos()
        print(f"  Initial qpos   : {fmt_qpos(initial_qpos)}")
        print(f"  Initial gripper: {float(initial_qpos[6]):.6f} m")

        collector.create_preview_window(window_name, collector.GLOBAL_WIDTH, collector.GLOBAL_HEIGHT)
        if not args.dry_run:
            print("\n[2/2] Camera preview before dataset writes ...")
            global_cam = create_global_camera(args.global_camera)
        else:
            print("\n[2/2] Dry-run skips cameras and dataset writes.")

        print()
        print("Controls:")
        print("  C            re-check start guard")
        print("  SPACE        start recording (only after GUARD PASS), or stop while recording")
        print("  ENTER        stop while recording")
        print("  R            confirmed reset_to_standard_start outside recording")
        print("  Q / ESC      quit")
        print()

        period = 1.0 / args.fps
        while True:
            t0 = time.time()
            episode_id = dry_run_episodes if args.dry_run else (
                int(getattr(dataset, "num_episodes", 0)) if dataset is not None else 0
            )
            active_episode_id = episode_id

            if state in (
                RecordState.WAIT_FOR_START_GUARD,
                RecordState.NEXT_EPISODE_START_GUARD,
            ) and request_guard_check:
                guard_result = check_start_guard(
                    bus, expected_start,
                    mode=start_guard_mode,
                    tolerance=tolerance,
                )
                request_guard_check = False
                if guard_result.passed:
                    guard_passed = True
                    state = RecordState.START_GUARD_PASS
                else:
                    guard_passed = False
                    state = RecordState.WAIT_FOR_START_GUARD

            if state == RecordState.START_GUARD_PASS:
                print("  GUARD PASS — press SPACE to start recording.")
                state = RecordState.WAIT_FOR_USER_START

            cur_qpos = None
            try:
                cur_qpos = bus.read_qpos()
            except Exception as exc:
                now = time.time()
                if now - camera_error_log.get("qpos", 0.0) > 2.0:
                    print(f"  [WARN] read qpos failed: {exc}")
                    camera_error_log["qpos"] = now

            if global_cam is not None:
                global_frame = collector.safe_read_camera("global", global_cam, camera_error_log)

            if state in (RecordState.RECORDING, RecordState.WAIT_FOR_USER_STOP):
                state = RecordState.WAIT_FOR_USER_STOP
                if not args.dry_run and dataset is not None and cur_qpos is not None:
                    if prev_qpos is not None and global_frame is not None:
                        frame = {
                            "observation.state": np.asarray(prev_qpos, dtype=np.float32),
                            "action": np.asarray(cur_qpos, dtype=np.float32),
                            "task": args.task,
                            args.camera_key: np.transpose(global_frame.rgb, (2, 0, 1)),
                        }
                        try:
                            dataset.add_frame(frame)
                        except Exception as exc:
                            print(f"  [WARN] dataset.add_frame failed: {exc}")
                    elif global_frame is None:
                        now = time.time()
                        if now - camera_error_log.get("frame_skip", 0.0) > 2.0:
                            print("  [WARN] recording frame skipped: global frame unavailable")
                            camera_error_log["frame_skip"] = now
                    prev_qpos = cur_qpos

            frame_count = collector.dataset_buffer_size(dataset) if dataset is not None else 0
            preview = collector.build_preview(
                None,
                global_frame,
                bus.is_enabled,
                state == RecordState.WAIT_FOR_USER_STOP,
                frame_count,
            )
            preview = draw_status_preview(
                cv2,
                preview,
                state=state,
                episode_id=episode_id,
                dry_run=args.dry_run,
                guard_passed=guard_passed,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if collector.should_quit(key, window_name):
                if state == RecordState.WAIT_FOR_USER_STOP and dataset is not None:
                    collector.clear_dataset_buffer(dataset)
                    print("  Recording buffer discarded on quit.")
                break

            if key in (ord("c"), ord("C")):
                if state == RecordState.WAIT_FOR_USER_STOP:
                    print("  [WARN] Stop the active episode before rechecking start guard.")
                else:
                    print("  Re-checking start guard ...")
                    state = RecordState.WAIT_FOR_START_GUARD
                    request_guard_check = True

            elif key in (ord("r"), ord("R")):
                if state == RecordState.WAIT_FOR_USER_STOP:
                    print("  [WARN] Reset is blocked while recording.")
                elif args.dry_run:
                    print("  DRY RUN: R sends no motion. Reset manually and press C.")
                else:
                    run_confirmed_reset(bus, expected_start)
                    state = RecordState.WAIT_FOR_START_GUARD
                    request_guard_check = True

            elif key == ord(" "):
                if state == RecordState.WAIT_FOR_USER_STOP:
                    state = RecordState.SAVE_EPISODE
                elif state == RecordState.WAIT_FOR_USER_START:
                    # Recheck at the start key so a post-pass manual move cannot bypass the gate.
                    guard_result = check_start_guard(
                        bus, expected_start,
                        mode=start_guard_mode,
                        tolerance=tolerance,
                    )
                    if not guard_result.passed:
                        guard_passed = False
                        state = RecordState.WAIT_FOR_START_GUARD
                        request_guard_check = False
                    elif not args.dry_run and is_frame_black(global_frame):
                        print("  [WARN] Global camera frame is black. Check camera connection before recording.")
                        print(f"  Selected device: {global_cam.device_id if global_cam else 'none'}")
                        guard_passed = True
                    else:
                        guard_passed = True
                        active_start_guard = guard_result
                        if not args.dry_run:
                            if dataset is None:
                                print("  Creating LeRobot dataset after operator start key ...")
                                dataset = create_or_resume_dataset(args)
                                active_episode_id = int(getattr(dataset, "num_episodes", 0))
                            collector.clear_dataset_buffer(dataset)
                        prev_qpos = None
                        operator_start_time = iso_now()
                        state = RecordState.RECORDING
                        print(f"  >>> Recording episode {active_episode_id} started.")
                else:
                    print("  [WARN] START GUARD PASS is required before SPACE can start recording. Press C to check.")

            elif key in (10, 13):
                if state == RecordState.WAIT_FOR_USER_STOP:
                    state = RecordState.SAVE_EPISODE

            if state == RecordState.SAVE_EPISODE:
                operator_stop_time = iso_now()
                if args.dry_run:
                    dry_run_episodes += 1
                    print(f"  DRY RUN: episode {dry_run_episodes} stop flow passed; dataset not written.")
                else:
                    n_frames = collector.dataset_buffer_size(dataset)
                    if dataset is None or n_frames < MIN_SAVE_FRAMES:
                        collector.clear_dataset_buffer(dataset)
                        print("  Episode has no saved frames; discarded.")
                    else:
                        dataset.save_episode()
                        metadata_path = save_episode_start_metadata(
                            Path(args.dataset_root),
                            episode_id=active_episode_id,
                            result=active_start_guard,
                            fps=args.fps,
                            camera_key=args.camera_key,
                            operator_start_time=operator_start_time,
                            operator_stop_time=operator_stop_time,
                            start_pose_file=args.start_pose_file,
                            start_guard_mode=start_guard_mode,
                        )
                        print(f"  Saved episode {active_episode_id} ({n_frames} frames)")
                        print(f"  Saved start metadata: {metadata_path}")
                        saved_episodes_this_run += 1
                prev_qpos = None
                operator_start_time = ""
                active_start_guard = None
                completed = dry_run_episodes if args.dry_run else saved_episodes_this_run
                if completed >= args.num_episodes:
                    print(f"  Requested episode count reached: {completed}/{args.num_episodes}.")
                    break
                print("  Next episode requires a fresh start guard check.")
                state = RecordState.NEXT_EPISODE_START_GUARD
                request_guard_check = True
                guard_passed = False

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if dataset is not None:
            print("  Finalizing LeRobot dataset ...")
            try:
                dataset.finalize()
            except Exception as exc:
                print(f"  [WARN] dataset finalize failed: {exc}")
        print("  Arm stays ENABLED when adapter-v2 recorder exits.")
        bus.disconnect()
        if global_cam is not None:
            global_cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
