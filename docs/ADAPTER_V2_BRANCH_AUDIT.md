# Adapter v2 Branch Audit

Date: 2026-05-22
Branch: `piper-lerobot-adapter-v2`
Audit type: static / read-only — no robot motion, no data collection, no training.

## 1. Baseline protection status

| Check | Status | Detail |
|---|---|---|
| Old deploy path untouched? | **PASS** | `inference/deploy.py` not modified by any adapter-v2 commit; `git diff act-10demo-success-before-migration..HEAD -- inference/deploy.py` is empty |
| Old training path untouched? | **PASS** | `training/` not modified |
| Old policies path untouched? | **PASS** | `policies/` not modified |
| Old config path untouched? | **PASS** | `config/` not modified |
| Old checkpoint 003000 exists? | **PASS** | `outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model/` (140MB, 5 files including model.safetensors) |
| Old checkpoints 001000, 002000 exist? | **PASS** | Both present |
| Old frozen backup exists? | **PASS** | `frozen_success/act_old_singlecam_10demo_success_checkpoint.tar.gz` (44MB) + `frozen_success/act_10demo_success_rollouts.tar.gz` (257MB) |
| Old dataset untouched? | **PASS** | `data/lerobot_dataset_piper_bottle_old_singlecam_10demo/` (32MB, 10 episodes, 1793 frames, features verified via info.json) |
| Tags exist? | **PASS** | `act-10demo-success-before-adapter-v2` and `act-10demo-success-before-migration` both present |

**Baseline protection: PASS** — No adapter-v2 commit touches any old deploy/training/policies/config path. The old dataset, checkpoint, and frozen backups are intact.

## 2. Files changed since baseline

All files changed relative to `act-10demo-success-before-adapter-v2` (17 files, net-new additions):

```
adapter_v2/__init__.py              NEW — registration surface
adapter_v2/piper_bus.py             NEW — SDK-backed motor bus
adapter_v2/piper_follower.py        NEW — LeRobot robot
adapter_v2/piper_leader.py          NEW — optional software leader scaffold
adapter_v2/reset.py                 NEW — confirmed reset helper
adapter_v2/schema.py                NEW — constants, conversions, safety limits
adapter_v2/start_pose.py            NEW — start-pose guard helpers
docs/PIPER_LEROBOT_ADAPTER_V2_MIGRATION.md   NEW — migration doc
docs/VA11HALL_PIPER_LEROBOT_LESSONS.md       UPDATED — lessons from VA11Hall reference
scripts/adapter_v2_check_start_pose.py       NEW — read-only start-pose check
scripts/adapter_v2_gripper_test.py           NEW — confirmed gripper sweep test
scripts/adapter_v2_smoke_read.py             NEW — read-only can0 smoke test
scripts/check_pilot_dataset.py               UPDATED — dataset sanity checks
scripts/record_adapter_v2.py                 NEW — guarded single-cam recorder
scripts/record_adapter_v2_mirror.py           NEW — mirror-flow alias
scripts/replay_adapter_v2.py                 NEW — replay entrypoint
teleop/data_collector.py                     UPDATED — helper functions exported for adapter_v2 reuse
```

**Impact on old deploy path: NONE.** All changes are additive (new modules/scripts) or additive exports in `teleop/data_collector.py`. No existing file was rewritten.

## 3. Adapter v2 architecture status

### 3.1 schema.py — PASS

| Check | Status |
|---|---|
| qpos/action schema = [j1...j6, gripper] | PASS (line 9-11: `JOINT_NAMES` + `MOTOR_NAMES`, 7 keys in `MOTOR_POS_KEYS`) |
| state dim = 7, action dim = 7 | PASS (line 13: `STATE_DIM = 7`) |
| joint units = rad | PASS (implicit in all conversions; `joint_limit_rad` naming) |
| gripper units = meters | PASS (line 18-21: `GRIPPER_OPEN_M = 0.0995`, `GRIPPER_STRONG_CLOSE_MIN_M = 0.045`, `GRIPPER_STRONG_CLOSE_MAX_M = 0.055`) |
| STANDARD_START_QPOS defined | PASS (lines 25-28) |
| safety limits defined | PASS (line 32: `DEFAULT_MAX_DELTA_RAD`; line 21: `PIPER_GRIPPER_MAX_M = 0.101`) |
| qpos_to_action / action_to_qpos round-trip safe | PASS (validate shape 7 + finite check in `as_qpos`) |

### 3.2 piper_bus.py — PASS

| Check | Status |
|---|---|
| Wraps existing PiperRobot | PASS (line 34: `self._robot = PiperRobot(...)`) |
| Read qpos (rad/meter) | PASS (line 65: `as_qpos(self._robot.get_joint_positions())`) |
| Read gripper (meters) | PASS (included in qpos[6]) |
| Send qpos target (rad/meter) | PASS (lines 67-76, clips joints to ±limit_rad, gripper to [0, 0.101]) |
| Send gripper target (meters) | PASS (target[6] clipped) |
| Does not silently change units | PASS (round-trips through as_qpos validation) |

### 3.3 piper_follower.py — PASS

| Check | Status |
|---|---|
| LeRobot-compatible Robot | PASS (extends `lerobot.robots.robot.Robot`) |
| Registered as `piper_follower_v2` | PASS (line 40, `@RobotConfig.register_subclass("piper_follower_v2")`) |
| Exposes observation.state | PASS (lines 100-107: `get_observation` reads qpos + cameras) |
| Exposes observation.images.global_rgb | PASS (cameras from config, read in `get_observation`) |
| Exposes action features | PASS (lines 73-75: `_motors_ft` = 7 float keys) |
| Uses can0 by default | PASS (line 26: `can_port: str = "can0"`) |

### 3.4 piper_leader.py — PASS (scaffold)

| Check | Status |
|---|---|
| Marked as optional software leader scaffold | PASS (docstring lines 1-5: "Optional software-leader"; `connect()` raises ValueError if no `can_port` with pointer to mirror flow) |
| Not required for current mirror path | PASS (`record_adapter_v2.py` does not import `PiperLeaderV2`) |

### 3.5 reset.py — PASS

| Check | Status |
|---|---|
| Requires manual confirmation | PASS (line 46: `if not confirmed: raise PermissionError(...)`) |
| Does not move robot without explicit operator confirm | PASS (caller `run_confirmed_reset` in record_adapter_v2.py:177-186 requires typing "RESET") |
| Opens gripper safely | PASS (gripper moves through interpolated steps, not instantaneous) |
| Keeps arm enabled | PASS (reset does not call `bus.disable()`) |

### 3.6 start_pose.py — PASS

| Check | Status |
|---|---|
| Compares current qpos with STANDARD_START_QPOS | PASS (line 19-20: `qpos_diff` + `start_pose_guard`) |
| Prints expected / current / abs diff | PASS (done in `check_start_guard` in record_adapter_v2.py:76-88) |
| Enforces tolerance | PASS (line 20: arm ±0.05 rad default, gripper ±0.01 m default) |
| Fails closed if out of tolerance | PASS (returns False → state stays WAIT_FOR_START_GUARD) |

## 4. Record pipeline status

| Check | Status |
|---|---|
| Active path is mirror flow, not PiperLeaderV2 | **PASS** — `record_adapter_v2_mirror.py` forwards to `record_adapter_v2.py`; PiperLeaderV2 requires explicit `--teleop.can_port` and is not imported in the recorder |
| Hardware topology documented (follower can0 only) | **PASS** — documented in migration doc lines 45-58 |
| Dataset format = LeRobotDataset | **PASS** — via `collector.create_or_resume_dataset` → `LeRobotDataset.create()` |
| observation.state feature present | **PASS** (line 336: `"observation.state"`) |
| action feature present | **PASS** (line 337) |
| observation.images.global_rgb feature present | **PASS** (line 339: `args.camera_key` = GLOBAL_CAMERA_KEY) |
| Single camera only, no wrist camera | **PASS** (line 250: "global camera only, no wrist camera") |
| camera_key = `observation.images.global_rgb` | **PASS** (schema.py line 16, validated in argparse line 229-230) |
| FPS configurable, default 10Hz | **PASS** (`--fps` argument, default=10) |
| State dim = 7, action dim = 7 | **PASS** |
| Episode metadata saved | **PASS** (`save_episode_start_metadata` writes JSON to `meta/adapter_v2_episode_metadata/`) |
| Reset motion NOT recorded | **PASS** — reset only happens in states WAIT_FOR_START_GUARD/NEXT_EPISODE_START_GUARD; recording starts only from RECORDING state; reset blocked during recording (line 384-385) |

## 5. Start guard status

12-point checklist against `record_adapter_v2.py`:

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Read current qpos/gripper | **PASS** | check_start_guard:71 reads `bus.read_qpos()` |
| 2 | Compare with STANDARD_START_QPOS | **PASS** | start_pose_guard:19-20 uses qpos_diff |
| 3 | Print expected qpos | **PASS** | line 77: `fmt_qpos(result.expected)` |
| 4 | Print current qpos | **PASS** | line 78: `fmt_qpos(result.current)` |
| 5 | Print abs diff | **PASS** | line 79: `fmt_qpos(result.diff)` |
| 6 | arm_start_tol default ~0.05 rad | **PASS** | QposTolerance.arm_rad = 0.05 (schema.py:37), argparse default 0.05 (record_adapter_v2.py:211) |
| 7 | gripper_start_tol default ~0.010 m | **PASS** | QposTolerance.gripper_m = 0.01 (schema.py:38), argparse default 0.010 (record_adapter_v2.py:212) |
| 8 | Refuse recording if not passed | **PASS** | line 311-313: state stays WAIT_FOR_START_GUARD on fail |
| 9 | SPACE rechecks before starting | **PASS** | lines 397-401: recheck at SPACE in WAIT_FOR_USER_START state |
| 10 | Every episode rechecks, not just first | **PASS** | lines 452-454: SAVE_EPISODE → NEXT_EPISODE_START_GUARD → recheck |
| 11 | Reset motion not recorded | **PASS** | `R` key blocked during recording (line 384-385) |
| 12 | User keypress start, user keypress stop | **PASS** | SPACE to start (line 396), SPACE/ENTER to stop (lines 394-395, 417-419) |

**State machine** (verified in RecordState enum):
```
WAIT_FOR_START_GUARD → START_GUARD_PASS → WAIT_FOR_USER_START → RECORDING
  → WAIT_FOR_USER_STOP → SAVE_EPISODE → NEXT_EPISODE_START_GUARD → (loop)
```
All transitions verified in the main loop. No bypass path exists.

## 6. Safety status

| Check | Status |
|---|---|
| No automatic arm movement by default | **PASS** — smoke_read, check_start_pose, replay (requires manual start pose) are read-only or explicit-start |
| Reset requires typing "RESET" | **PASS** — `confirm_reset()` lines 178-186: required input is literal `"RESET"` |
| Gripper test has human confirmation for each phase | **PASS** — `confirm()` requires "YES" before open, close, and reopen |
| Exit does not unexpectedly disable torque | **PASS** — `record_adapter_v2.py:469`: "Arm stays ENABLED when adapter-v2 recorder exits."; gripper_test also prints same |
| Max_delta / joint limits / gripper clamp active | **PASS** — `piper_bus.py:69-70`: `np.clip` on joints and gripper; `schema.py:32`: DEFAULT_MAX_DELTA_RAD defined |
| No auto multi-episode recording | **PASS** — each episode requires manual SPACE start; reached episode count triggers break |
| No shortcut to bypass start guard | **PASS** — `--require-start-guard` is always True; start guard is the only path from boot to RECORDING |

## 7. Scripts inventory

| Script | Exists | Compiles | --help | Type | Status |
|---|---|---|---|---|---|
| `adapter_v2_smoke_read.py` | YES | PASS | PASS | read-only | **READY** |
| `adapter_v2_gripper_test.py` | YES | PASS | PASS | confirmed motion | **READY** (needs operator YES) |
| `adapter_v2_check_start_pose.py` | YES | PASS | PASS | read-only | **READY** |
| `adapter_v2_reset_to_start.py` | **MISSING** | — | — | — | **REMOVED** (intentionally deleted in 6e49ed5; functionality consolidated into `adapter_v2/reset.py` and called by `record_adapter_v2.py` `R` key) |
| `record_adapter_v2.py` | YES | PASS | PASS | guarded record | **READY** (dry-run available) |
| `replay_adapter_v2.py` | YES | PASS | PASS | replay (motion) | **READY** (requires manual start alignment first) |
| `record_adapter_v2_mirror.py` | YES | PASS | PASS | alias | **READY** (forwards to record_adapter_v2.py) |

Note: `adapter_v2_reset_to_start.py` was a standalone script in the initial scaffold that was deleted in commit `6e49ed5` when the reset logic was consolidated into `adapter_v2/reset.py` and the `R` key in `record_adapter_v2.py`. This is **intentional** and not a regression.

## 8. Documentation consistency

`docs/PIPER_LEROBOT_ADAPTER_V2_MIGRATION.md` reviewed. Covers:

- Frozen baseline (checkpoint, backup, tags) — **YES**
- Current topology (follower can0, powered teaching arm mirror) — **YES**
- PiperLeaderV2 is scaffold only — **YES** (lines 45-58, 74-76)
- Active record path is mirror flow — **YES** (line 82: "current one-CAN guarded recorder")
- PiperFollowerV2 is LeRobot-style Robot — **YES** (line 73)
- Validation order: smoke → gripper → start_pose → dry-run → record → dataset check → replay — **YES** (Steps 1-6)
- Must not train before replay PASS — **YES** (line 200: "Do not train ACT until replay succeeds on the adapter-v2 data path.")

**Doc status: PASS** — Migration document is consistent with code and correctly describes the active path.

## 9. Remaining blockers before one-demo record

| Blocker | Severity | Detail |
|---|---|---|
| Robot not tested with adapter-v2 code | **BLOCKER** | All checks are static; PipermotorsBusV2 has never been run against real hardware via adapter_v2 path |
| Camera not confirmed | **BLOCKER** | `--global-camera auto` USB camera detection not validated with actual camera |
| Gripper test not executed | **SHOULD** | Gripper open/close safety validation pending |
| Start pose alignment not validated | **SHOULD** | `STANDARD_START_QPOS` from 10-demo baseline needs manual teaching-arm alignment verification |
| Dataset sanity script not run on adapter-v2 data | **N/A yet** | Cannot run before recording |

## 10. Final recommendation

### Read-only commands safe to execute NOW:

```bash
# All commands under piper_act conda env
conda activate piper_act

# Step 1: smoke test (reads qpos, sends NO motion)
python3 scripts/adapter_v2_smoke_read.py --can-port can0

# Step 2: start pose check (reads qpos, sends NO motion)
python3 scripts/adapter_v2_check_start_pose.py --can-port can0

# Step 3: dry-run recorder (connects Piper, reads qpos, checks guard, writes NO dataset)
python3 scripts/record_adapter_v2.py --can-port can0 --dry-run
```

### Confirmed-motion commands (require operator consent):

```bash
# Gripper test (asks YES before each phase)
python3 scripts/adapter_v2_gripper_test.py --can-port can0
```

### NOT ready (blocked until one-demo sanity + replay PASS):

| Action | Status |
|---|---|
| Record one demo | **OK to attempt** after smoke + start_pose + gripper pass; dry-run first |
| Replay one demo | **OK to attempt** after successful record + dataset sanity check; requires manual start alignment |
| Train ACT | **NO** — blocked until one-demo record → dataset sanity → replay all PASS |

### Overall branch health

**PASS** — The `piper-lerobot-adapter-v2` branch is well-structured, does not touch the old successful baseline, and has a complete guarded recording pipeline with thorough safety checks. The adapter v2 code is ready for read-only hardware smoke testing and operator-confirmed gripper testing. No training should be attempted until the full Stage 1 validation chain (smoke → gripper → start_pose → dry-run → record → dataset sanity → replay) passes.
