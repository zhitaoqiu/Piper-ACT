# Phase Report — Piper ACT Mock Bridge Verification

**Date**: 2026-06-09  
**Branch**: `piper-lerobot-adapter-v2`  
**Status**: all gates passed

---

## 1. New: `ros_bridge/` mock bridge layer (630 lines)

Full ACT action → ROS topic → safety gate → mock driver → joint_states
closed-loop validation without real hardware.

| File | Lines | Role |
|------|-------|------|
| `common.py` | 80 | Shared constants: joint names, limits, MAX_DELTA, mock initial qpos, UDP config |
| `act_udp_action_client.py` | 45 | stdlib-only UDP client (conda side), sends 7D action JSON |
| `udp_to_ros_action_node.py` | 87 | ROS2 node: UDP listener → `/policy/target_joint_raw` |
| `safety_gate_node.py` | 139 | ROS2 safety gate: dim check, NaN/Inf filter, joint-limit clip, per-step MAX_DELTA clip → `/piper/command_joint_safe` |
| `mock_piper_driver_node.py` | 73 | ROS2 mock driver: subscribes safe command, updates internal state, 50Hz `/piper/joint_states` |
| `mock_piper_robot.py` | 57 | MockPiperRobot (conda side), mimics PiperRobot interface, sends action via UDP |
| `piper_sdk_state_node.py` | 148 | **New**: read-only real Piper SDK state bridge — reads joint states over CAN, NEVER sends control commands |

## 2. Safety design verification (4-step audit, all passed)

1. **Disconnect risk audit** — `sdk_adapter.disconnect()` only calls `DisableArm` when `_enabled==True`; this node never enables. Safe.
2. **Static boundary check** — all SDK imports gated behind `--allow-real-read` flag; zero control function call paths exist.
3. **Real CAN read-only test** — connected to `can0`, published real joint states at 30Hz, zero control commands triggered.
4. **Shadow verification** — safety gate correctly clips against REAL joint state. J2 delta +0.030 (exact MAX_DELTA match), gripper -0.004 (exact match).

## 3. Inference entrypoint safety hardening

`inference/deploy.py` and `inference/deploy_diffusion.py`:
- `--control-backend` (direct_sdk / ros_mock), default `direct_sdk`
- `--obs-backend` (real / mock), default `real`; mock mode skips RealSense, injects zero arrays
- All `hardware.piper_wrapper.PiperRobot` imports moved to lazy imports inside `direct_sdk` branch only

## 4. Cleanup

- Removed old `adapter_v2/` (migrated to `piper_driver/`)
- Removed 20+ backup config files, old export data, old docs, old training scripts
- `training/train.sh` rewritten for modern `lerobot-train` CLI with env-var overrides

## 5. New Cube training assets

- `configs/train_cube_64_dual_d256.json` — 64-demo dual-view training config
- `data/cube_64_dual/` — merged Cube dataset metadata (parquet/video gitignored)
- `docs/CUBE_DATASET_PREPARATION.md` — dataset health check report
- `docs/CUBE_TRAINING_CAMPAIGN.md` — training campaign plan
- `scripts/prepare_training_datasets.py` — dataset merge tool
- `scripts/record_demo.py`, `reset_arm.sh`, `view_cameras.sh` — utility scripts

---

**Stats**: 646 files changed, +2526 / −23757 lines
