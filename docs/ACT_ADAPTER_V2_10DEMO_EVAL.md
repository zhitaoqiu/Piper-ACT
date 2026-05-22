# ACT Adapter v2 10-Demo Real-Robot Staged Evaluation

Date: 2026-05-22
Branch: `piper-lerobot-adapter-v2`

## Dataset and Checkpoint

| Item | Path |
|---|---|
| Dataset | `data/lerobot_dataset_piper_bottle_adapter_v2_10demo/` |
| Training output | `outputs/train/act_adapter_v2_10demo/` |
| Selected checkpoint | `outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model/` |
| Camera | `observation.images.global_rgb` (single, /dev/video6) |
| Start pose file | `config/adapter_v2_start_pose.json` |

Training result: loss 0.590 → 0.108 over 3000 steps, sanity check 10/10 PASS.

## Hardware Configuration

| Setting | Value |
|---|---|
| CAN port | can0 |
| Global camera | /dev/video6 (5MP USB Camera) |
| Wrist camera | none (`--no-wrist`) |
| Control Hz | 10 |
| Chunk execution | target_reached |
| Target tolerance | 0.04 rad |

The `--enforce-start-reset` flag reads the dataset first-frame qpos from the
checkpoint's `train_config.json`, which points to the adapter v2 10-demo dataset.
This means the start guard compares against the actual recorded start pose,
**not** the old baseline `STANDARD_START_QPOS`.

## Evaluation Order (strict staging)

```
1. approach  →  2. close  →  3. lift  →  4. release  →  5. full
```

**Do not skip stages. Do not run full directly.** Each stage must pass before
proceeding to the next. If a stage fails, inspect logs and images before retrying.

Go/No-Go per stage: 2/3 attempts must pass. If a stage cannot reach 2/3, stop
and diagnose before continuing.

## Common Flags (all stages)

```
--policy-type act-full
--checkpt outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model
--test-mode full-e2e
--hz 10
--no-wrist
--global-camera /dev/video6
--open-gripper-on-start
--enforce-start-reset
--act-full-chunk-exec target_reached
--act-full-target-tol 0.04
--save-rollout --save-final-images
--hold-after-stop 8 --no-auto-return
--debug-actions --debug-policy-io
--allow-real-full-e2e
```

---

## Pre-Flight: Reset Check (run once before Stage A)

Verifies the robot can reach the recorded start pose. This is read-only
diagnostics — it moves to start, checks alignment, prints result, and exits
**before** any policy rollout.

```bash
conda activate piper_act

python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model \
  --reset-to-recorded-start \
  --open-gripper-on-start \
  --gripper-start-open-value 0.0995 \
  --can-port can0 \
  --allow-real-full-e2e
```

Expected output:
```
Reset guard expected start: [X.XXXX, X.XXXX, ...]
[SUCCESS] Start reset verified.
```

If this fails (joint differences > 0.05 rad or gripper > 0.01 m), manually move
the arm closer to the start pose and retry.

---

## Stage A: Approach

Policy moves the arm toward the bottle. Gripper stays open. Stops on
target-reached (all joints within 0.04 rad of chunk final) or max steps.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after approach \
  --hz 10 \
  --approach-steps 220 \
  --no-wrist \
  --global-camera /dev/video6 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --can-port can0 \
  --allow-real-full-e2e
```

### Stage A Pass Criteria

- [ ] Gripper aligns with the bottle position
- [ ] J2 height is reasonable (not too high, not crashing into table)
- [ ] No collision with table or fixture
- [ ] Gripper stays open throughout
- [ ] Global camera view matches dataset recording perspective
- [ ] `debug-actions` output shows plausible joint targets (not NaN, not extreme)

### Stage A Fail — What to Inspect

1. `logs/rollouts/test_a_*/final/global.jpg` — does the camera see the bottle?
2. `logs/rollouts/test_a_*/final/approach_alignment_debug.json` — stop reason, final qpos
3. `logs/rollouts/test_a_*/step_log.csv` — did any joint hit a limit?
4. Check `debug-policy-io` output — are observation/action values in expected ranges?

---

## Stage B: Close

Approach then close gripper on bottle. Stops after strong close detected
(gripper drops below close-strong-threshold).

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after close \
  --hz 10 \
  --approach-steps 260 \
  --no-wrist \
  --global-camera /dev/video6 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --can-port can0 \
  --allow-real-full-e2e
```

### Stage B Pass Criteria

- [ ] Approach phase still passes
- [ ] Bottle is between the fingers before gripper closes
- [ ] Gripper closes on the bottle (not in empty air)
- [ ] Gripper does not reopen after close
- [ ] Bottle remains held at the stop pose
- [ ] `approach_alignment_debug.json` shows `strong_close_detected: true`

### Stage B Fail — What to Inspect

1. Rollout images — is the gripper aligned with the bottle before closing?
2. `step_log.csv` — check gripper column: did it close? At what step?
3. If gripper closes in air: Stage A approach position may be off. Re-run Stage A.
4. If gripper never closes: check gripper close detection thresholds in `debug-actions` output.

---

## Stage C: Lift

Approach → close → lift. Stops 30 steps after strong close detected.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after lift \
  --hz 10 \
  --approach-steps 300 \
  --no-wrist \
  --global-camera /dev/video6 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --can-port can0 \
  --allow-real-full-e2e
```

### Stage C Pass Criteria

- [ ] Approach and close both still pass
- [ ] Bottle visibly leaves the table surface
- [ ] Bottle does not drop immediately after lift
- [ ] Robot arm remains stable (no oscillation)
- [ ] Global camera final image shows bottle elevated

### Stage C Fail — What to Inspect

1. `step_log.csv` — does J2 increase after close (lifting motion)?
2. If bottle drops: check gripper value in step_log. Did gripper loosen?
3. If no lift motion: check if close detection triggered too early (before full grip).
4. Rollout images — review global camera frames around the close→lift transition.

---

## Stage D: Release

Approach → close → lift → release. Stops after release onset detected
(gripper reopens above release threshold after confirmed strong close).

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after release \
  --release-stop-min-steps 10 \
  --hz 10 \
  --approach-steps 360 \
  --no-wrist \
  --global-camera /dev/video6 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --can-port can0 \
  --allow-real-full-e2e
```

### Stage D Pass Criteria

- [ ] Approach, close, and lift all still pass
- [ ] Release onset detected after confirmed strong close
- [ ] Gripper visibly reopens enough to release the bottle
- [ ] Bottle is placed (not dropped from height)
- [ ] Robot arm remains stable at stop pose

### Stage D Fail — What to Inspect

1. `approach_alignment_debug.json` — `release_onset_detected`, `strong_close_detected` flags
2. `step_log.csv` — gripper column: does it rise after the lift phase?
3. If release never triggers: check `--release-stop-min-steps` and release onset threshold.
4. Final images — does the bottle end up in the intended place location?

---

## Stage E: Full

Full trajectory with no early stop. Only run after Stages A–D all pass.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after full \
  --hz 10 \
  --approach-steps 360 \
  --no-wrist \
  --global-camera /dev/video6 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-global-video --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --can-port can0 \
  --allow-real-full-e2e
```

### Full Pass Criteria

- [ ] All staged criteria met
- [ ] Complete trajectory: approach → close → lift → release
- [ ] Bottle ends at release location (not dropped mid-trajectory)
- [ ] Global video shows smooth, plausible motion
- [ ] 2/3 full attempts succeed end-to-end

---

## Rollout Outputs

Each attempt creates a directory under `logs/rollouts/test_a_YYYYMMDD_HHMMSS/`:

| File | Content |
|---|---|
| `global_*.jpg` | Global camera frames (every 20 steps) |
| `step_*.npz` | robot_state, raw_action, sent_target per step |
| `step_log.csv` | Per-step: qpos, action, gripper phase, close flags, arm error |
| `final/global.jpg` | Final global camera image |
| `final/approach_alignment_debug.json` | Stop reason, final qpos, close/release flags |
| `global_view.mp4` | Full video (only with `--save-global-video`) |

---

## Fallback: Checkpoint Sweep Plan

If checkpoint `003000` shows degraded behavior (jittery motion, poor alignment,
collapse), sweep earlier checkpoints in this order:

### 1. Try 002000

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_adapter_v2_10demo/checkpoints/002000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after approach \
  --hz 10 --approach-steps 220 \
  --no-wrist --global-camera /dev/video6 \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --can-port can0 \
  --allow-real-full-e2e
```

### 2. Try 001000

Same command, replacing `002000` with `001000`.

### 3. If all checkpoints fail

- Check offline eval first: run `python3 scripts/check_policy_collapse.py --checkpt <path> --dataset-root data/lerobot_dataset_piper_bottle_adapter_v2_10demo --episode 0`
- Compare against one-demo overfit checkpoint: `outputs/train/act_adapter_v2_one_demo/checkpoints/01000/pretrained_model/`
- Possible causes:
  - Camera placement mismatch vs dataset recording
  - Start pose drift (re-capture with `scripts/adapter_v2_capture_start_pose.py`)
  - Dataset quality issue in specific episodes
- If one-demo checkpoint works but 10-demo doesn't: individual episode may have bad data. Run per-episode eval to isolate.

### 4. Offline eval before retraining

```bash
python3 inference/eval.py \
  --checkpt outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model \
  --dataset-root data/lerobot_dataset_piper_bottle_adapter_v2_10demo \
  --dataset-repo-id piper/adapter_v2_10demo \
  --episodes 0
```
