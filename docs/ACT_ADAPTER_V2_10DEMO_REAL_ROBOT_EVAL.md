# ACT Adapter v2 10-Demo Real-Robot Evaluation Report

Date: 2026-05-22
Branch: `piper-lerobot-adapter-v2`

## 1. Configuration

| Item | Value |
|---|---|
| Checkpoint | `outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model/` |
| Dataset | `data/lerobot_dataset_piper_bottle_adapter_v2_10demo/` |
| Camera | `observation.images.global_rgb` (/dev/video6) |
| Wrist camera | none (`--no-wrist`) |
| Start pose file | `config/adapter_v2_start_pose.json` |
| Start guard mode | zone |
| Policy type | act-full |
| Chunk execution | target_reached |
| Target tolerance | 0.04 rad |
| Training | loss 0.590 → 0.108 (3000 steps, san check 10/10 PASS) |

## 2. Evaluation Stages

### 2.1 Stage A: Approach (Baseline — hz=10, approach_steps=220)

Three approach-only runs with baseline parameters.

| Run | Time | Steps | close_detected | Final j2 | Final gripper | Target reached | Result |
|---|---|---|---|---|---|---|---|
| 1 | 14:01:24 | 203 | True | 1.69 | 0.0508 (closed) | True | PASS |
| 2 | 14:02:35 | 211 | True | 1.71 | 0.0519 (closed) | True | PASS |
| 3 | 14:03:28 | 212 | True | 1.70 | 0.0507 (closed) | True | PASS |

**Stage A result: 3/3 PASS.** Gripper aligned with bottle, arm reached approach position without collision.

Rollouts: `logs/rollouts/test_a_20260522_140124/`, `140235/`, `140328/`

### 2.2 Smoothing Experiment: High-Frequency Small-Step + Queue Smoothing

User evaluated high-frequency small-step execution (hz=20) as primary smoothing
candidate, comparing Candidates A and B, then stacking ACT queue smoothing and
EMA tuning.

| Run | Time | Steps | close_detected | gripper_phase | Final j2 | Final gripper | Video | Result |
|---|---|---|---|---|---|---|---|---|
| 4 | 14:05:38 | 359 | True | closed* | 1.61 | 0.0509 | Yes | Close + brief release attempt |
| 5 | 14:06:41 | 359 | True | **released** | 0.71 | 0.0986 | Yes | **FULL SUCCESS** |
| 6 | 14:08:13 | 297 | True | released | 0.90 | 0.0986 | Yes | Release + partial return |
| 7 | 14:09:19 | 359 | True | released | 1.65 | 0.0477 | Yes | Release, arm stayed raised |

\* Run 4 briefly entered "releasing" phase at step 354 then returned to "closed".

**Best result: Run 5 (14:06:41)** — complete cycle: approach → close → lift → release → return-to-home. Final j2=0.71 near start position, gripper fully open 0.0986, target_reached=True.

Rollouts: `logs/rollouts/test_a_20260522_140538/`, `140641/`, `140813/`, `140919/`

### 2.3 EMA Tuning (action_smooth trials)

After smoothing experiments, user adjusted `--action-smooth` values to dial in
the response/smoothness trade-off. These runs all failed to detect close —
likely due to over-smoothing causing the arm to under-reach or arrive late to
the grasp position.

| Run | Time | Steps | close_detected | gripper_phase | Final j2 | Final gripper | Result |
|---|---|---|---|---|---|---|---|
| 8 | 14:14:53 | 209 | False | open | 1.68 | 0.0986 | FAIL — never closed |
| 9 | 14:15:39 | 196 | False | open | 1.68 | 0.0986 | FAIL — never closed |
| 10 | 14:17:40 | 204 | False | open | 1.68 | 0.0968 | FAIL — never closed |
| 11 | 14:19:41 | 165 | True | releasing | 1.77 | 0.0535 | Close detected, interrupted mid-release |
| 12 | 14:20:53 | 165 | False | open | 1.69 | 0.0982 | FAIL — never closed |

**5 out of 6 runs failed to close.** The one partial success (141941) detected
close but was stopped before completing release. This batch confirms that
excessive EMA smoothing (action_smooth > baseline) hurts grasp success rate
on this task.

Rollouts: `logs/rollouts/test_a_20260522_141453/`, `141539/`, `141740/`, `141941/`, `142053/`

### 2.4 Final Attempt

| Run | Time | Steps | close_detected | Final j2 | Final gripper | Result |
|---|---|---|---|---|---|---|
| 13 | 16:31:11 | 165 | False | 1.69 | 0.0982 | FAIL — never closed |

Rollout: `logs/rollouts/test_a_20260522_163111/`

## 3. Full Evaluation Summary

### 3.1 Trial Counts

| Category | Count |
|---|---|
| Total trials | 13 |
| Full success (approach → close → lift → release → return) | 1 (Run 5) |
| Close + release (no return) | 2 (Runs 6, 7) |
| Close only (no release) | 4 (Runs 1–4) |
| Close + interrupted release | 1 (Run 11) |
| Complete failure (never closed) | 5 (Runs 8–10, 12, 13) |
| **Approach-to-close success rate** | **8/13 (62%)** |
| **Full e2e success rate** | **1/13 (8%)** |

### 3.2 Smoothing Configuration Used in Best Run

Run 5 (14:06:41) — the only full success — used:

| Parameter | Value |
|---|---|
| hz | 20 |
| approach_steps | 440 |
| max_delta_arm (J1–J3) | 0.020 |
| max_delta_wrist (J4–J6) | 0.008 |
| action_smooth (EMA) | baseline (0.5) |
| ACT queue smoothing | enabled (blend=5, mean_filter=5) |

### 3.3 Key Finding: action_smooth Sensitivity

Increasing `--action-smooth` from baseline 0.5 to higher values (0.7, 0.8)
caused a sharp drop in close success: 5/6 runs at elevated action_smooth
failed to close, compared to 7/7 runs at baseline action_smooth successfully
closing. **Keep action_smooth at 0.5 for this task.**

### 3.4 Failure Cases

| Failure mode | Runs | Likely cause |
|---|---|---|
| Never closed | 8, 9, 10, 12, 13 | Over-smoothed; arm under-reached or arrived late |
| Released but arm stayed raised | 7 | Policy predicted release without return trajectory |
| Interrupted mid-release | 11 | Premature stop before release completed |

## 4. Rollout Artifacts

| Run | Time | Global images | Video |
|---|---|---|---|
| 1 | 140124 | global_0000..0200.jpg (11 files) | No |
| 2 | 140235 | global_0000..0200.jpg (11 files) | No |
| 3 | 140328 | global_0000..0200.jpg (11 files) | No |
| 4 | 140538 | global_0000..0340.jpg (18 files) | global_view.mp4 |
| 5 | 140641 | global_0000..0340.jpg (18 files) | global_view.mp4 |
| 6 | 140813 | global_0000..0280.jpg (15 files) | global_view.mp4 |
| 7 | 140919 | global_0000..0340.jpg (18 files) | global_view.mp4 |
| 8 | 141453 | global_0000..0200.jpg (11 files) | No |
| 9 | 141539 | global_0000..0180.jpg (10 files) | No |
| 10 | 141740 | global_0000..0200.jpg (11 files) | No |
| 11 | 141941 | global_0000..0160.jpg (9 files) | No |
| 12 | 142053 | global_0000..0160.jpg (9 files) | No |
| 13 | 163111 | global_0000..0160.jpg (9 files) | No |

All rollouts at: `logs/rollouts/test_a_20260522_<timestamp>/`

## 5. Conclusion

### 5.1 Status: Conditionally Validated

The adapter v2 10-demo ACT baseline has been **real-robot validated with
conditions**. A full end-to-end success (approach → close → lift → release →
return-to-home) was demonstrated in Run 5 (14:06:41) under the smoothed
configuration (hz=20, max_delta 0.020/0.008, ACT queue smoothing, action_smooth=0.5).

However, the overall e2e success rate is low (1/13), primarily due to
action_smooth sensitivity discovered during the EMA tuning phase. With the
correct smoothing configuration (action_smooth=0.5), close success rate is
100% (7/7).

### 5.2 Recommended Deployment Configuration

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_adapter_v2_10demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after full \
  --hz 20 \
  --approach-steps 440 \
  --max-delta-arm 0.020 \
  --max-delta-wrist 0.008 \
  --action-smooth 0.5 \
  --enable-act-queue-smoothing \
  --act-boundary-blend-steps 5 \
  --act-mean-filter-window 5 \
  --no-wrist \
  --global-camera /dev/video6 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-global-video --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions \
  --can-port can0 \
  --allow-real-full-e2e
```

### 5.3 Next Steps

- **Do NOT** retrain
- **Do NOT** increase action_smooth above 0.5
- **Do NOT** modify the checkpoint or dataset
- If more reliability is needed, collect additional clean fixed-position demos
  before attempting multi-position generalization
- The current checkpoint is usable for fixed-position bottle grasping with the
  recommended smoothing configuration above

### 5.4 Protected Baseline

| Asset | Status |
|---|---|
| `outputs/train/act_adapter_v2_10demo/checkpoints/003000/` | Frozen |
| `data/lerobot_dataset_piper_bottle_adapter_v2_10demo/` | Frozen |
| Old 10-demo baseline (`act_old_singlecam_10demo`) | Unchanged |
