# ACT Old Single-Camera 30-Demo — Checkpoint Sweep (Approach-Only)

Date: 2026-05-21

## Issue

The 30-demo `last` checkpoint (step 5000, MSE 0.000609) produces near-zero actions on the real robot. The arm barely moves, gripper stays open, no approach behavior. The 10-demo checkpoint (step 3000, MSE 0.000654) moves normally.

**Hypothesis**: cross-attention collapse — the model learned to ignore the image and output near-constant actions. Lower MSE does not mean a better policy.

## Dataset Start-Pose Analysis

### 10-demo (10 episodes)

| Metric | J1 | J2 | J3 | J4 | J5 | J6 | Grip |
|--------|----|----|----|----|----|----|------|
| Ep0 start | +0.063 | +0.008 | -0.004 | +0.027 | +0.309 | -0.098 | 0.0995 |
| Mean | +0.010 | +0.007 | -0.004 | +0.020 | +0.324 | -0.021 | 0.0994 |
| Std | 0.020 | 0.001 | 0.000 | 0.020 | 0.006 | 0.042 | 0.000 |
| Min | -0.009 | +0.005 | -0.004 | -0.012 | +0.309 | -0.098 | 0.0994 |
| Max | +0.063 | +0.008 | -0.003 | +0.052 | +0.330 | +0.029 | 0.0995 |

### 30-demo (30 episodes)

| Metric | J1 | J2 | J3 | J4 | J5 | J6 | Grip |
|--------|----|----|----|----|----|----|------|
| Ep0 start | +0.012 | +0.008 | -0.003 | -0.017 | +0.323 | -0.017 | 0.0994 |
| Mean | -0.007 | +0.009 | -0.004 | +0.019 | +0.324 | -0.017 | 0.0992 |
| Std | **0.028** | **0.010** | 0.001 | **0.044** | **0.010** | **0.063** | 0.001 |
| Min | **-0.098** | +0.005 | -0.005 | **-0.048** | +0.293 | **-0.149** | 0.0946 |
| Max | +0.063 | **+0.061** | -0.001 | **+0.146** | +0.333 | **+0.093** | 0.0995 |

### Key Findings

1. **30-demo contains all 10 of the 10-demo episodes** — the 30-demo set is a superset: 10 known episodes + 20 new episodes
2. **30-demo Ep0 is a NEW episode** (J1=+0.012), not the 10-demo Ep0 (J1=+0.063). The 10-demo Ep0 became 30-demo Ep1
3. **30-demo has 2-3x the start-pose variance** of 10-demo — especially J1 (0.028 vs 0.020 std), J4 (0.044 vs 0.020), J6 (0.063 vs 0.042)
4. **The 20 new episodes have extreme start poses** — J1 to -0.098, J4 to +0.146, J6 to -0.149
5. The 10-demo episodes are a more **homogeneous** subset, which may have helped ACT learn a consistent policy

### Ep0 Mismatch

The `--enforce-start-reset` resets to the dataset's first-frame (Ep0) position:

| | 10-demo Ep0 (=30demo Ep1) | 30-demo Ep0 |
|---|---|---|
| J1 | **+0.063** | **+0.012** |
| J4 | **+0.027** | **-0.017** |
| J6 | **-0.098** | **-0.017** |

The robot resets to a different position for each dataset. Both are valid starting positions, but the 30-demo model sees this position less frequently (Ep0 is only 1 of 30).

---

## Checkpoint Sweep Commands

All checkpoints: `outputs/train/act_old_singlecam_30demo/checkpoints/`

### Checkpoint 002000 (MSE 0.001987)

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/002000/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after approach \
  --hz 10 --approach-steps 220 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

### Checkpoint 003000 (MSE 0.000945)

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after approach \
  --hz 10 --approach-steps 220 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

### Checkpoint 001000 (MSE 0.003974)

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/001000/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after approach \
  --hz 10 --approach-steps 220 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

### Checkpoint 004000 (MSE 0.000712)

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/004000/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after approach \
  --hz 10 --approach-steps 220 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

### Checkpoint last/005000 (MSE 0.000609) — REFERENCE ONLY

Already tested. Near-zero actions, no approach. Do not re-run.

---

## Recording Template

For each checkpoint, record:

| Field | Value |
|-------|-------|
| Checkpoint | |
| Steps completed | |
| Stop reason | |
| Near-zero actions? | |
| J2 range in chunks | |
| J3 range in chunks | |
| Grip range in chunks | |
| Any close_candidate? | |
| Max arm delta | |
| Final qpos | |
| Final gripper | |
| Visually approaches bottle? | |
| Resembles 10-demo behavior? | |

---

## 10-Demo Reference (Successful Approach)

From successful 10-demo Stage A runs earlier today:

- Robot moves toward bottle
- J2/J3 have meaningful deltas (not near-zero)
- Gripper stays open during approach (correct)
- Close detected (grip drops below threshold)
- Typical run: 166-360 steps, gripper drops from ~0.099 to 0.050-0.067

10-demo checkpoint: `outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model/`

---

## Decision Rules

1. **If any 30-demo checkpoint moves normally** (meaningful J2/J3 deltas, arm approaches bottle) → use that checkpoint as the 30-demo candidate for staged eval
2. **If all 30-demo checkpoints collapse** (near-zero actions, no approach) → **fall back to the 10-demo checkpoint** (003000) for the Friday demo
3. **After the demo**: investigate 30-demo data/start-pose consistency and training collapse. Options:
   - Filter the 30-demo set to remove extreme start-pose outliers
   - Sort episodes so a known-good start is Ep0
   - Try curriculum training (10-demo first, then add diverse episodes)
   - Consider delta-mode actions instead of absolute
