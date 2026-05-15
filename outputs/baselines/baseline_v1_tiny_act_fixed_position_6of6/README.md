# Baseline v1 — Tiny ACT Fixed Position 6/6 Success

**Date**: 2026-05-15
**Status**: FROZEN — do not modify

## Model

- Checkpoint: `outputs/train/piper_bottle_approach_tiny_1ep/checkpoints/003000/pretrained_model`
- Policy: ACT, chunk_size=1, n_action_steps=1
- Architecture: dim_model=128, dim_feedforward=512, n_heads=4, n_encoder_layers=2, n_decoder_layers=2
- ~11M parameters
- Training data: 1 episode approach-only (from start_pose to bottle approach)
- Absolute action mode
- dropout=0, use_vae=false

## Deployment Parameters (v0.7.0)

```python
GRIPPER_OPEN = 0.08          # fully open (m)
GRIPPER_CLOSE = 0.0          # fully closed (m)
MAX_DELTA_PER_JOINT = np.array([0.03, 0.03, 0.03, 0.012, 0.012, 0.012])  # rad
ACTION_SMOOTH_ALPHA = 0.5    # EMA smoothing
APPROACH_STEPS_DEFAULT = 200
WRIST_FREEZE_J2 = 1.45       # freeze J4-J6 when J2 exceeds this
READY_J2 = 1.50              # J2 threshold for ready_count
READY_COUNT_MIN = 5          # consecutive steps to trigger stop
STAGNATION_STEPS = 20
STAGNATION_THRESHOLD = 0.0008
PIPER_GRIPPER_MAX_M = 0.101
```

## Test D Pipeline (scripted post-approach)

```
approach (ACT, ~175 steps) → ready stop (J2 > 1.50 ×5)
  ↓
hold position (0.3s)
  ↓
close gripper: interpolate to GRIPPER_CLOSE, hold 0.6s
  ↓
lift: J3 -= 0.06 rad, hold 0.5s
  ↓
place: J1 += 0.30 rad (to side)
  ↓
release: interpolate grip to GRIPPER_OPEN, hold 0.5s
  ↓
return: interpolate to start_pose (captured at script startup)
```

## Safety Mechanisms

- Per-joint independent delta clamping (not proportional)
- Wrist freeze @ J2 > 1.45
- Gripper forced OPEN during ACT approach
- Stagnation detection (20 steps no progress)
- Keyboard interrupt → hold position, no torque disable
- Disconnect does NOT disable torque

## 6/6 Success Results (2026-05-14)

| Run | J2 end | Steps | Grasp | Place | Return |
|-----|--------|-------|:-----:|:-----:|:------:|
| 1   | 1.5403 | 179   | ✓     | ✓     | ✓      |
| 2   | 1.5415 | 174   | ✓     | ✓     | ✓      |
| 3   | 1.5422 | 174   | ✓     | ✓     | ✓      |
| 4   | 1.5392 | 176   | ✓     | ✓     | ✓      |
| 5   | 1.5389 | 179   | ✓     | ✓     | ✓      |
| 6   | 1.5441 | 178   | ✓     | ✓     | ✓      |

J2 mean: 1.5410 ± 0.003 rad
Success rate: 100%

## Files in this baseline

- `pretrained_model/` — Tiny ACT 3k checkpoint (full)
- `deploy.py` — v0.7.0 deployment script
- `train_act_approach_1ep_tiny.sh` — training script
- `README.md` — this document
