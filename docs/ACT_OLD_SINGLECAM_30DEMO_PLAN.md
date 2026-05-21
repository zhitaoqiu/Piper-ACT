# ACT Old Single-Camera 30-Demo Clean Baseline — Plan

## Objective

Train an ACT baseline on ~30 clean old single-camera fixed-position demos.

## Dataset

Source:
```text
/home/huatec/piper_diffusion_bottle_grasp-master/data/lerobot_dataset_env2_30fixed
```
40 episodes total. 10 FPS. Single camera (`observation.images.global_rgb`).

Filtered target:
```text
data/lerobot_dataset_piper_bottle_old_singlecam_30demo_clean/
```

## Selection Criteria

- successful grasp with clear open-to-close gripper transition
- gripper starts open around 0.0995 (±0.012)
- gripper reaches close range 0.045–0.058
- release/reopen phase after close
- no NaN/Inf
- no all-zero actions
- no missing camera frames
- single camera only (`observation.images.global_rgb`)

## Selection Script

```bash
python3 scripts/select_old_singlecam_30demo_clean.py --dry-run   # preview only
python3 scripts/select_old_singlecam_30demo_clean.py              # actual export
```

## Sanity Check

```bash
python3 scripts/check_pilot_dataset.py \
  --dataset data/lerobot_dataset_piper_bottle_old_singlecam_30demo_clean/ \
  --expected-episodes 30 \
  --min-pass-episodes 25 \
  --require-single-camera \
  --camera-key observation.images.global_rgb
```

## Training

```bash
bash scripts/train_act_old_singlecam_30demo.sh
```

Or with explicit logging:
```bash
DEVICE=cuda bash scripts/train_act_old_singlecam_30demo.sh \
  > logs/train_act_old_singlecam_30demo_$(date +%Y%m%d_%H%M%S).log 2>&1
```

Training config:
- Policy: ACT, chunk_size=10, n_action_steps=10
- dim_model=128, n_heads=4, n_encoder_layers=2, n_decoder_layers=2
- lr=3e-4, batch_size=8, steps=5000
- Save/checkpoint every 1000 steps
- Device: cuda (RTX 3060), no CPU fallback

## Staged Real-Robot Evaluation

Do not run the robot automatically. Manual staged evaluation:

### Reset before each stage

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --reset-to-recorded-start \
  --open-gripper-on-start \
  --no-wrist --global-camera auto \
  --allow-real-full-e2e
```

### Stage A: Approach

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after approach \
  --hz 10 --approach-steps 220 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

### Stage B: Close

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after close \
  --hz 10 --approach-steps 260 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

### Stage C: Lift

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after lift \
  --hz 10 --approach-steps 300 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

### Stage D: Release

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after release \
  --release-stop-min-steps 10 --hz 10 --approach-steps 360 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

### Full Attempt

Only after all staged checks pass.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after full \
  --hz 10 --approach-steps 360 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-global-video --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

## Comparison Baseline

The previous 10-demo baseline is at:
```text
outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model/
```

Do not overwrite it. Keep both for comparison.
