# ACT Old Single-Camera 10-Demo Eval

This route uses the old fixed-position single-camera demos first.

## Dataset And Checkpoint

Dataset:

```text
data/lerobot_dataset_piper_bottle_old_singlecam_10demo/
```

Training output:

```text
outputs/train/act_old_singlecam_10demo/
```

Expected checkpoint:

```text
outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model/
```

Camera key:

```text
observation.images.global_rgb
```

Do not use the wrist camera for this baseline. Training and deployment must stay single-camera.

## Before Training

```bash
python3 scripts/select_old_singlecam_10demo.py

python3 scripts/check_pilot_dataset.py \
  --dataset data/lerobot_dataset_piper_bottle_old_singlecam_10demo/ \
  --expected-episodes 10 \
  --min-pass-episodes 8 \
  --require-single-camera \
  --camera-key observation.images.global_rgb
```

Train:

```bash
bash scripts/train_act_old_singlecam_10demo.sh
```

The training script defaults to `DEVICE=cuda` and refuses CPU fallback. If another training job is using the GPU, wait and run this later instead of starting a CPU run.

## Reset Before Each Eval Stage

Run reset-only first. It moves to recorded start and exits before policy rollout.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model \
  --reset-to-recorded-start \
  --open-gripper-on-start \
  --gripper-start-open-value 0.0995 \
  --allow-real-full-e2e
```

## Stage A: Approach

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after approach \
  --hz 10 \
  --approach-steps 220 \
  --no-wrist \
  --global-camera auto \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --allow-real-full-e2e
```

Pass criteria:

- Gripper aligns with the fixed-position bottle.
- Height is reasonable.
- No collision.
- Global camera view matches the old dataset view.

## Stage B: Close

Only run after Stage A passes.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after close \
  --hz 10 \
  --approach-steps 260 \
  --no-wrist \
  --global-camera auto \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --allow-real-full-e2e
```

Pass criteria:

- Bottle is between fingers before close.
- Gripper closes on the bottle, not in air.
- Gripper does not reopen.
- Object remains held at stop pose.

## Stage C: Lift

Only run after Stage B passes.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after lift \
  --hz 10 \
  --approach-steps 300 \
  --no-wrist \
  --global-camera auto \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --allow-real-full-e2e
```

Pass criteria:

- Approach and close both pass.
- Bottle leaves the table.
- Bottle does not drop immediately.
- Robot remains stable.

Do not run full directly. If this fails, inspect old data quality, camera placement mismatch, and deploy camera mismatch before changing model family.
