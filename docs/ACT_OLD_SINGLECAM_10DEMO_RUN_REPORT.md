# ACT Old Single-Camera 10-Demo Run Report

Run date: 2026-05-21  
Route: ACT old single-camera fixed-position Piper bottle baseline

## 1. Dataset path

```text
data/lerobot_dataset_piper_bottle_old_singlecam_10demo/
```

## 2. Camera key

```text
observation.images.global_rgb
```

The trained policy input features are:

```text
observation.state
observation.images.global_rgb
```

No wrist camera is present in this checkpoint.

## 3. Dataset sanity result

Result: PASS

- Episodes: 10
- Passed: 10
- Failed: 0
- Frames: 1793
- FPS: 10
- State shape: 7
- Action shape: 7
- Camera frames: `observation.images.global_rgb`, 1793 frames
- Gripper starts open around `0.0994-0.0995`
- Gripper closes to about `0.0472-0.0508`
- Lift phase heuristic present in all 10 episodes

## 4. Training command

```bash
DEVICE=cuda bash scripts/train_act_old_singlecam_10demo.sh \
  > logs/train_act_old_singlecam_10demo_20260521_093000.log 2>&1
```

The script reran dataset sanity check before training and refused CPU fallback.

## 5. Training start/end time

- Start: 2026-05-21 09:29:59 CST
- End: 2026-05-21 09:38:47 CST

## 6. Training exit code

```text
0
```

## 7. Log path

```text
logs/train_act_old_singlecam_10demo_20260521_093000.log
```

## 8. Output directory

```text
outputs/train/act_old_singlecam_10demo/
```

## 9. Checkpoint path

Final checkpoint:

```text
outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model/
```

Intermediate checkpoints:

```text
outputs/train/act_old_singlecam_10demo/checkpoints/001000/pretrained_model/
outputs/train/act_old_singlecam_10demo/checkpoints/002000/pretrained_model/
```

Final checkpoint files present:

- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`
- normalizer / unnormalizer processor weights
- `train_config.json`

## 10. Whether CUDA was used

Yes.

- GPU: NVIDIA GeForce RTX 3060
- PyTorch: `2.10.0+cu128`
- CUDA reported by PyTorch: `12.8`
- Training config policy device: `cuda`

## 11. Whether training finished successfully

Yes.

Final metric line:

```text
step:3K smpl:24K ep:134 epch:13.39 loss:0.092 grdn:1.812 lr:3.0e-04
```

Log scan:

- NaN: not found
- Traceback: not found
- OOM: not found
- CUDA error: not found
- RuntimeError: not found
- DataLoader error: not found

## 12. Offline evaluation result

Result: PASS

Offline eval log:

```text
logs/offline_eval_act_old_singlecam_10demo_20260521_094000.log
```

Checks performed:

- Loaded final checkpoint.
- Loaded old single-camera 10-demo dataset.
- Verified camera key matches `observation.images.global_rgb`.
- Verified no wrist camera is required by the policy.
- Ran policy forward on sampled frames from all 10 episodes.
- Verified action chunk shape is `(1, 10, 7)`.
- Verified output action dimension is 7.
- Verified no NaN/Inf in predicted actions.
- Verified gripper prediction shows open-to-close trend in all 10 episodes.

Offline statistics:

```text
sampled predictions: (363, 7)
mse_all:      0.0006537631
mse_arm:      0.0007618266
mse_gripper:  0.0000053820
pred gripper range: 0.04620 - 0.10134
true gripper range: 0.04720 - 0.09950
```

Per-episode gripper trend:

```text
10 / 10 episodes PASS
```

## 13. Errors/warnings

Warnings:

- `torchcodec` is unavailable, so LeRobot fell back to `pyav` for video decoding.
- Torchvision video decoding emits a deprecation warning.
- A first offline smoke-test attempt failed only because the diagnostic script tried to JSON-serialize LeRobot `PolicyFeature` objects. The checkpoint had loaded successfully. The corrected offline smoke test passed.

No training error was found.

## 14. Next recommended step

Do not run the robot automatically.

Next manual evaluation should be staged, using the commands prepared in:

```text
docs/ACT_OLD_SINGLECAM_10DEMO_EVAL.md
```

Manual order:

1. Approach
2. Close
3. Lift
4. Release

Only after approach, close, lift, and release pass should a full real-robot attempt be considered.

